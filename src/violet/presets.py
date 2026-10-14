"""
Named configurations, as frozen dataclasses and as TOML.

A preset is data, not code. The built-in ones live in ``data/presets.toml``
alongside this module and are read with :mod:`tomllib` from the standard
library, exactly the same way a user's own file is — which is the point. There
is no privileged path for the presets that ship with the package, so anything
they can express, yours can too.

That is also where the base frequency lives. No module in this package
contains one, and a test enforces it: a frequency written into Python is a
frequency the package has an opinion about, and it should not have one.

The layer table
---------------
Each preset lists its layers as a table with a ``kind``::

    [[preset.layer]]
    kind = "pair"
    ratio = 1.0
    level = 0.22

Ratios are relative to the carrier — the base frequency shifted by whole
octaves into the binaural window — so a preset transposes correctly to any
base. The pedal is the exception and sounds the base itself, because being at
the literal frequency is its whole job.

Modulation
----------
A preset can also carry an amplitude modulation, which applies to the whole
mix::

    modulation = { hz = 16, depth = 0.4 }
    modulation = { hz = 16, bpm = 84, depth = 0.4 }   # locked to the tempo

See :mod:`violet.dsp.modulation`. A ``track`` layer brings in recorded music
from a file, with its path relative to the preset file that names it, and it
sets the preset's sample rate unless the preset gives one.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import soundfile as sf

from violet import tuning
from violet.dsp.curves import BeatCurve
from violet.dsp.env import Lfo, plan_swells
from violet.dsp.modulation import Modulation, tempo_locked
from violet.engine import LoopConfig, RenderConfig, measured_gain
from violet.harmony import WalkConfig, plan_loop_progression, plan_progression
from violet.layers import (
    Air,
    BinauralPair,
    ChordBed,
    Layer,
    Modulated,
    Ocean,
    Pedal,
    Track,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "BANDS",
    "BUILTIN_PRESETS",
    "AirSpec",
    "ChordsSpec",
    "LayerSpec",
    "OceanSpec",
    "PairSpec",
    "PedalSpec",
    "Preset",
    "PresetLibrary",
    "TrackSpec",
    "load_library",
    "resolve_beat",
    "resolve_modulation",
]

#: EEG band names, as a convenience for the beat rate. The names describe
#: measured brain activity; using one here says nothing about what the audio
#: does to it. See the README.
BANDS: dict[str, float] = {
    "delta": 2.0,
    "theta": 4.0,
    "schumann": 7.83,
    "alpha": 10.0,
    "beta": 16.0,
}

BUILTIN_PRESETS = Path(__file__).parent / "data" / "presets.toml"


def resolve_beat(
    beat: str | float | Mapping[str, Any] | BeatCurve,
) -> float | BeatCurve:
    """
    Turn ``"theta"``, ``4.0`` or a curve table into a beat rate.

    A table becomes a :class:`~violet.dsp.curves.BeatCurve`::

        beat = { shape = "exponential", at_minutes = [0, 8, 25], hz = [10, 10, 2] }

    Times are in minutes, because a forty-five minute descent written in
    seconds is a wall of four-digit numbers nobody can check by eye. The key
    says so, to keep it from being mistaken for seconds.
    """
    if isinstance(beat, BeatCurve):
        return beat
    if isinstance(beat, Mapping):
        table = dict(beat)
        shape = table.pop("shape", "linear")
        try:
            at_minutes = tuple(float(v) for v in table.pop("at_minutes"))
            rates = tuple(float(v) for v in table.pop("hz"))
        except KeyError as error:
            msg = f"a beat curve needs at_minutes and hz; missing {error}"
            raise ValueError(msg) from None
        if table:
            msg = f"unknown keys in beat curve: {sorted(table)}"
            raise ValueError(msg)
        return BeatCurve.from_minutes(at_minutes, rates, shape=shape)
    if isinstance(beat, int | float):
        return float(beat)
    name = beat.strip().lower()
    if name in BANDS:
        return BANDS[name]
    try:
        return float(name)
    except ValueError:
        known = ", ".join(BANDS)
        msg = f"unknown beat {beat!r}; give a frequency in Hz or one of: {known}"
        raise ValueError(msg) from None


def resolve_modulation(table: Mapping[str, Any] | Modulation) -> Modulation:
    """
    Turn a ``modulation`` table into a :class:`~violet.dsp.modulation.Modulation`.

    ``hz`` takes a number or a band name, the same as a beat. Give ``bpm`` and
    the rate is moved to the nearest even subdivision of that tempo — 16 Hz at
    84 bpm becomes 16.8, twelve pulses to the beat. Give ``at_minutes`` and
    ``hz`` becomes a list and the rate moves, exactly as a beat curve does.
    ``band_hz`` is the range that gets modulated; ``offset_seconds`` puts the
    top of the first pulse on the first beat of a track that does not start on
    one.
    """
    if isinstance(table, Modulation):
        return table
    fields = dict(table)
    if "hz" not in fields:
        msg = "a modulation table needs hz"
        raise ValueError(msg)
    hz = fields.pop("hz")
    bpm = fields.pop("bpm", None)

    rate: float | BeatCurve
    if "at_minutes" in fields:
        if bpm is not None:
            msg = (
                "a modulation curve cannot also be locked to a tempo: the "
                "subdivision would have to change as the rate moved"
            )
            raise ValueError(msg)
        rate = resolve_beat(
            {
                "at_minutes": fields.pop("at_minutes"),
                "hz": hz,
                "shape": fields.pop("shape", "linear"),
            }
        )
    else:
        if not isinstance(hz, str | int | float):
            msg = f"modulation hz must be one rate without at_minutes, got {hz!r}"
            raise ValueError(msg)
        fixed = resolve_beat(hz)
        assert isinstance(fixed, float)  # noqa: S101 - a scalar resolves to one
        rate = fixed if bpm is None else tempo_locked(fixed, float(bpm))

    depth = float(fields.pop("depth", 0.5))
    low, high = fields.pop("band_hz", (200.0, 1000.0))
    offset = float(fields.pop("offset_seconds", 0.0))
    if fields:
        msg = f"unknown keys in modulation: {sorted(fields)}"
        raise ValueError(msg)
    return Modulation(
        rate=rate, depth=depth, band=(float(low), float(high)), offset=offset
    )


# ---------------------------------------------------------------------------
# layer specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LfoSpec:
    """A slow swell on a layer."""

    period: float
    low: float = 0.55
    high: float = 1.0
    phase: float = 0.0

    def build(self) -> Lfo:
        """The envelope this describes."""
        return Lfo(period=self.period, low=self.low, high=self.high, phase=self.phase)


@dataclass(frozen=True, slots=True)
class PairSpec:
    """A binaural voice at ``ratio`` times the carrier."""

    level: float
    ratio: float = 1.0
    swell: LfoSpec | None = None
    kind: Literal["pair"] = "pair"


@dataclass(frozen=True, slots=True)
class PedalSpec:
    """The mono drone, at the base frequency itself."""

    level: float
    octaves: int = 0
    kind: Literal["pedal"] = "pedal"


@dataclass(frozen=True, slots=True)
class ChordsSpec:
    """A moving chord bed over the carrier."""

    level: float
    crossfade: float = 16.0
    min_seconds: float = 38.0
    max_seconds: float = 62.0
    kind: Literal["chords"] = "chords"

    def walk_for(self, seconds: float, *, looping: bool) -> WalkConfig:
        """
        The walk configuration for a render of this length.

        Chord lengths are shortened, in proportion, when a *looping* render is
        too short to hold two of them. A loop has to close, so two chords is
        the floor — one chord crossfading into itself is not a progression —
        and a five-second loop of forty-second chords is not a thing that
        exists. Ordinary renders never need this: they simply stop mid-chord,
        which is what the fade is for.
        """
        low, high, crossfade = self.min_seconds, self.max_seconds, self.crossfade
        if looping and seconds < low * 2.0:
            squeeze = seconds / (low * 2.0)
            low, high = low * squeeze, high * squeeze
        return WalkConfig(
            crossfade=min(crossfade, low),
            min_seconds=low,
            max_seconds=high,
        )


@dataclass(frozen=True, slots=True)
class OceanSpec:
    """The decorrelated noise bed, notched at the carrier."""

    level: float
    q: float = 2.0
    dark_hz: float = 520.0
    bright_hz: tuple[float, float] = (900.0, 6500.0)
    bright_mix: float = 0.55
    set_seconds: tuple[float, float] = (85.0, 125.0)
    seed: int = 2027
    kind: Literal["ocean"] = "ocean"


@dataclass(frozen=True, slots=True)
class AirSpec:
    """The quiet mono noise floor."""

    level: float
    cutoff_hz: float = 141.7974
    seed: int = 7
    kind: Literal["air"] = "air"


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """
    Recorded music from a file.

    In a preset file the path is relative to that file, so a preset and the
    music it names can move around together. Leave the path out and the preset
    has a slot for music rather than music: ``--music`` fills it.
    """

    path: Path | None = None
    level: float = 1.0
    repeat: bool = True
    kind: Literal["track"] = "track"

    @property
    def sample_rate(self) -> int:
        """The file's own rate, which a render containing it has to use."""
        if self.path is None:
            msg = "this track has no file yet; give one with --music"
            raise FileNotFoundError(msg)
        if not self.path.is_file():
            msg = f"no audio file at {self.path}"
            raise FileNotFoundError(msg)
        return int(sf.info(str(self.path)).samplerate)


LayerSpec = PairSpec | PedalSpec | ChordsSpec | OceanSpec | AirSpec | TrackSpec

_SPECS: dict[str, type[LayerSpec]] = {
    "pair": PairSpec,
    "pedal": PedalSpec,
    "chords": ChordsSpec,
    "ocean": OceanSpec,
    "air": AirSpec,
    "track": TrackSpec,
}


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Preset:
    """Everything needed to render, before the command line has its say."""

    name: str
    description: str
    base_hz: float

    #: A fixed rate in Hz, or a curve that moves over the render.
    beat: float | BeatCurve
    layers: tuple[LayerSpec, ...]

    sample_rate: int = 44100
    minutes: float = 20.0
    seed: int = 5
    gain: float | None = None
    ceiling: float = 0.89
    fade_seconds: float = 8.0
    fade_denominator: int = 4
    block_seconds: float = 10.0
    loop: bool = False
    loop_crossfade: float = 12.0

    #: An amplitude modulation over the whole mix, or none.
    modulation: Modulation | None = None

    def __post_init__(self) -> None:
        if self.base_hz <= 0.0:
            msg = f"{self.name}: base frequency must be positive"
            raise ValueError(msg)
        if isinstance(self.beat, float | int) and self.beat < 0.0:
            msg = f"{self.name}: beat must not be negative"
            raise ValueError(msg)
        if self.loop and isinstance(self.beat, BeatCurve) and self.beat.moves:
            msg = (
                f"{self.name}: a moving beat and a seamless loop do not go "
                f"together. The rate at the end of the curve "
                f"({self.beat.end_hz:g} Hz) is not the rate at the start "
                f"({self.beat.start_hz:g} Hz), so however smooth the waveform is "
                f"at the join, the pulse jumps. Pick one."
            )
            raise ValueError(msg)
        if self.loop and self.modulation is not None:
            msg = (
                f"{self.name}: a modulated mix cannot be closed into a seamless "
                f"loop. The modulation's band filter carries memory that no "
                f"rounding makes periodic, so the whole mix would have to be "
                f"crossfaded at the join, and crossfading a tone against a "
                f"phase-shifted copy of itself comb-filters it. Pick one."
            )
            raise ValueError(msg)
        if self.minutes <= 0.0:
            msg = f"{self.name}: minutes must be positive"
            raise ValueError(msg)
        if not self.layers:
            msg = f"{self.name}: a preset with no layers renders silence"
            raise ValueError(msg)

    @property
    def carrier_hz(self) -> float:
        """The base frequency, octave-shifted into the binaural window."""
        return tuning.carrier_for(self.base_hz)

    @property
    def seconds(self) -> float:
        """Length of the render."""
        return self.minutes * 60.0

    def with_overrides(
        self,
        *,
        minutes: float | None = None,
        beat: float | BeatCurve | None = None,
        base_hz: float | None = None,
        seed: int | None = None,
        loop: bool | None = None,
        block_seconds: float | None = None,
        modulation: Modulation | None = None,
        music: Path | None = None,
    ) -> Preset:
        """
        A copy with the command line's say applied. ``None`` means "leave it".

        Deliberately a fixed set rather than arbitrary keywords: these are the
        things worth changing from a command line, and anything else belongs in
        a preset file where it can be kept.

        ``music`` puts a track under the preset: it replaces any track the
        preset already has, joins the other layers otherwise, and moves the
        render to the file's own sample rate.
        """
        changes: dict[str, object] = {
            "minutes": minutes,
            "beat": beat,
            "base_hz": base_hz,
            "seed": seed,
            "loop": loop,
            "block_seconds": block_seconds,
            "modulation": modulation,
        }
        if music is not None:
            slot = next(
                (spec for spec in self.layers if isinstance(spec, TrackSpec)), None
            )
            track = TrackSpec(path=music) if slot is None else replace(slot, path=music)
            others = tuple(s for s in self.layers if not isinstance(s, TrackSpec))
            changes["layers"] = (track, *others)
            changes["sample_rate"] = track.sample_rate
        wanted = {key: value for key, value in changes.items() if value is not None}
        return replace(self, **wanted) if wanted else self  # type: ignore[arg-type]

    @property
    def tests_something(self) -> bool:
        """Whether there is a beat or a modulation for a trial to switch off."""
        fastest = self.beat.span[1] if isinstance(self.beat, BeatCurve) else self.beat
        active = self.modulation is not None and self.modulation.active
        return fastest > 0.0 or active

    def placebo(self) -> Preset:
        """
        The same preset with everything under test switched off.

        No beat and no modulation depth, and nothing else changed: same seed,
        same progression, same waves, same music. A beat of zero leaves both
        ears identical; a depth of zero leaves the mix untouched, sample for
        sample. The modulation's rate and band stay put, so the two presets
        still describe the same thing apart from whether it is switched on.
        """
        modulation = (
            None if self.modulation is None else replace(self.modulation, depth=0.0)
        )
        return replace(self, beat=0.0, modulation=modulation)

    def treatment(self) -> str:
        """What a trial of this preset compares against nothing, in words."""
        parts: list[str] = []
        if isinstance(self.beat, BeatCurve):
            parts.append(f"beat {self.beat.describe()}")
        elif self.beat > 0.0:
            parts.append(f"{self.beat:g} Hz beat")
        if self.modulation is not None and self.modulation.active:
            parts.append(f"modulation {self.modulation.describe()}")
        return " + ".join(parts) or "nothing"

    @property
    def fold_seconds(self) -> float:
        """
        The loop crossfade, capped at half the render.

        A twelve-second fold is right for a five-minute loop and absurd for a
        ten-second one, where it would be most of the file. The engine refuses
        a fold longer than the render; this keeps a short ``--minutes`` from
        walking into that.
        """
        return min(self.loop_crossfade, self.seconds / 2.0)

    def render_config(self) -> RenderConfig:
        """The engine configuration this preset asks for."""
        return RenderConfig(
            sample_rate=self.sample_rate,
            duration=self.seconds,
            block_seconds=self.block_seconds,
            gain=self.gain,
            ceiling=self.ceiling,
            fade_seconds=0.0 if self.loop else self.fade_seconds,
            fade_max_denominator=self.fade_denominator,
            loop=LoopConfig(crossfade=self.fold_seconds) if self.loop else None,
        )

    def settled_config(self, layers: Sequence[Layer]) -> RenderConfig:
        """
        :meth:`render_config`, with the master gain decided once and for all.

        A gain given in the preset is used as it is. Otherwise an unmodulated
        mix keeps the analytic bound the engine already derives, and a
        modulated one is *measured*: rendered once at unity gain, and
        attenuated only as far as the peak it actually reached requires. The
        analytic bound for a modulated mix assumes the band filter rings to its
        limit against the full signal at the bottom of every pulse, and on a
        mastered track that would cost the best part of 6 dB for nothing.

        ``layers`` must be the ones about to be rendered; they are reset after
        the measuring pass, by the engine, like any other render.
        """
        config = self.render_config()
        if self.gain is not None or self.modulation is None:
            return config
        return replace(config, gain=measured_gain(layers, config))

    def build_layers(self) -> list[Layer]:
        """
        Turn the specifications into layers.

        One generator, seeded once, feeds every layer that needs randomness, in
        the order the layers are listed. That is what makes a seed mean
        something: it names the whole piece, not one part of it.
        """
        rng = np.random.default_rng(self.seed)
        carrier = self.carrier_hz
        layers: list[Layer] = []

        for spec in self.layers:
            match spec:
                case PairSpec():
                    layers.append(
                        BinauralPair(
                            carrier=carrier * spec.ratio,
                            beat=self.beat,
                            level=spec.level,
                            swell=None if spec.swell is None else spec.swell.build(),
                        )
                    )
                case PedalSpec():
                    layers.append(
                        Pedal(
                            freq=self.base_hz * 2.0**spec.octaves,
                            level=spec.level,
                        )
                    )
                case ChordsSpec():
                    walk = spec.walk_for(self.seconds, looping=self.loop)
                    events = (
                        plan_loop_progression(self.seconds, rng, walk)
                        if self.loop
                        else plan_progression(self.seconds, rng, walk)
                    )
                    layers.append(
                        ChordBed(
                            events=events,
                            root=carrier,
                            beat=self.beat,
                            level=spec.level,
                            crossfade=walk.crossfade,
                        )
                    )
                case OceanSpec():
                    swells = plan_swells(self.seconds, rng)
                    set_period = float(rng.uniform(*spec.set_seconds))
                    layers.append(
                        Ocean(
                            swells=swells,
                            set_period=set_period,
                            notch_hz=carrier,
                            level=spec.level,
                            sample_rate=self.sample_rate,
                            seed=spec.seed,
                            q=spec.q,
                            dark_hz=spec.dark_hz,
                            bright_hz=spec.bright_hz,
                            bright_mix=spec.bright_mix,
                        )
                    )
                case AirSpec():
                    layers.append(
                        Air(
                            level=spec.level,
                            sample_rate=self.sample_rate,
                            cutoff_hz=spec.cutoff_hz,
                            seed=spec.seed,
                        )
                    )
                case TrackSpec():
                    if spec.path is None:
                        msg = (
                            f"{self.name} has a slot for music and nothing in "
                            f"it; give a file with --music"
                        )
                        raise ValueError(msg)
                    layers.append(
                        Track(path=spec.path, level=spec.level, repeat=spec.repeat)
                    )

        if self.modulation is not None:
            return [
                Modulated(
                    layers=tuple(layers),
                    modulation=self.modulation,
                    sample_rate=self.sample_rate,
                )
            ]
        return layers


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _layer_from_table(
    table: Mapping[str, Any], preset: str, folder: Path | None = None
) -> LayerSpec:
    fields = dict(table)
    kind = fields.pop("kind", None)
    if kind not in _SPECS:
        known = ", ".join(_SPECS)
        msg = f"{preset}: layer kind {kind!r} is not one of: {known}"
        raise ValueError(msg)

    try:
        if kind == "track" and "path" in fields:
            path = Path(fields["path"])
            fields["path"] = path if folder is None else folder / path
        if kind == "pair" and "swell" in fields:
            fields["swell"] = LfoSpec(**fields["swell"])
        for key in ("bright_hz", "set_seconds"):
            if key in fields:
                low, high = fields[key]
                fields[key] = (float(low), float(high))
        return _SPECS[kind](**fields)
    except TypeError as error:
        msg = f"{preset}: bad {kind} layer — {error}"
        raise ValueError(msg) from None


def _preset_from_table(
    name: str, table: Mapping[str, Any], folder: Path | None = None
) -> Preset:
    fields = dict(table)
    layer_tables = fields.pop("layer", [])
    if not layer_tables:
        msg = f"{name}: no [[{name}.layer]] tables"
        raise ValueError(msg)

    if "beat" in fields:
        fields["beat"] = resolve_beat(fields["beat"])
    if "modulation" in fields:
        try:
            fields["modulation"] = resolve_modulation(fields["modulation"])
        except ValueError as error:
            msg = f"{name}: {error}"
            raise ValueError(msg) from None

    layers = tuple(_layer_from_table(one, name, folder) for one in layer_tables)
    tracks = [spec for spec in layers if isinstance(spec, TrackSpec)]
    # A missing file is left for build time to report, so one moved track
    # does not stop the rest of a preset file from loading.
    first = tracks[0].path if tracks else None
    if first is not None and "sample_rate" not in fields and first.is_file():
        fields["sample_rate"] = tracks[0].sample_rate

    try:
        return Preset(name=name, layers=layers, **fields)
    except TypeError as error:
        msg = f"{name}: bad preset — {error}"
        raise ValueError(msg) from None


@dataclass(frozen=True, slots=True)
class PresetLibrary:
    """The presets available to render, by name."""

    presets: dict[str, Preset] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Preset:
        try:
            return self.presets[name]
        except KeyError:
            known = ", ".join(sorted(self.presets))
            msg = f"unknown preset {name!r}; available: {known}"
            raise KeyError(msg) from None

    def __contains__(self, name: str) -> bool:
        return name in self.presets

    def __iter__(self) -> Iterator[Preset]:
        return iter(self.presets.values())

    def __len__(self) -> int:
        return len(self.presets)

    @property
    def names(self) -> list[str]:
        """Preset names, in the order they were defined."""
        return list(self.presets)


def load_library(*paths: str | Path, builtin: bool = True) -> PresetLibrary:
    """
    Read presets from TOML, later files overriding earlier ones by name.

    The built-in file is read first unless you say otherwise, so a user file
    can replace a shipped preset by reusing its name.
    """
    presets: dict[str, Preset] = {}
    files = ([BUILTIN_PRESETS] if builtin else []) + [Path(p) for p in paths]

    for path in files:
        if not path.is_file():
            msg = f"no preset file at {path}"
            raise FileNotFoundError(msg)
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        for name, table in document.items():
            if not isinstance(table, dict):
                msg = f"{path}: top-level {name!r} is not a preset table"
                raise ValueError(msg)  # noqa: TRY004 - a bad file, not a bad call
            presets[name] = _preset_from_table(name, table, path.parent)

    return PresetLibrary(presets=presets)
