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
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from violet import tuning
from violet.dsp.env import Lfo, plan_swells
from violet.engine import LoopConfig, RenderConfig
from violet.harmony import WalkConfig, plan_loop_progression, plan_progression
from violet.layers import Air, BinauralPair, ChordBed, Layer, Ocean, Pedal

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

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
    "load_library",
    "resolve_beat",
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


def resolve_beat(beat: str | float) -> float:
    """Turn ``"theta"`` or ``4.0`` into a beat frequency in Hz."""
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


LayerSpec = PairSpec | PedalSpec | ChordsSpec | OceanSpec | AirSpec

_SPECS: dict[str, type[LayerSpec]] = {
    "pair": PairSpec,
    "pedal": PedalSpec,
    "chords": ChordsSpec,
    "ocean": OceanSpec,
    "air": AirSpec,
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
    beat: float
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

    def __post_init__(self) -> None:
        if self.base_hz <= 0.0:
            msg = f"{self.name}: base frequency must be positive"
            raise ValueError(msg)
        if self.beat < 0.0:
            msg = f"{self.name}: beat must not be negative"
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
        beat: float | None = None,
        base_hz: float | None = None,
        seed: int | None = None,
        loop: bool | None = None,
        block_seconds: float | None = None,
    ) -> Preset:
        """
        A copy with the command line's say applied. ``None`` means "leave it".

        Deliberately a fixed set rather than arbitrary keywords: these are the
        things worth changing from a command line, and anything else belongs in
        a preset file where it can be kept.
        """
        changes: dict[str, object] = {
            "minutes": minutes,
            "beat": beat,
            "base_hz": base_hz,
            "seed": seed,
            "loop": loop,
            "block_seconds": block_seconds,
        }
        wanted = {key: value for key, value in changes.items() if value is not None}
        return replace(self, **wanted) if wanted else self  # type: ignore[arg-type]

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

        return layers


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _layer_from_table(table: Mapping[str, Any], preset: str) -> LayerSpec:
    fields = dict(table)
    kind = fields.pop("kind", None)
    if kind not in _SPECS:
        known = ", ".join(_SPECS)
        msg = f"{preset}: layer kind {kind!r} is not one of: {known}"
        raise ValueError(msg)

    if kind == "pair" and "swell" in fields:
        fields["swell"] = LfoSpec(**fields["swell"])
    for key in ("bright_hz", "set_seconds"):
        if key in fields:
            low, high = fields[key]
            fields[key] = (float(low), float(high))

    try:
        return _SPECS[kind](**fields)
    except TypeError as error:
        msg = f"{preset}: bad {kind} layer — {error}"
        raise ValueError(msg) from None


def _preset_from_table(name: str, table: Mapping[str, Any]) -> Preset:
    fields = dict(table)
    layer_tables = fields.pop("layer", [])
    if not layer_tables:
        msg = f"{name}: no [[{name}.layer]] tables"
        raise ValueError(msg)

    if "beat" in fields:
        fields["beat"] = resolve_beat(fields["beat"])

    try:
        return Preset(
            name=name,
            layers=tuple(_layer_from_table(one, name) for one in layer_tables),
            **fields,
        )
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
            presets[name] = _preset_from_table(name, table)

    return PresetLibrary(presets=presets)
