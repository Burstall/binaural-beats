"""
The error paths.

Every one of these is a message someone will read at three in the morning
wondering why their render stopped. They are worth having tested for the same
reason the happy path is: a validation branch that has never run is a branch
that might not raise, or might raise the wrong thing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from violet.cli import app
from violet.dsp.env import Lfo, fade_envelope, fade_length, slow_lfo
from violet.dsp.noise import KelletPink, OnePole, PinkNoise
from violet.dsp.osc import sine, sine_pair
from violet.harmony import (
    CHORDS,
    ChordEvent,
    WalkConfig,
    plan_loop_progression,
    tiles_loop,
)
from violet.layers import Air, BinauralPair, Ocean, Pedal, Span
from violet.presets import (
    AirSpec,
    ChordsSpec,
    LfoSpec,
    PairSpec,
    Preset,
    PresetLibrary,
    load_library,
)

if TYPE_CHECKING:
    from pathlib import Path

SR = 32000
ROOT = 335.796


# ---------------------------------------------------------------------------
# envelopes
# ---------------------------------------------------------------------------


def test_a_fade_needs_room_for_a_head_and_a_tail() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fade_length(44100, 44100, 8.0, max_denominator=1)


def test_a_zero_fade_leaves_the_signal_alone() -> None:
    envelope = fade_envelope(Span(0, 100, SR).indices, 100, 0)
    assert np.array_equal(envelope, np.ones(100))


def test_an_lfo_needs_a_positive_period() -> None:
    with pytest.raises(ValueError, match="period must be positive"):
        slow_lfo(np.zeros(4), period=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        Lfo(period=10.0).snapped_to_loop(0.0)


def test_an_lfo_snaps_onto_a_divisor_of_the_loop() -> None:
    snapped = Lfo(period=70.0, low=0.4, phase=1.7).snapped_to_loop(300.0)
    assert 300.0 / snapped.period == pytest.approx(round(300.0 / snapped.period))
    assert snapped.period == pytest.approx(75.0)
    assert snapped.low == 0.4
    assert snapped.phase == 1.7


def test_a_period_longer_than_the_loop_becomes_the_loop() -> None:
    """Rounding down to zero cycles would be a division by zero."""
    assert Lfo(period=500.0).snapped_to_loop(60.0).period == 60.0


# ---------------------------------------------------------------------------
# noise and filters
# ---------------------------------------------------------------------------


def test_pink_noise_reports_its_scale() -> None:
    assert PinkNoise(seed=1, scale=0.35).scale == 0.35


def test_a_one_pole_cutoff_has_to_be_a_frequency() -> None:
    with pytest.raises(ValueError, match="must sit between"):
        OnePole(SR, 0.0)
    with pytest.raises(ValueError, match="must sit between"):
        OnePole(SR, SR / 2.0)


def test_the_kellet_filter_restarts_cleanly() -> None:
    noise = KelletPink(seed=7, scale=0.06)
    first = noise.block(4000)
    noise.reset()
    assert np.array_equal(noise.block(4000), first)


def test_the_kellet_filter_is_pinkish() -> None:
    """Coarser than the three-pole fit, and still falling with frequency."""
    x = KelletPink(seed=7, scale=1.0).block(2**18)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    power = np.abs(np.fft.rfft(x)) ** 2

    def band(low: float, high: float) -> float:
        return float(power[(freqs >= low) & (freqs < high)].mean())

    for low in (200.0, 400.0, 800.0):
        drop = 10.0 * np.log10(band(low, low * 2) / band(low * 2, low * 4))
        assert 1.5 < drop < 5.0, (low, drop)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: BinauralPair(carrier=ROOT, beat=4.0, level=-0.1),
        lambda: Pedal(freq=83.949, level=-0.1),
        lambda: Air(level=-0.1, sample_rate=SR),
    ],
)
def test_no_layer_accepts_a_negative_level(build: object) -> None:
    with pytest.raises(ValueError, match="level must not be negative"):
        build()  # type: ignore[operator]


def test_a_negative_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        sine(np.zeros(4), freq=-1.0)
    with pytest.raises(ValueError, match="beat must not be negative"):
        sine_pair(np.zeros(4), freq=400.0, beat=-1.0)


def test_the_air_bed_refuses_a_span_out_of_order() -> None:
    """Same contract as the ocean: filter state cannot be seeked."""
    air = Air(level=0.06, sample_rate=SR)
    air.render(Span(0, 1000, SR))
    with pytest.raises(ValueError, match="must be rendered in order"):
        air.render(Span(9000, 10000, SR))
    air.reset()
    air.render(Span(0, 1000, SR))


def test_a_pedal_snaps_onto_whole_cycles() -> None:
    snapped, notes = Pedal(freq=83.949, level=0.09).snapped_to_loop(300.0)
    assert isinstance(snapped, Pedal)
    assert snapped.freq * 300.0 == pytest.approx(round(snapped.freq * 300.0))
    assert any("pedal" in note for note in notes)


def test_a_pedal_already_on_whole_cycles_reports_nothing() -> None:
    exact = 100.0  # 100 Hz over a 10-second loop is 1000 whole cycles
    snapped, notes = Pedal(freq=exact, level=0.09).snapped_to_loop(10.0)
    assert isinstance(snapped, Pedal)
    assert snapped.freq == exact
    assert notes == ()


def test_an_ocean_bound_scales_with_its_level() -> None:
    bed = Ocean(swells=(), set_period=90.0, notch_hz=ROOT, level=0.5, sample_rate=SR)
    assert bed.peak == pytest.approx(0.5 * bed.peak_estimate)


# ---------------------------------------------------------------------------
# harmony
# ---------------------------------------------------------------------------


def test_a_walk_that_cannot_get_home_says_so() -> None:
    """
    A loop needs a final chord that leads back to the first.

    Here nothing does — the only move is iv to itself — so the walk stops with
    an explanation rather than quietly producing a progression that jumps at
    the join.
    """
    walk = WalkConfig(
        crossfade=4.0,
        min_seconds=12.0,
        max_seconds=18.0,
        transitions={"i": (("iv", 1),), "iv": (("iv", 1),)},
        chords={"i": CHORDS["i"], "iv": CHORDS["iv"]},
    )
    with pytest.raises(ValueError, match="reaches any of"):
        plan_loop_progression(48.0, np.random.default_rng(1), walk)


def event(start: float, end: float, **flags: bool) -> ChordEvent:
    return ChordEvent(CHORDS["i"], start, end, **flags)


@pytest.mark.parametrize(
    ("events", "why"),
    [
        ((), "no events at all"),
        ((event(4.0, 60.0),), "does not start at zero"),
        ((event(0.0, 59.0),), "does not reach the end"),
        ((event(0.0, 60.0, fade_in=False),), "an edge that does not fade"),
        ((event(0.0, 30.0), event(31.0, 60.0)), "a gap in the middle"),
    ],
)
def test_a_progression_that_does_not_tile_is_not_a_loop(
    events: tuple[ChordEvent, ...], why: str
) -> None:
    assert not tiles_loop(events, 60.0), why


def test_a_progression_that_tiles_is_a_loop() -> None:
    assert tiles_loop((event(0.0, 30.0), event(30.0, 60.0)), 60.0)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def spec_preset(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "x",
        "description": "d",
        "base_hz": 100.0,
        "beat": 4.0,
        "layers": (PairSpec(level=0.2),),
    }
    return base | changes


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"base_hz": 0.0}, "base frequency must be positive"),
        ({"beat": -1.0}, "beat must not be negative"),
        ({"minutes": 0.0}, "minutes must be positive"),
        ({"layers": ()}, "renders silence"),
    ],
)
def test_a_preset_validates_itself(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Preset(**spec_preset(**changes))  # type: ignore[arg-type]


def test_a_top_level_scalar_is_not_a_preset(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a preset table"):
        load_library(path)


def test_a_bad_swell_table_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[x]\ndescription = "d"\nbase_hz = 100.0\nbeat = 4.0\n'
        '[[x.layer]]\nkind = "pair"\nlevel = 0.2\n'
        "swell = { period = 70.0, wobble = 3 }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad pair layer.*wobble"):
        load_library(path)


def test_the_library_behaves_like_a_collection() -> None:
    library = load_library()
    assert len(library) == len(library.names)
    assert "ocean" in library
    assert "trumpet" not in library
    assert [preset.name for preset in library] == library.names
    assert PresetLibrary().names == []


def test_a_chord_spec_keeps_its_lengths_for_an_ordinary_render() -> None:
    """Only a loop that cannot hold two chords gets them shortened."""
    spec = ChordsSpec(level=0.07)
    ordinary = spec.walk_for(10.0, looping=False)
    assert ordinary.min_seconds == spec.min_seconds
    assert ordinary.max_seconds == spec.max_seconds

    squeezed = spec.walk_for(20.0, looping=True)
    assert squeezed.min_seconds < spec.min_seconds
    assert squeezed.crossfade <= squeezed.min_seconds


def test_layer_specs_carry_their_kind() -> None:
    assert PairSpec(level=0.2).kind == "pair"
    assert ChordsSpec(level=0.07).kind == "chords"
    assert AirSpec(level=0.06).kind == "air"
    assert LfoSpec(period=70.0).build().period == 70.0


def test_the_fold_is_capped_at_half_the_render() -> None:
    preset = load_library()["ocean-loop"]
    assert preset.fold_seconds == preset.loop_crossfade
    assert preset.with_overrides(minutes=0.2).fold_seconds == 6.0


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_the_cli_reports_a_zero_beat_preset_honestly(tmp_path: Path) -> None:
    """`show` says there is no beat, rather than printing two equal numbers."""
    path = tmp_path / "null.toml"
    path.write_text(
        '[null]\ndescription = "The null condition."\nbase_hz = 83.949\n'
        "beat = 0.0\nminutes = 1.0\n"
        '[[null.layer]]\nkind = "pair"\nlevel = 0.3\n',
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["show", "null", "--presets", str(path)])
    assert result.exit_code == 0
    assert "no interaural difference" in result.output


def test_a_loop_far_shorter_than_its_fold_still_renders(tmp_path: Path) -> None:
    """
    The fold cap earns its keep here.

    A twelve-second fold asked for inside a 1.2-second render is impossible,
    and the engine refuses it outright. The preset caps the fold at half the
    render before the engine ever sees it, so what would have been an error
    comes out as a short loop with a short fold.
    """
    path = tmp_path / "brief.toml"
    path.write_text(
        '[brief]\ndescription = "Shorter than its own crossfade."\n'
        "base_hz = 83.949\nbeat = 4.0\nminutes = 0.02\nloop = true\n"
        "loop_crossfade = 12.0\n"
        '[[brief.layer]]\nkind = "ocean"\nlevel = 0.4\n',
        encoding="utf-8",
    )
    out = tmp_path / "brief.wav"
    result = CliRunner().invoke(
        app, ["render", "brief", "--presets", str(path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "loop fold 1 s" in result.output


def test_the_cli_reports_an_unknown_preset_to_render() -> None:
    result = CliRunner().invoke(app, ["render", "trumpet"])
    assert result.exit_code == 1
    assert "unknown preset" in result.output
