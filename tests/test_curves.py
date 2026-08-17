"""
Stage 8: beat rates that move, and the phase integral they need.

The headline tests are the two named for the naive expression, because the
whole module exists to avoid one specific mistake and a test that only checks
the right answer does not record what the wrong one looks like. There are two
because the mistake has two faces: a click where the rate steps, and — far more
dangerous — a perfectly smooth signal at the wrong rate where it glides.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

import spectral
from violet.cli import app
from violet.dsp.curves import SHAPES, BeatCurve
from violet.dsp.osc import swept_pair
from violet.engine import ArraySink, RenderConfig, render
from violet.harmony import CHORDS, ChordEvent
from violet.layers import BinauralPair, ChordBed, Layer, Span
from violet.presets import load_library, resolve_beat
from violet.trial import Trial

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SR = 32000
ROOT = 335.796

DESCENT = BeatCurve.from_minutes((0, 8, 25, 45), (10, 10, 2, 2), shape="exponential")
RAMP = BeatCurve(points=((0.0, 10.0), (60.0, 2.0)))


def times(seconds: float, sample_rate: int = SR) -> np.ndarray:
    return Span(0, int(seconds * sample_rate), sample_rate).t


# ---------------------------------------------------------------------------
# the integral
# ---------------------------------------------------------------------------


def test_a_constant_curve_integrates_to_frequency_times_time() -> None:
    curve = BeatCurve.constant(4.0)
    t = times(10.0, 1000)
    assert np.allclose(curve.at(t), 4.0)
    assert np.allclose(curve.cycles(t), 4.0 * t)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_closed_form_matches_numerical_integration(shape: str) -> None:
    """
    The check that the algebra is right, done the slow way.

    A fine-grained cumulative trapezoid over the instantaneous rate, against
    the closed form. If the segment formulae are wrong this diverges, and it
    diverges most where the curve bends.
    """
    curve = BeatCurve(
        points=((0.0, 10.0), (30.0, 7.0), (90.0, 2.0), (150.0, 2.0)),
        shape=shape,  # type: ignore[arg-type]
    )
    t = np.linspace(0.0, 200.0, 2_000_001)
    rates = curve.at(t)
    step = float(t[1] - t[0])

    numeric = np.concatenate([[0.0], np.cumsum((rates[:-1] + rates[1:]) / 2.0 * step)])
    closed = curve.cycles(t)
    assert np.max(np.abs(closed - numeric)) < 1e-4


def test_the_integral_never_goes_backwards() -> None:
    """A rate cannot be negative, so the phase can only advance."""
    t = times(200.0, 200)
    for curve in (DESCENT, RAMP):
        assert np.all(np.diff(curve.cycles(t)) >= 0.0)


def test_the_integral_is_a_function_of_absolute_time() -> None:
    """
    The property an accumulator would give up.

    Evaluated over a whole span or over a slice of it, the same sample gets the
    same answer — no state, so nothing to carry and nothing to drift.
    """
    whole = DESCENT.cycles(times(600.0, 100))
    part = DESCENT.cycles(Span(20_000, 30_000, 100).t)
    assert np.array_equal(whole[20_000:30_000], part)


def test_the_rate_is_held_before_and_after_the_curve() -> None:
    curve = BeatCurve(points=((10.0, 8.0), (20.0, 4.0)))
    t = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 100.0])
    assert curve.at(t)[0] == 8.0
    assert curve.at(t)[1] == 8.0
    assert curve.at(t)[2] == 8.0
    assert curve.at(t)[3] == pytest.approx(6.0)
    assert curve.at(t)[4] == 4.0
    assert curve.at(t)[5] == 4.0

    # Held flat means the integral over the flat part is just rate times time.
    assert curve.cycles(np.array([5.0]))[0] == pytest.approx(40.0)
    assert curve.cycles(np.array([100.0]))[0] == pytest.approx(
        8.0 * 10.0 + (8.0 + 4.0) / 2.0 * 10.0 + 4.0 * 80.0
    )


def test_exponential_moves_by_ratio_and_linear_by_difference() -> None:
    """
    Halfway through, the two shapes are in different places.

    Linear sits at the arithmetic mean of the ends, exponential at the
    geometric one — which for 10 Hz down to 2.5 is 6.25 against 5.
    """
    ends = ((0.0, 10.0), (100.0, 2.5))
    midpoint = np.array([50.0])
    assert BeatCurve(points=ends, shape="linear").at(midpoint)[0] == pytest.approx(6.25)
    assert BeatCurve(points=ends, shape="exponential").at(midpoint)[0] == pytest.approx(
        5.0
    )


def test_an_exponential_segment_with_equal_ends_is_flat() -> None:
    """The degenerate case: ratio one, and a logarithm that would divide by zero."""
    curve = BeatCurve(points=((0.0, 4.0), (10.0, 4.0)), shape="exponential")
    t = times(10.0, 100)
    assert np.allclose(curve.at(t), 4.0)
    assert np.allclose(curve.cycles(t), 4.0 * t)


def test_curves_are_values() -> None:
    """Equal, hashable, and safe to sit inside a frozen config."""
    one = BeatCurve(points=((0.0, 10.0), (60.0, 2.0)))
    two = BeatCurve(points=((0.0, 10.0), (60.0, 2.0)))
    assert one == two
    assert hash(one) == hash(two)
    assert one != BeatCurve(points=((0.0, 10.0), (60.0, 2.0)), shape="exponential")
    assert len({one, two}) == 1


def test_curve_description_reads_as_a_plan() -> None:
    assert "10 Hz" in DESCENT.describe()
    assert "exponential" in DESCENT.describe()
    assert BeatCurve.constant(4.0).describe() == "4 Hz, constant"
    assert not BeatCurve.constant(4.0).moves
    assert DESCENT.moves
    assert DESCENT.span == (2.0, 10.0)
    assert DESCENT.start_hz == 10.0
    assert DESCENT.end_hz == 2.0


@pytest.mark.parametrize(
    ("points", "shape", "message"),
    [
        ((), "linear", "at least one point"),
        (((0.0, 4.0),), "sideways", "shape must be one of"),
        (((-1.0, 4.0),), "linear", "must not be negative"),
        (((10.0, 4.0), (5.0, 2.0)), "linear", "must increase"),
        (((0.0, 4.0), (0.0, 2.0)), "linear", "must increase"),
        (((0.0, -4.0),), "linear", "rates must not be negative"),
        (((0.0, 4.0), (10.0, 0.0)), "exponential", "cannot pass through zero"),
    ],
)
def test_a_nonsense_curve_is_rejected(
    points: tuple[tuple[float, float], ...], shape: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BeatCurve(points=points, shape=shape)  # type: ignore[arg-type]


def test_mismatched_minutes_and_rates_are_rejected() -> None:
    with pytest.raises(ValueError, match="one rate per time"):
        BeatCurve.from_minutes((0, 10, 20), (10, 2))


# ---------------------------------------------------------------------------
# the oscillator
# ---------------------------------------------------------------------------


def measure_beat(left: np.ndarray, right: np.ndarray) -> float:
    """The beat between two channels, from their spectral peaks."""
    (low, high), *_ = spectral.matched_pairs(left, right, SR, voices=1)
    return high - low


def test_the_naive_expression_plays_the_wrong_rate() -> None:
    """
    Why the integral is not optional — and it is not about clicks.

    For a smooth curve the naive expression is perfectly smooth too, which is
    exactly why it survives a listen. Differentiating its phase gives an
    instantaneous rate of ``f(t) + t*f'(t)``: gliding 10 Hz to 2 Hz over a
    minute, at thirty seconds in you ask for 6 Hz and it delivers 2. Measured,
    not asserted.
    """
    at_second, window = 30.0, 4.0
    span = Span(
        int((at_second - window / 2) * SR), int((at_second + window / 2) * SR), SR
    )
    t = span.t
    asked = float(RAMP.at(np.array([at_second]))[0])
    assert asked == pytest.approx(6.0, abs=0.01)

    honest = measure_beat(*swept_pair(t, ROOT, RAMP))
    assert honest == pytest.approx(asked, abs=0.05)

    # The mistake, written out: the instantaneous rate dropped into 2*pi*f*t.
    rates = RAMP.at(t)
    naive = measure_beat(
        np.sin(2 * np.pi * (ROOT - rates / 2.0) * t),
        np.sin(2 * np.pi * (ROOT + rates / 2.0) * t),
    )
    slope = (RAMP.end_hz - RAMP.start_hz) / 60.0
    assert naive == pytest.approx(asked + at_second * slope, abs=0.05)
    assert abs(naive - asked) > 3.0


def test_the_naive_expression_clicks_where_the_rate_steps() -> None:
    """
    The other failure mode, for a curve that jumps rather than glides.

    Here it *is* a click, and a loud one — seventeen times the largest
    legitimate sample-to-sample move.
    """
    stepped = BeatCurve(points=((0.0, 10.0), (5.0, 10.0), (5.001, 2.0), (20.0, 2.0)))
    t = times(10.0)

    honest, _ = swept_pair(t, ROOT, stepped)
    rates = stepped.at(t)
    naive = np.sin(2 * np.pi * (ROOT - rates / 2.0) * t)

    honest_step = float(np.max(np.abs(np.diff(honest))))
    naive_step = float(np.max(np.abs(np.diff(naive))))
    assert naive_step > 10.0 * honest_step


@pytest.mark.parametrize("at_second", [5.0, 25.0, 55.0])
def test_a_swept_pair_holds_its_carrier_still(at_second: float) -> None:
    """
    The pitch must not drift while the pulse slows.

    The two ears move apart symmetrically, so whatever the rate is doing their
    midpoint stays on the carrier. Measured per channel: the mono sum will not
    do, because summing them turns the carrier into a suppressed one with the
    energy in the sidebands.
    """
    window = 4.0
    span = Span(
        int((at_second - window / 2) * SR), int((at_second + window / 2) * SR), SR
    )
    left, right = swept_pair(span.t, ROOT, RAMP)
    (low, high), *_ = spectral.matched_pairs(left, right, SR, voices=1)
    assert (low + high) / 2.0 == pytest.approx(ROOT, abs=0.02)


@pytest.mark.parametrize("at_second", [2.0, 20.0, 50.0])
def test_the_instantaneous_beat_matches_the_curve(at_second: float) -> None:
    """
    Measured where the curve says it should be, at three points along it.

    A four-second window, short enough that the rate barely moves inside it and
    long enough to resolve two peaks a few hertz apart.
    """
    window = 4.0
    curve = BeatCurve(points=((0.0, 10.0), (60.0, 3.0)))
    t = Span(
        int((at_second - window / 2) * SR), int((at_second + window / 2) * SR), SR
    ).t
    left, right = swept_pair(t, ROOT, curve)

    (low, high), *rest = spectral.matched_pairs(left, right, SR, voices=1)
    assert not rest
    expected = float(curve.at(np.array([at_second]))[0])
    assert high - low == pytest.approx(expected, abs=0.05)


def test_a_swept_pair_refuses_a_carrier_it_would_swamp() -> None:
    t = times(1.0, 100)
    with pytest.raises(ValueError, match="fastest rate in the curve"):
        swept_pair(t, 4.0, BeatCurve(points=((0.0, 10.0),)))


# ---------------------------------------------------------------------------
# the layers
# ---------------------------------------------------------------------------


def render_array(layers: Sequence[Layer], seconds: float, block: float) -> np.ndarray:
    sink = ArraySink()
    render(
        layers,
        RenderConfig(SR, seconds, block_seconds=block, gain=1.0, fade_seconds=0.0),
        sink,
    )
    return sink.result


@pytest.mark.parametrize("block", [0.1, 0.37, 1.0, 10.0])
def test_a_swept_pair_is_block_size_invariant(block: float) -> None:
    """Exactly, not approximately — which an accumulator could not promise."""
    layers = [BinauralPair(carrier=ROOT, beat=RAMP, level=0.3)]
    reference = render_array(layers, 8.0, 10.0)
    other = render_array(layers, 8.0, block)
    assert np.array_equal(other, reference)


def test_a_swept_pair_has_no_step_at_block_boundaries() -> None:
    audio = render_array([BinauralPair(carrier=ROOT, beat=RAMP, level=0.3)], 8.0, 0.5)
    steps = np.abs(np.diff(audio[:, 0]))
    boundaries = np.arange(SR // 2 - 1, len(audio) - 1, SR // 2)
    assert len(boundaries) >= 14
    assert steps[boundaries].max() <= steps.max()


def test_a_swept_chord_bed_keeps_one_beat_across_every_voice() -> None:
    """
    Constraint 3 still holds while the rate is moving.

    Four voices, eight tones, all sweeping together — every pair still differs
    by the same amount at any instant.
    """
    curve = BeatCurve(points=((0.0, 9.0), (120.0, 4.0)))
    held = ChordEvent(CHORDS["i"], 0.0, 100.0, fade_in=False)
    bed = ChordBed(events=(held,), root=ROOT, beat=curve, level=0.07)

    at_second = 30.0
    audio = render_array([bed], 40.0, 10.0)
    start = int((at_second - 2.0) * SR)
    window = audio[start : start + 4 * SR]

    pairs = spectral.matched_pairs(
        window[:, 0], window[:, 1], SR, voices=4, min_separation_hz=20.0
    )
    expected = float(curve.at(np.array([at_second]))[0])
    for low, high in pairs:
        assert high - low == pytest.approx(expected, abs=0.08)


def test_a_swept_pair_declines_to_close_a_loop() -> None:
    """
    A moving rate and a seamless loop cannot both be had.

    The layer says so by declining to snap, and the engine falls back to
    crossfading it — which is why the preset refuses the combination outright
    rather than quietly producing a muddy join.
    """
    assert (
        BinauralPair(carrier=ROOT, beat=RAMP, level=0.3).snapped_to_loop(60.0) is None
    )
    assert BinauralPair(carrier=ROOT, beat=4.0, level=0.3).snapped_to_loop(60.0)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def test_a_curve_can_be_written_in_toml() -> None:
    curve = resolve_beat(
        {"shape": "exponential", "at_minutes": [0, 8, 25], "hz": [10, 10, 2]}
    )
    assert isinstance(curve, BeatCurve)
    assert curve.shape == "exponential"
    assert curve.points[1] == (480.0, 10.0)
    assert curve.span == (2.0, 10.0)


def test_a_scalar_beat_still_resolves() -> None:
    assert resolve_beat("theta") == 4.0
    assert resolve_beat(4.0) == 4.0
    assert resolve_beat(BeatCurve.constant(3.0)) == BeatCurve.constant(3.0)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"at_minutes": [0, 8]}, "needs at_minutes and hz"),
        ({"hz": [10, 2]}, "needs at_minutes and hz"),
        ({"at_minutes": [0], "hz": [4], "wobble": 1}, "unknown keys"),
    ],
)
def test_a_broken_curve_table_is_reported(
    table: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_beat(table)


def test_the_descent_preset_is_a_descent() -> None:
    preset = load_library()["descent"]
    assert isinstance(preset.beat, BeatCurve)
    assert preset.beat.start_hz == 10.0
    assert preset.beat.end_hz == 2.0
    assert preset.beat.shape == "exponential"
    assert preset.minutes == 45.0

    # Alpha for the first eight minutes, delta for the last twenty.
    rates = preset.beat.at(np.array([0.0, 60.0, 480.0, 1500.0, 2700.0]))
    assert rates[0] == pytest.approx(10.0)
    assert rates[1] == pytest.approx(10.0)
    assert rates[2] == pytest.approx(10.0)
    assert rates[3] == pytest.approx(2.0)
    assert rates[4] == pytest.approx(2.0)
    # And it passes through theta somewhere in the middle of the glide.
    middle = float(preset.beat.at(np.array([900.0]))[0])
    assert 3.5 < middle < 7.0


def test_a_moving_beat_and_a_loop_are_refused_together() -> None:
    """
    An honest "these two cannot coexist" rather than a muddy compromise.

    The rate at the end is not the rate at the start, so however smooth the
    waveform is at the join, the pulse jumps.
    """
    preset = load_library()["descent"]
    with pytest.raises(ValueError, match="do not go together"):
        replace(preset, loop=True)


def test_a_flat_curve_may_still_loop() -> None:
    """It is the *moving* that is the problem, not the curve."""
    preset = load_library()["descent"]
    flat = replace(preset, beat=BeatCurve.constant(4.0), loop=True)
    assert flat.loop


def test_the_cli_describes_a_curve_rather_than_formatting_it_as_a_number() -> None:
    result = CliRunner().invoke(app, ["show", "descent"])
    assert result.exit_code == 0, result.output
    assert "beat curve" in result.output
    assert "exponential" in result.output
    assert "ears sweep" in result.output


def test_a_swept_preset_can_be_trialled(tmp_path: Path) -> None:
    """The descent is exactly the thing worth blinding."""
    preset = load_library()["descent"].with_overrides(minutes=0.02)
    trial = Trial.create(preset, tmp_path / "t", "t")
    assert "exponential" in trial.beat_label

    reloaded = Trial.load(trial.directory)
    assert isinstance(reloaded.beat, BeatCurve)
    assert reloaded.beat == preset.beat
