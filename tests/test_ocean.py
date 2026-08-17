"""
Stage 5: filters, noise, swells and the ocean.

Four things this stage has to get right, all of which the prototype got wrong
in at least one respect: filter state survives block boundaries, the noise
realisation does not depend on the block size, the swell envelope is a
function of absolute time, and there is a real notch at the carrier.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from violet.dsp.env import (
    SWELL_TAIL_SECONDS,
    Breathing,
    Swell,
    plan_swells,
    swell_envelope,
)
from violet.dsp.filters import FilterDesign
from violet.dsp.noise import PINKING, PinkNoise, WhiteNoise, spawn_seeds
from violet.engine import ArraySink, RenderConfig, render
from violet.layers import Layer, Ocean, Span, StatefulLayer

SR = 32000
ROOT = 335.796
OCEAN_LEVEL = 0.42


def ocean(seed: int = 5, seconds: float = 60.0, level: float = OCEAN_LEVEL) -> Ocean:
    rng = np.random.default_rng(seed)
    return Ocean(
        swells=plan_swells(seconds, rng),
        set_period=float(rng.uniform(85.0, 125.0)),
        notch_hz=ROOT,
        level=level,
        sample_rate=SR,
    )


def render_array(layers: list[Layer], **kwargs: float) -> np.ndarray:
    config = RenderConfig(
        SR,
        duration=float(kwargs.pop("duration", 8.0)),
        block_seconds=float(kwargs.pop("block_seconds", 10.0)),
        gain=1.0,
        fade_seconds=0.0,
    )
    sink = ArraySink()
    render(layers, config, sink)
    return sink.result


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_filter_state_makes_blocks_invisible() -> None:
    """
    The same signal filtered whole, and filtered in pieces, comes out the same.

    Without ``zi`` each piece would start from silence and the joins would ring.
    """
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(10_000)
    design = FilterDesign.butterworth(SR, 520.0, "low")

    whole = design.stream().process(signal)

    piecewise = design.stream()
    pieces = np.concatenate(
        [piecewise.process(signal[at : at + 137]) for at in range(0, 10_000, 137)]
    )
    assert np.max(np.abs(pieces - whole)) < 1e-12


def test_dropping_filter_state_is_visibly_wrong() -> None:
    """The control: a fresh filter per block really does damage the signal."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(10_000)
    design = FilterDesign.butterworth(SR, 520.0, "low")

    whole = design.stream().process(signal)
    naive = np.concatenate(
        [design.stream().process(signal[at : at + 137]) for at in range(0, 10_000, 137)]
    )
    assert np.max(np.abs(naive - whole)) > 1e-3


def test_reset_returns_a_filter_to_silence() -> None:
    design = FilterDesign.butterworth(SR, 520.0, "low")
    stream = design.stream()
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(500)

    first = stream.process(signal)
    stream.reset()
    assert np.array_equal(stream.process(signal), first)


def test_the_notch_removes_its_frequency() -> None:
    design = FilterDesign.notch(SR, ROOT, q=2.0)
    t = np.arange(SR) / SR
    at_notch = design.stream().process(np.sin(2 * np.pi * ROOT * t))
    elsewhere = design.stream().process(np.sin(2 * np.pi * 1500.0 * t))

    settled = slice(SR // 2, None)
    assert np.max(np.abs(at_notch[settled])) < 0.02
    assert np.max(np.abs(elsewhere[settled])) > 0.95


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cutoff": 0.0}, "must sit between"),
        ({"cutoff": 20000.0}, "must sit between"),
    ],
)
def test_butterworth_validates_its_cutoff(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FilterDesign.butterworth(SR, **kwargs)  # type: ignore[arg-type]


def test_notch_validates_itself() -> None:
    with pytest.raises(ValueError, match="must sit between"):
        FilterDesign.notch(SR, 20000.0)
    with pytest.raises(ValueError, match="q must be positive"):
        FilterDesign.notch(SR, ROOT, q=0.0)


def test_filter_design_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="first-order"):
        FilterDesign(b=(1.0,), a=(1.0,))
    with pytest.raises(ValueError, match="a\\[0\\]"):
        FilterDesign(b=(1.0,), a=(0.0, 1.0))


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


def test_noise_is_continuous_across_block_boundaries() -> None:
    """
    No spike where one block ends and the next begins.

    A generator or filter restarted per block leaves a step at every boundary.
    The check is distributional: the biggest sample-to-sample jump at a
    boundary is no larger than the biggest jump anywhere inside a block.
    """
    block = 997
    noise = PinkNoise(seed=1, scale=0.35)
    stream = np.concatenate([noise.block(block) for _ in range(40)])

    steps = np.abs(np.diff(stream))
    boundaries = np.arange(block - 1, len(stream) - 1, block)
    assert len(boundaries) >= 30
    assert steps[boundaries].max() <= steps.max()
    assert steps[boundaries].mean() == pytest.approx(steps.mean(), rel=0.35)


def test_noise_does_not_depend_on_the_block_size() -> None:
    """One stream, chopped two ways, gives the same samples."""
    one = PinkNoise(seed=1)
    ten = PinkNoise(seed=1)
    a = np.concatenate([one.block(100) for _ in range(100)])
    b = np.concatenate([ten.block(1000) for _ in range(10)])
    assert np.array_equal(a, b)


def test_noise_resets_to_the_same_stream() -> None:
    noise = PinkNoise(seed=1)
    first = noise.block(5000)
    noise.reset()
    assert np.array_equal(noise.block(5000), first)


def test_pink_noise_falls_at_three_db_per_octave() -> None:
    """What makes it pink rather than white: equal energy per octave."""
    noise = PinkNoise(seed=3, scale=1.0)
    x = noise.block(2**19)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    power = np.abs(np.fft.rfft(x)) ** 2

    def band_power(low: float, high: float) -> float:
        inside = (freqs >= low) & (freqs < high)
        return float(power[inside].mean())

    for low in (100.0, 200.0, 400.0, 800.0):
        drop = 10.0 * np.log10(band_power(low, low * 2) / band_power(low * 2, low * 4))
        assert 2.0 < drop < 4.0, (low, drop)


def test_spawned_streams_are_independent() -> None:
    left, right = (WhiteNoise(seed) for seed in spawn_seeds(2027, 2))
    a, b = left.block(200_000), right.block(200_000)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.01


def test_spawn_seeds_validates_its_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        spawn_seeds(1, 0)


def test_the_pinking_filter_is_stable() -> None:
    roots = np.roots(np.asarray(PINKING.a))
    assert np.all(np.abs(roots) < 1.0)


# ---------------------------------------------------------------------------
# swells
# ---------------------------------------------------------------------------


def test_a_swell_rises_fast_and_falls_slowly() -> None:
    """The asymmetry that makes it read as water rather than as a siren."""
    swell = Swell(peak_time=10.0, rise=2.0, decay=8.0)
    t = np.linspace(0.0, 60.0, 60_001)
    gain = swell.gain(t)

    assert gain.argmax() == pytest.approx(10.0 * 1000, abs=2)
    assert gain.max() == pytest.approx(1.0, abs=1e-6)
    assert np.all(gain[t < 8.0] == 0.0)

    # A tenth of peak by the time `decay` has elapsed, and still falling.
    at_eighteen = gain[18_000]  # t = 18.0 s, one decay after the peak
    assert at_eighteen == pytest.approx(float(np.exp(-2.4)), rel=1e-6)

    # Two seconds up, and nearly eight to get back down to a tenth of peak.
    to_tenth = float(t[len(t) - 1 - np.argmax(gain[::-1] > 0.1)]) - 10.0
    assert to_tenth == pytest.approx(swell.tau * float(np.log(10.0)), abs=2e-3)
    assert to_tenth > 3.0 * swell.rise


def test_a_swell_is_exactly_zero_outside_its_support() -> None:
    """What makes the envelope independent of where the blocks fall."""
    swell = Swell(peak_time=100.0, rise=2.0, decay=10.0)
    low, high = swell.support()
    assert low == 98.0
    assert high == 100.0 + SWELL_TAIL_SECONDS

    t = np.linspace(0.0, 400.0, 40_001)
    gain = swell.gain(t)
    assert np.all(gain[t < low] == 0.0)
    assert np.all(gain[t > high] == 0.0)
    assert gain[t > high - 1.0][0] < 1e-11


def test_swell_envelope_is_a_function_of_absolute_time() -> None:
    """
    The prototype's window was measured from the block start, not the sample.

    A wave 20 seconds old was included in a 10-second block and excluded from
    a 1-second one, so the envelope depended on the block layout. It does not
    here, and this is the test that says so.
    """
    swells = plan_swells(200.0, np.random.default_rng(5))
    whole = swell_envelope(Span(0, SR * 120, SR).t, swells, 90.0)
    for start, stop in ((0, SR), (SR * 40, SR * 41), (SR * 119, SR * 120)):
        part = swell_envelope(Span(start, stop, SR).t, swells, 90.0)
        assert np.array_equal(whole[start:stop], part), start


def test_swell_envelope_stays_in_range() -> None:
    swells = plan_swells(150.0, np.random.default_rng(5))
    env = swell_envelope(Span(0, SR * 150, SR).t, swells, 95.0)
    assert env.min() >= 0.0
    assert env.max() <= 2.2


def test_swell_envelope_validates_its_set_period() -> None:
    with pytest.raises(ValueError, match="set_period"):
        swell_envelope(np.linspace(0, 1, 10), (), 0.0)


def test_swell_validates_itself() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Swell(peak_time=0.0, rise=0.0, decay=5.0)


def test_plan_swells_spaces_them_out() -> None:
    swells = plan_swells(300.0, np.random.default_rng(5))
    assert swells[0].peak_time == 4.0
    assert swells[-1].peak_time >= 300.0
    for before, after in pairwise(swells):
        assert 6.5 <= after.peak_time - before.peak_time <= 13.0
        assert 1.3 <= before.rise <= 2.8
        assert 5.0 <= before.decay <= 10.5
        assert 0.55 <= before.amplitude <= 1.0


def test_breathing_snaps_every_period_to_a_divisor() -> None:
    """
    Every period divides the loop, and stays close to what it was.

    "Close" gets looser the fewer times a period fits: a 104-second drift in a
    240-second loop has to become 120 seconds, because two is the only whole
    number nearby. That is a 15% change to a period whose exact value was
    arbitrary, so it costs nothing.
    """
    loop = 240.0
    snapped = Breathing().snapped_to_loop(loop, voices=4)
    assert snapped.periods is not None
    for voice, period in enumerate(snapped.periods):
        assert loop / period == pytest.approx(round(loop / period))
        assert period == pytest.approx(Breathing().period_for(voice), rel=0.3)

    # And they stay distinct, so the voices still drift against each other
    # instead of pulsing together.
    assert len(set(snapped.periods)) == len(snapped.periods)


def test_a_short_loop_collapses_the_longest_breathing_periods() -> None:
    """
    Documented limit: below a few minutes, the slowest voices share a period.

    An 85-second and a 104-second drift both round to the whole loop when the
    loop is only two minutes long. They keep their separate phases, so the
    chord still moves, but two of the four voices now breathe at the same rate.
    """
    snapped = Breathing().snapped_to_loop(120.0, voices=4)
    assert snapped.periods is not None
    assert len(set(snapped.periods)) < len(snapped.periods)


def test_breathing_validates_its_loop() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Breathing().snapped_to_loop(0.0)


# ---------------------------------------------------------------------------
# the ocean layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_seconds", [0.1, 0.33, 1.0, 3.0])
def test_the_ocean_is_block_size_invariant(block_seconds: float) -> None:
    """
    The fix that matters most in this stage.

    Both halves of it: one generator per channel rather than two channels
    interleaved through one, and a swell window measured from the sample's own
    time rather than from the block's.
    """
    reference = render_array([ocean()], duration=8.0, block_seconds=10.0)
    other = render_array([ocean()], duration=8.0, block_seconds=block_seconds)
    assert np.max(np.abs(other - reference)) < 1e-6


def test_the_ocean_is_bit_identical_across_block_sizes() -> None:
    """Not merely within tolerance — the same samples."""
    reference = render_array([ocean()], duration=6.0, block_seconds=10.0)
    other = render_array([ocean()], duration=6.0, block_seconds=0.25)
    assert np.array_equal(other, reference)


def test_the_ocean_notches_the_carrier() -> None:
    """Broadband noise over the carrier would mask the beat."""
    audio = render_array([ocean(level=1.0)], duration=20.0)
    x = audio[:, 0]
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x))))

    at_notch = (freqs > ROOT - 2.0) & (freqs < ROOT + 2.0)
    around = ((freqs > 250.0) & (freqs < 300.0)) | ((freqs > 380.0) & (freqs < 430.0))
    depth = 20.0 * np.log10(power[at_notch].mean() / power[around].mean())
    assert depth < -20.0


def test_the_ocean_channels_are_decorrelated() -> None:
    """
    Which is exactly why the ocean carries no beat.

    The binaural percept needs correlated input at both ears. Two independent
    noise streams give it nothing to work with, so the texture stays out of the
    way of the harmony, where the beat actually lives.
    """
    audio = render_array([ocean(level=1.0)], duration=30.0)
    correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    assert abs(correlation) < 0.1

    # The control. Both channels share a wave envelope, so the residual
    # correlation above is envelope, not signal; a bed that really was mono
    # would come back at 1.0 and would sit right on top of the beat.
    mono = float(np.corrcoef(audio[:, 0], audio[:, 0])[0, 1])
    assert mono == pytest.approx(1.0)


def test_the_ocean_is_continuous_at_block_boundaries() -> None:
    audio = render_array([ocean(level=1.0)], duration=8.0, block_seconds=0.5)
    steps = np.abs(np.diff(audio[:, 0]))
    boundaries = np.arange(SR // 2 - 1, len(audio) - 1, SR // 2)
    assert len(boundaries) >= 15
    assert steps[boundaries].max() <= steps.max()


def test_the_ocean_declares_a_usable_peak_bound() -> None:
    audio = render_array([ocean(seconds=60.0, level=1.0)], duration=30.0)
    measured = float(np.max(np.abs(audio)))
    assert measured <= ocean(level=1.0).peak
    assert measured > ocean(level=1.0).peak / 4.0  # not absurdly loose


def test_the_ocean_declares_itself_stateful() -> None:
    bed = ocean()
    assert bed.stateful is True
    assert isinstance(bed, StatefulLayer)


def test_the_ocean_refuses_a_span_out_of_order() -> None:
    """Filter state cannot be seeked, and pretending otherwise would be silent."""
    bed = ocean()
    bed.render(Span(0, 1000, SR))
    with pytest.raises(ValueError, match="must be rendered in order"):
        bed.render(Span(5000, 6000, SR))


def test_the_ocean_refuses_a_span_at_the_wrong_rate() -> None:
    bed = ocean()
    with pytest.raises(ValueError, match="designed for one rate"):
        bed.render(Span(0, 1000, 44100))


def test_the_ocean_resets_between_renders() -> None:
    bed = ocean()
    first = render_array([bed], duration=4.0)
    second = render_array([bed], duration=4.0)
    assert np.array_equal(first, second)


def test_the_ocean_validates_itself() -> None:
    with pytest.raises(ValueError, match="level must not be negative"):
        Ocean(swells=(), set_period=90.0, notch_hz=ROOT, level=-0.1, sample_rate=SR)
    with pytest.raises(ValueError, match="sample rate must be positive"):
        Ocean(swells=(), set_period=90.0, notch_hz=ROOT, level=0.4, sample_rate=0)


def test_a_louder_ocean_is_proportionally_louder() -> None:
    quiet = render_array([ocean(level=0.2)], duration=4.0)
    loud = render_array([ocean(level=0.4)], duration=4.0)
    assert np.allclose(loud, quiet * 2.0)
