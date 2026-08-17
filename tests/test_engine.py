"""
Stage 3: the engine, the layer protocol and the first layers.

The four constraints this stage exists to pin down:

* phase is indexed from absolute sample position, so block boundaries are
  inaudible
* rendering at any block size gives the same samples
* a binaural voice's two ears differ by exactly the beat and nothing else
* the peak stays under the ceiling, with no clipped samples
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

import spectral
from violet import tuning
from violet.dsp.env import fade_envelope, fade_length, slow_lfo
from violet.dsp.osc import sine, sine_pair
from violet.engine import (
    ArraySink,
    RenderConfig,
    RenderResult,
    headroom_gain,
    loop_length,
    render,
    render_to_file,
)
from violet.layers import BinauralPair, Layer, Pedal, Span, StatefulLayer

if TYPE_CHECKING:
    from pathlib import Path

SR = 44100
ORIGIN_HZ = 83.949
CARRIER = 335.796
BEAT = 4.0


def prototype_one_layers(
    beat: float = BEAT,
    level: float = 0.30,
    drone: float = 0.09,
    base: float = ORIGIN_HZ,
) -> list[Layer]:
    """The layer stack of reference/binaural_e_violet.py, with its defaults."""
    return [
        BinauralPair(carrier=tuning.carrier_for(base), beat=beat, level=level),
        Pedal(freq=base, level=drone),
    ]


def render_array(layers: list[Layer], config: RenderConfig) -> np.ndarray:
    sink = ArraySink()
    render(layers, config, sink)
    return sink.result


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------


def test_sine_is_a_function_of_absolute_time() -> None:
    """The same index gives the same sample, whatever span it arrives in."""
    whole = sine(Span(0, 10000, SR).t, 335.796)
    part = sine(Span(4000, 6000, SR).t, 335.796)
    assert np.array_equal(whole[4000:6000], part)


def test_sine_pair_splits_the_beat_around_the_carrier() -> None:
    t = Span(0, SR, SR).t
    left, right = sine_pair(t, 400.0, 4.0, 0.5)
    assert np.allclose(left, sine(t, 398.0, 0.5))
    assert np.allclose(right, sine(t, 402.0, 0.5))


def test_a_zero_beat_pair_is_two_identical_tones() -> None:
    """The null condition for a blinded trial: no interaural difference."""
    t = Span(0, SR, SR).t
    left, right = sine_pair(t, 400.0, 0.0, 0.5)
    assert np.array_equal(left, right)


def test_sine_pair_rejects_a_beat_wider_than_the_carrier() -> None:
    t = Span(0, 100, SR).t
    with pytest.raises(ValueError, match="must exceed half the beat"):
        sine_pair(t, 1.0, 4.0, 0.5)


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------


def test_span_time_base_is_absolute() -> None:
    span = Span(88200, 88300, SR)
    assert span.n == 100
    assert span.t[0] == pytest.approx(2.0)
    assert span.t0 == pytest.approx(2.0)
    assert span.t1 == pytest.approx(88299 / SR)
    assert span.indices[0] == 88200


def test_span_caches_its_arrays() -> None:
    span = Span(0, 10, SR)
    assert span.t is span.t
    assert span.indices is span.indices


@pytest.mark.parametrize(
    ("start", "stop", "rate"),
    [(-1, 10, SR), (10, 10, SR), (10, 5, SR), (0, 10, 0)],
)
def test_span_validates_itself(start: int, stop: int, rate: int) -> None:
    with pytest.raises(ValueError, match=r"span|sample rate"):
        Span(start, stop, rate)


# ---------------------------------------------------------------------------
# envelopes
# ---------------------------------------------------------------------------


def test_fade_envelope_reaches_zero_at_both_ends() -> None:
    total = 1000
    env = fade_envelope(Span(0, total, SR).indices, total, 100)
    assert env[0] == pytest.approx(0.0)
    assert env[-1] == pytest.approx(0.0)
    assert env[500] == pytest.approx(1.0)
    assert np.all(env >= 0.0)
    assert np.all(env <= 1.0)


def test_fade_envelope_is_smooth() -> None:
    """No corner where the fade meets the body — that is why it is sin^2."""
    total = 44100
    env = fade_envelope(Span(0, total, SR).indices, total, 4410)
    second_difference = np.diff(env, n=2)
    assert np.max(np.abs(second_difference)) < 1e-6


def test_fade_envelope_is_position_dependent_not_block_dependent() -> None:
    total = 10000
    whole = fade_envelope(Span(0, total, SR).indices, total, 2000)
    part = fade_envelope(Span(3000, 4000, SR).indices, total, 2000)
    assert np.array_equal(whole[3000:4000], part)


def test_fade_length_is_capped_for_short_renders() -> None:
    assert fade_length(SR * 60, SR, 8.0) == SR * 8
    assert fade_length(SR * 10, SR, 8.0) == SR * 10 // 4
    assert fade_length(SR * 10, SR, 0.0) == 0
    assert fade_length(SR * 10, SR, 22.0, max_denominator=5) == SR * 10 // 5


def test_slow_lfo_stays_between_its_bounds() -> None:
    t = Span(0, SR * 200, SR).t
    lfo = slow_lfo(t, period=70.0, low=0.55, high=1.0)
    assert lfo.min() == pytest.approx(0.55, abs=1e-6)
    assert lfo.max() == pytest.approx(1.0, abs=1e-6)


def test_slow_lfo_is_a_function_of_absolute_time() -> None:
    whole = slow_lfo(Span(0, 10000, SR).t, period=70.0)
    part = slow_lfo(Span(4000, 6000, SR).t, period=70.0)
    assert np.array_equal(whole[4000:6000], part)


# ---------------------------------------------------------------------------
# constraint 1: absolute-time phase indexing
# ---------------------------------------------------------------------------


def test_no_discontinuity_at_block_boundaries() -> None:
    """
    A per-block phase reset shows up as a step at every boundary.

    The test is not "the signal is continuous" — a sine is always continuous —
    it is that the largest sample-to-sample step at a boundary is no larger
    than the largest step anywhere else. A reset would put a step of up to
    twice the amplitude right there.
    """
    config = RenderConfig(SR, duration=4.0, block_seconds=0.5, fade_seconds=0.0)
    audio = render_array(prototype_one_layers(), config)
    steps = np.abs(np.diff(audio[:, 0]))

    block = config.block_frames
    boundaries = np.arange(block - 1, len(audio) - 1, block)
    assert len(boundaries) >= 7
    assert steps[boundaries].max() <= steps.max()


# ---------------------------------------------------------------------------
# constraint 2: block-size invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_seconds", [0.1, 0.37, 1.0, 3.0, 10.0])
def test_block_size_invariance(block_seconds: float) -> None:
    """One second against ten, and awkward sizes that leave a part block."""
    reference = render_array(
        prototype_one_layers(),
        RenderConfig(SR, duration=5.0, block_seconds=10.0),
    )
    other = render_array(
        prototype_one_layers(),
        RenderConfig(SR, duration=5.0, block_seconds=block_seconds),
    )
    assert other.shape == reference.shape
    assert np.max(np.abs(other - reference)) < 1e-6


def test_block_size_does_not_change_the_reported_result() -> None:
    def result_at(block_seconds: float) -> RenderResult:
        return render(
            prototype_one_layers(),
            RenderConfig(SR, duration=5.0, block_seconds=block_seconds),
            ArraySink(),
        )

    one = result_at(1.0)
    ten = result_at(10.0)
    assert one.frames == ten.frames
    assert one.gain == ten.gain
    assert one.peak == pytest.approx(ten.peak, abs=1e-12)
    assert one.blocks == 5
    assert ten.blocks == 1


# ---------------------------------------------------------------------------
# constraint 3: one beat per mix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beat", [2.0, 4.0, 7.83, 10.0])
def test_beat_frequency_is_recovered_by_fft(beat: float) -> None:
    config = RenderConfig(SR, duration=5.0, block_seconds=1.0, fade_seconds=0.0)
    audio = render_array(
        [BinauralPair(carrier=CARRIER, beat=beat, level=0.3)],
        config,
    )
    (left_hz, right_hz), *rest = spectral.matched_pairs(
        audio[:, 0], audio[:, 1], SR, voices=1
    )
    assert not rest
    assert left_hz == pytest.approx(CARRIER - beat / 2, abs=0.01)
    assert right_hz == pytest.approx(CARRIER + beat / 2, abs=0.01)
    assert right_hz - left_hz == pytest.approx(beat, abs=0.01)


def test_the_pedal_carries_no_beat() -> None:
    """Identical in both ears, so there is no interaural difference at all."""
    config = RenderConfig(SR, duration=3.0, block_seconds=1.0, fade_seconds=0.0)
    audio = render_array([Pedal(freq=ORIGIN_HZ, level=0.3)], config)
    assert np.array_equal(audio[:, 0], audio[:, 1])


def test_a_zero_beat_render_has_identical_channels() -> None:
    """
    The null condition, end to end.

    Everything else about the render is unchanged; there is simply no
    interaural difference for the brainstem to find.
    """
    config = RenderConfig(SR, duration=3.0, block_seconds=1.0)
    audio = render_array(prototype_one_layers(beat=0.0), config)
    assert np.array_equal(audio[:, 0], audio[:, 1])


def test_the_beat_is_audible_as_amplitude_modulation_when_summed() -> None:
    """
    Summing the ears to mono turns the beat into a 4 Hz amplitude envelope.

    This is what a speaker does, and why the effect needs headphones: in a
    room the two tones mix in the air and you get a physical tremolo instead
    of the interaural percept.
    """
    config = RenderConfig(SR, duration=8.0, block_seconds=1.0, fade_seconds=0.0)
    audio = render_array([BinauralPair(carrier=CARRIER, beat=BEAT, level=0.3)], config)

    mono = audio[:, 0] + audio[:, 1]
    envelope = np.abs(mono)
    envelope -= envelope.mean()
    peaks = spectral.peak_frequencies(envelope, SR, count=1, min_separation_hz=0.5)
    assert peaks[0] == pytest.approx(BEAT, abs=0.05)


# ---------------------------------------------------------------------------
# constraint 7: peak safety
# ---------------------------------------------------------------------------


def test_prototype_one_stays_under_the_ceiling() -> None:
    config = RenderConfig(SR, duration=5.0, block_seconds=1.0)
    result = render(prototype_one_layers(), config, ArraySink())
    assert result.peak < 0.95
    assert result.clipped == 0


def test_headroom_gain_only_ever_attenuates() -> None:
    quiet: list[Layer] = [BinauralPair(CARRIER, BEAT, 0.1), Pedal(ORIGIN_HZ, 0.05)]
    assert headroom_gain(quiet, ceiling=0.89) == 1.0

    loud: list[Layer] = [BinauralPair(CARRIER, BEAT, 0.7), Pedal(ORIGIN_HZ, 0.5)]
    assert headroom_gain(loud, ceiling=0.89) == pytest.approx(0.89 / 1.2)
    assert headroom_gain([], ceiling=0.89) == 1.0


def test_a_loud_stack_is_pulled_under_the_ceiling() -> None:
    layers: list[Layer] = [BinauralPair(CARRIER, BEAT, 0.7), Pedal(ORIGIN_HZ, 0.5)]
    config = RenderConfig(SR, duration=3.0, block_seconds=1.0, ceiling=0.89)
    result = render(layers, config, ArraySink())
    assert result.gain < 1.0
    assert result.peak <= 0.89 + 1e-12
    assert result.clipped == 0


def test_explicit_gain_overrides_the_headroom_calculation() -> None:
    layers = prototype_one_layers()
    config = RenderConfig(SR, duration=2.0, block_seconds=1.0, gain=0.5)
    result = render(layers, config, ArraySink())
    assert result.gain == 0.5


# ---------------------------------------------------------------------------
# determinism and state
# ---------------------------------------------------------------------------


def test_rendering_twice_gives_identical_samples() -> None:
    config = RenderConfig(SR, duration=3.0, block_seconds=0.7)
    first = render_array(prototype_one_layers(), config)
    second = render_array(prototype_one_layers(), config)
    assert np.array_equal(first, second)


def test_stateless_layers_declare_themselves_stateless() -> None:
    for layer in prototype_one_layers():
        assert layer.stateful is False
        assert not isinstance(layer, StatefulLayer)
        assert isinstance(layer, Layer)


def test_the_engine_resets_stateful_layers_before_rendering() -> None:
    class Counter:
        stateful = True
        peak = 0.0

        def __init__(self) -> None:
            self.resets = 0
            self.blocks = 0

        def reset(self) -> None:
            self.resets += 1

        def render(self, span: Span) -> tuple[np.ndarray, np.ndarray]:
            self.blocks += 1
            silence = np.zeros(span.n)
            return silence, silence

    counter = Counter()
    assert isinstance(counter, StatefulLayer)
    config = RenderConfig(SR, duration=2.0, block_seconds=1.0)
    render([counter], config, ArraySink())
    render([counter], config, ArraySink())
    assert counter.resets == 2
    assert counter.blocks == 4


# ---------------------------------------------------------------------------
# looping
# ---------------------------------------------------------------------------


def test_loop_length_snaps_to_whole_beat_cycles() -> None:
    assert loop_length(20.0 * 60, 4.0) == 1200.0
    assert loop_length(1200.3, 4.0) == pytest.approx(1200.25)
    assert loop_length(100.0, 7.83) == pytest.approx(round(100.0 * 7.83) / 7.83)
    assert loop_length(0.01, 4.0) == pytest.approx(0.25)
    assert loop_length(123.4, 0.0) == 123.4


def wrap_and_internal_steps(
    carrier: float, beat: float, seconds: float
) -> tuple[float, float]:
    """Largest step at the loop join, and the largest step anywhere inside."""
    config = RenderConfig(SR, duration=seconds, block_seconds=1.0, fade_seconds=0.0)
    audio = render_array([BinauralPair(carrier=carrier, beat=beat, level=0.3)], config)
    internal = float(np.abs(np.diff(audio[:, 0])).max())
    wrap = float(abs(audio[0, 0] - audio[-1, 0]))
    return wrap, internal


def test_a_loop_joins_seamlessly_when_the_carrier_completes_whole_cycles() -> None:
    """
    336 Hz and a 4 Hz beat: 1002 and 1014 whole cycles in three seconds.

    Both ears come back to the phase they started at, so the step from the
    last sample round to the first is an ordinary step, no bigger than any
    step inside the file. This is what seamless actually requires.
    """
    beat, seconds = 4.0, 3.0
    for freq in (336.0 - beat / 2, 336.0 + beat / 2):
        assert (freq * seconds) % 1.0 == pytest.approx(0.0)

    wrap, internal = wrap_and_internal_steps(336.0, beat, seconds)
    assert wrap <= internal


def test_the_prototypes_loop_rule_still_steps_on_an_arbitrary_carrier() -> None:
    """
    A documented limitation, pinned so that fixing it is a deliberate act.

    Snapping the length to whole beat cycles brings the *beat* back into phase
    but not the carrier: 333.796 Hz fits 1001.388 cycles into three seconds,
    and the leftover 0.388 is a step in the waveform at the join. The pulse
    loops; the tone clicks.
    """
    beat = 4.0
    seconds = loop_length(3.0, beat)
    assert (seconds * beat) % 1.0 == pytest.approx(0.0)
    assert (CARRIER - beat / 2) * seconds % 1.0 == pytest.approx(0.388, abs=1e-3)

    wrap, internal = wrap_and_internal_steps(CARRIER, beat, seconds)
    assert wrap > internal * 5


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("suffix", "subtype"), [(".wav", "PCM_16"), (".flac", None)])
def test_files_are_written_without_ffmpeg(
    tmp_path: Path, suffix: str, subtype: str | None
) -> None:
    path = tmp_path / f"session{suffix}"
    config = RenderConfig(SR, duration=2.0, block_seconds=0.5)
    result = render_to_file(prototype_one_layers(), config, path, subtype=subtype)

    assert path.exists()
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    assert rate == SR
    assert audio.shape == (result.frames, 2)

    expected = render_array(prototype_one_layers(), config)
    assert np.max(np.abs(audio - expected)) < 2.0 / 32767.0


def test_render_result_reports_the_shape_of_the_render() -> None:
    config = RenderConfig(SR, duration=2.5, block_seconds=1.0)
    result = render(prototype_one_layers(), config, ArraySink())
    assert result.frames == round(SR * 2.5)
    assert result.blocks == 3
    assert result.seconds == pytest.approx(2.5)
    assert result.peak_dbfs < 0.0
    assert result.peak_dbfs == pytest.approx(20.0 * math.log10(result.peak))


def test_array_sink_starts_empty() -> None:
    assert ArraySink().result.shape == (0, 2)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"duration": 0.0},
        {"duration": -1.0},
        {"block_seconds": 0.0},
        {"gain": 0.0},
        {"ceiling": 0.0},
    ],
)
def test_render_config_validates_itself(kwargs: dict[str, float]) -> None:
    base = {"sample_rate": SR, "duration": 10.0}
    with pytest.raises(ValueError, match="must be positive"):
        RenderConfig(**{**base, **kwargs})  # type: ignore[arg-type]


def test_a_render_shorter_than_one_sample_is_rejected() -> None:
    config = RenderConfig(SR, duration=1e-6, block_seconds=1.0)
    with pytest.raises(ValueError, match="zero samples"):
        render(prototype_one_layers(), config, ArraySink())
