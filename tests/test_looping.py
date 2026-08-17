"""
Seamless looping: a render that can be played on repeat with no break.

Two mechanisms, and the tests here are as much about *which* one gets used as
about the result. Fixed-frequency layers are made exactly periodic by rounding
their frequencies onto whole cycles. Noise cannot be, so it is folded back on
itself with an equal-power crossfade.

Both are measured two ways, because a loop can fail at two scales. A step
between the last sample and the first is a click. A jump in the short-term
level — the surf stopping mid-wave, a chord vanishing — is a lurch, and the
sample-step test cannot see it at all.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from violet import tuning
from violet.dsp.env import plan_swells
from violet.engine import (
    ArraySink,
    LoopConfig,
    RenderConfig,
    _loop_gains,
    render,
)
from violet.harmony import (
    TRANSITIONS,
    WalkConfig,
    plan_loop_progression,
    plan_progression,
)
from violet.layers import (
    BinauralPair,
    ChordBed,
    Layer,
    LoopableLayer,
    Ocean,
    Pedal,
    snap_frequency,
)

SR = 32000
BASE = 83.949
ROOT = tuning.carrier_for(BASE)
BEAT = 4.0

#: Loops in these tests are short and their chords are correspondingly short,
#: so a 48-second render still crosses three chord changes and a dozen wave
#: events. The shipped presets use the real lengths; nothing here depends on
#: the numbers being the shipped ones.
LOOP = 48.0
CROSSFADE = 6.0
FAST_WALK = WalkConfig(crossfade=5.0, min_seconds=12.0, max_seconds=18.0)


def tonal_layers() -> list[Layer]:
    return [BinauralPair(carrier=ROOT, beat=BEAT, level=0.30), Pedal(BASE, 0.09)]


def ocean_layers(seconds: float = LOOP, seed: int = 5, *, closed: bool) -> list[Layer]:
    rng = np.random.default_rng(seed)
    events = (
        plan_loop_progression(seconds, rng, FAST_WALK)
        if closed
        else plan_progression(seconds, rng, FAST_WALK)
    )
    swells = plan_swells(seconds, rng)
    set_period = float(rng.uniform(85.0, 125.0))
    return [
        ChordBed(
            events=events,
            root=ROOT,
            beat=BEAT,
            level=0.070,
            crossfade=FAST_WALK.crossfade,
        ),
        Pedal(freq=BASE, level=0.080),
        Ocean(
            swells=swells,
            set_period=set_period,
            notch_hz=ROOT,
            level=0.42,
            sample_rate=SR,
        ),
    ]


def render_loop(
    layers: list[Layer],
    seconds: float = LOOP,
    crossfade: float = CROSSFADE,
    block_seconds: float = 10.0,
) -> tuple[np.ndarray, object]:
    sink = ArraySink()
    result = render(
        layers,
        RenderConfig(
            SR,
            seconds,
            block_seconds=block_seconds,
            gain=1.25,
            fade_seconds=0.0,
            loop=LoopConfig(crossfade=crossfade),
        ),
        sink,
    )
    return sink.result, result


def render_plain_with_fade(layers: list[Layer], seconds: float = LOOP) -> np.ndarray:
    """An ordinary render, with the fades a non-looping preset would use."""
    sink = ArraySink()
    render(
        layers,
        RenderConfig(
            SR,
            seconds,
            block_seconds=10.0,
            gain=1.25,
            fade_seconds=22.0,
            fade_max_denominator=5,
        ),
        sink,
    )
    return sink.result


def render_plain(layers: list[Layer], seconds: float = LOOP) -> np.ndarray:
    sink = ArraySink()
    render(
        layers,
        RenderConfig(SR, seconds, block_seconds=10.0, gain=1.25, fade_seconds=0.0),
        sink,
    )
    return sink.result


def join_step(audio: np.ndarray) -> float:
    """How far the signal moves from the last sample round to the first."""
    return float(np.max(np.abs(audio[0] - audio[-1])))


def worst_interior_step(audio: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(audio, axis=0))))


def worst_step_near_the_join(audio: np.ndarray, window: float = 0.1) -> float:
    """
    The largest step in the tenth of a second either side of the join.

    The fair comparison for a join step. Comparing against the largest step in
    the whole file is a coin toss when the join happens to land on a loud
    passage, and comparing against the average is far too lenient. What matters
    is whether the join looks like its own neighbourhood.
    """
    size = int(SR * window)
    neighbourhood = np.concatenate([audio[-size:], audio[:size]])
    return float(np.max(np.abs(np.diff(neighbourhood, axis=0))))


def level_step_across_join(audio: np.ndarray, window: float = 0.25) -> float:
    """Change in short-term level between the end of the file and its start."""
    size = int(SR * window)
    usable = audio[: len(audio) // size * size, 0].reshape(-1, size)
    rms = np.sqrt((usable**2).mean(axis=1))
    return float(abs(rms[0] - rms[-1]))


def interior_level_steps(audio: np.ndarray, window: float = 0.25) -> np.ndarray:
    size = int(SR * window)
    usable = audio[: len(audio) // size * size, 0].reshape(-1, size)
    rms = np.sqrt((usable**2).mean(axis=1))
    return np.abs(np.diff(rms))


def loudness_across_join(audio: np.ndarray, seconds: float = 3.0) -> float:
    """Level in the few seconds spanning the join, as the loop would play it."""
    size = int(SR * seconds)
    edge = np.concatenate([audio[-size:], audio[:size]])
    return float(np.sqrt((edge**2).mean()))


def median_loudness(audio: np.ndarray, seconds: float = 3.0) -> float:
    size = int(SR * seconds)
    usable = audio[: len(audio) // size * size, 0].reshape(-1, size)
    return float(np.median(np.sqrt((usable**2).mean(axis=1))))


# ---------------------------------------------------------------------------
# snapping
# ---------------------------------------------------------------------------


def test_snapping_puts_a_frequency_on_whole_cycles() -> None:
    for loop in (30.0, 120.0, 300.0):
        snapped = snap_frequency(335.796, loop)
        assert snapped * loop == pytest.approx(round(snapped * loop))
        assert abs(snapped - 335.796) <= 0.5 / loop


def test_snapping_error_shrinks_with_the_loop() -> None:
    """A five-minute loop moves the carrier by a hundredth of a cent."""
    short = abs(snap_frequency(335.796, 30.0) - 335.796)
    long = abs(snap_frequency(335.796, 300.0) - 335.796)
    assert long < short
    cents = 1200.0 * np.log2(snap_frequency(335.796, 300.0) / 335.796)
    assert abs(cents) < 0.02


def test_snapping_rejects_a_nonsense_loop() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        snap_frequency(100.0, 0.0)


def test_a_snapped_pair_keeps_both_ears_on_whole_cycles() -> None:
    """
    Not just the carrier — both ears, which is why the beat is snapped first.

    The beat goes onto an even multiple of one over the loop, so that
    ``carrier -/+ beat/2`` lands on whole cycles as well.
    """
    pair = BinauralPair(carrier=ROOT, beat=7.83, level=0.3)
    snapped, notes = pair.snapped_to_loop(LOOP)
    assert isinstance(snapped, BinauralPair)

    half = snapped.beat / 2
    for freq in (snapped.carrier - half, snapped.carrier + half):
        assert freq * LOOP == pytest.approx(round(freq * LOOP))

    assert snapped.beat == pytest.approx(7.83, abs=1.0 / LOOP)
    assert any("beat" in note for note in notes)


def test_a_snapped_pair_keeps_an_already_exact_beat() -> None:
    """4 Hz over a 120-second loop is 480 cycles. Nothing to change."""
    snapped, _ = BinauralPair(carrier=ROOT, beat=4.0, level=0.3).snapped_to_loop(LOOP)
    assert isinstance(snapped, BinauralPair)
    assert snapped.beat == 4.0


def test_snapped_layers_are_genuinely_periodic() -> None:
    """
    The real test: render two loops back to back and compare the halves.

    Not the join step, which only looks at two samples — the whole waveform,
    every sample of the second loop against the first.
    """
    snapped: list[Layer] = []
    for layer in tonal_layers():
        assert isinstance(layer, LoopableLayer)
        result = layer.snapped_to_loop(LOOP)
        assert result is not None
        snapped.append(result[0])

    sink = ArraySink()
    render(
        snapped,
        RenderConfig(SR, 2 * LOOP, block_seconds=10.0, gain=1.0, fade_seconds=0.0),
        sink,
    )
    audio = sink.result
    half = len(audio) // 2
    assert np.max(np.abs(audio[:half] - audio[half:])) < 1e-9


def test_a_closed_chord_bed_is_genuinely_periodic() -> None:
    events = plan_loop_progression(LOOP, np.random.default_rng(5), FAST_WALK)
    bed = ChordBed(
        events=events,
        root=ROOT,
        beat=BEAT,
        level=0.070,
        crossfade=FAST_WALK.crossfade,
    )
    snapped = bed.snapped_to_loop(LOOP)
    assert snapped is not None

    sink = ArraySink()
    render(
        [snapped[0]],
        RenderConfig(SR, 2 * LOOP, block_seconds=10.0, gain=1.0, fade_seconds=0.0),
        sink,
    )
    audio = sink.result
    half = len(audio) // 2
    assert np.max(np.abs(audio[:half] - audio[half:])) < 1e-9


def test_an_open_chord_bed_declines_to_snap() -> None:
    """
    It cannot close, so it says so and the engine crossfades it instead.

    Pretending otherwise would be worse than the crossfade: the chord at the
    end of an open progression usually shares voices with the chord at the
    start, and two copies of the same tone at different phases comb-filter.
    """
    events = plan_progression(LOOP, np.random.default_rng(5), FAST_WALK)
    bed = ChordBed(events=events, root=ROOT, beat=BEAT, level=0.070)
    assert bed.snapped_to_loop(LOOP) is None


def test_a_looping_chord_bed_rejects_a_progression_that_does_not_close() -> None:
    events = plan_progression(LOOP, np.random.default_rng(5), FAST_WALK)
    with pytest.raises(ValueError, match="tiles the loop"):
        ChordBed(events=events, root=ROOT, beat=BEAT, level=0.070, loop_seconds=LOOP)


# ---------------------------------------------------------------------------
# the closed progression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [5, 12, 2027])
def test_a_loop_progression_tiles_the_loop(seed: int) -> None:
    events = plan_loop_progression(LOOP, np.random.default_rng(seed), FAST_WALK)
    assert events[0].start == 0.0
    assert events[-1].end == pytest.approx(LOOP)
    for before, after in pairwise(events):
        assert before.end == pytest.approx(after.start)
    assert all(event.fade_in and event.fade_out for event in events)


@pytest.mark.parametrize("seed", [5, 12, 2027, 99])
def test_a_loop_progression_can_get_home(seed: int) -> None:
    """The last chord has to be one that leads back to the first."""
    events = plan_loop_progression(240.0, np.random.default_rng(seed))
    targets = [target for target, _ in TRANSITIONS[events[-1].chord.numeral]]
    assert events[0].chord.numeral in targets


def test_loop_progression_lengths_stay_musical() -> None:
    """Rescaling to close the loop moves each chord by a few percent, not more."""
    events = plan_loop_progression(300.0, np.random.default_rng(5))
    for event in events:
        assert 30.0 <= event.end - event.start <= 70.0


def test_a_loop_too_short_for_two_chords_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot hold two chords"):
        plan_loop_progression(50.0, np.random.default_rng(5))


def test_loop_progressions_are_deterministic() -> None:
    one = plan_loop_progression(LOOP, np.random.default_rng(5), FAST_WALK)
    two = plan_loop_progression(LOOP, np.random.default_rng(5), FAST_WALK)
    assert one == two


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


def test_a_tonal_loop_needs_no_crossfade_at_all() -> None:
    audio, result = render_loop(tonal_layers())
    assert result.loop_crossfade_frames == 0  # type: ignore[attr-defined]
    assert join_step(audio) <= worst_interior_step(audio)
    assert join_step(audio) <= worst_step_near_the_join(audio)


def test_a_tonal_loop_reports_what_it_moved() -> None:
    _, result = render_loop(tonal_layers())
    notes = result.notes  # type: ignore[attr-defined]
    assert any("carrier" in note for note in notes)
    assert any("pedal" in note for note in notes)


def test_the_full_mix_joins_without_a_step() -> None:
    """
    The step across the join is an ordinary step for that part of the signal.

    Both halves of the mix are continuous across the join by construction — the
    snapped layers because they are periodic, the crossfaded ones because the
    opening samples *are* the continuation of the closing ones — so what comes
    out should look like any other pair of adjacent samples nearby.
    """
    audio, result = render_loop(ocean_layers(closed=True))
    assert result.loop_crossfade_frames > 0  # type: ignore[attr-defined]
    assert join_step(audio) <= worst_step_near_the_join(audio)


def test_the_full_mix_joins_without_a_lurch() -> None:
    """
    The test the sample-step measure cannot make.

    A file that stops mid-wave and restarts on flat water has no *step* between
    the two samples either side of the join — both are small numbers — but the
    level jumps, and the level is what is actually heard. The same mix is
    rendered twice, once looping and once not, and only the looping one keeps
    its level change across the join inside the range of changes happening
    inside the file.
    """
    looped, _ = render_loop(ocean_layers(closed=True))
    plain = render_plain(ocean_layers(closed=True))

    inside = float(np.percentile(interior_level_steps(looped), 99))
    across_looped = level_step_across_join(looped)
    across_plain = level_step_across_join(plain)

    assert across_looped <= inside
    assert across_plain > 2.0 * across_looped


def test_an_ordinary_render_falls_into_a_hole_at_the_join() -> None:
    """
    The break the loop mode exists to remove, measured.

    An ordinary render fades in and out. Played on repeat that is not a click,
    it is a hole: three quarters of a minute where the piece drains away to
    nothing and then climbs back. The looped render carries its level straight
    through the join instead.
    """
    faded = render_plain_with_fade(ocean_layers(closed=True))
    assert loudness_across_join(faded) < 0.25 * median_loudness(faded)

    looped, _ = render_loop(ocean_layers(closed=True))
    assert loudness_across_join(looped) == pytest.approx(
        median_loudness(looped), rel=0.5
    )


def test_an_open_progression_still_loops_via_the_crossfade() -> None:
    """Second best, but it works: the fallback path has to hold up too."""
    audio, result = render_loop(ocean_layers(closed=False))
    assert result.loop_crossfade_frames > 0  # type: ignore[attr-defined]
    assert join_step(audio) <= worst_step_near_the_join(audio)
    across = level_step_across_join(audio)
    assert across <= float(np.percentile(interior_level_steps(audio), 99))


# ---------------------------------------------------------------------------
# the loop machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_seconds", [1.0, 3.0, 10.0])
def test_looping_is_block_size_invariant(block_seconds: float) -> None:
    """The crossfade must not depend on where the blocks fall either."""
    reference, _ = render_loop(ocean_layers(closed=True), block_seconds=10.0)
    other, _ = render_loop(ocean_layers(closed=True), block_seconds=block_seconds)
    assert np.max(np.abs(other - reference)) < 1e-6


def test_looping_is_deterministic() -> None:
    one, _ = render_loop(ocean_layers(closed=True))
    two, _ = render_loop(ocean_layers(closed=True))
    assert np.array_equal(one, two)


def test_a_looping_render_stays_under_the_ceiling() -> None:
    """
    Including the fold, where the crossfaded layers contribute twice.

    The headroom calculation allows root two for that, which is what
    equal-power gains can sum to.
    """
    _, result = render_loop(ocean_layers(closed=True))
    assert result.peak < 0.95  # type: ignore[attr-defined]
    assert result.clipped == 0  # type: ignore[attr-defined]


def test_a_loop_cannot_also_have_a_master_fade() -> None:
    with pytest.raises(ValueError, match="cannot have a master fade"):
        RenderConfig(SR, 60.0, fade_seconds=8.0, loop=LoopConfig())


def test_a_crossfade_longer_than_the_render_is_rejected() -> None:
    config = RenderConfig(
        SR, 5.0, block_seconds=1.0, fade_seconds=0.0, loop=LoopConfig(crossfade=30.0)
    )
    with pytest.raises(ValueError, match="does not fit"):
        render(ocean_layers(seconds=5.0, closed=False), config, ArraySink())


def test_loop_config_validates_itself() -> None:
    with pytest.raises(ValueError, match="crossfade must be positive"):
        LoopConfig(crossfade=0.0)


def test_the_crossfade_is_equal_power() -> None:
    """Folding uncorrelated material, so the powers add rather than the levels."""
    position = np.linspace(0.0, 100.0, 1001)
    rise, fall = _loop_gains(position, 100)
    assert np.max(np.abs(rise**2 + fall**2 - 1.0)) < 1e-12
    assert rise[0] == pytest.approx(0.0)
    assert fall[0] == pytest.approx(1.0)
