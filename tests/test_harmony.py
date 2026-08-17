"""
Stage 4: chords as ratio sets, the seeded walk, and the crossfades.

The headline property is the one that makes moving harmony possible at all:
however many voices are sounding and whatever chord they belong to, the two
ears differ by exactly one beat frequency.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest

import spectral
from violet import tuning
from violet.engine import ArraySink, RenderConfig, render
from violet.harmony import (
    CHORDS,
    TRANSITIONS,
    Chord,
    ChordEvent,
    WalkConfig,
    crossfade_gain,
    plan_progression,
    validate_transitions,
)
from violet.layers import ChordBed, Layer, Pedal, Span

if TYPE_CHECKING:
    from types import ModuleType

SR = 32000
ORIGIN_HZ = 83.949
ROOT = 335.796
BEAT = 4.0
VOICE_LEVEL = 0.070

#: How the prototype spelled these chords, for a root of E. Kept here in the
#: tests, where it belongs: it is a fact about one render, not about harmony.
PROTOTYPE_NAMES = {
    "i": "Em",
    "i7": "Em7",
    "VI": "C",
    "III": "G",
    "iv": "Am",
    "VII": "D",
}


def chord_bed(
    seconds: float = 20.0,
    seed: int = 5,
    beat: float = BEAT,
    root: float = ROOT,
) -> ChordBed:
    events = plan_progression(seconds, np.random.default_rng(seed))
    return ChordBed(events=events, root=root, beat=beat, level=VOICE_LEVEL)


def render_array(layers: list[Layer], config: RenderConfig) -> np.ndarray:
    sink = ArraySink()
    render(layers, config, sink)
    return sink.result


# ---------------------------------------------------------------------------
# chords as ratio sets
# ---------------------------------------------------------------------------


def test_chord_ratios_match_the_prototype(reference_ocean: ModuleType) -> None:
    """
    Every voicing reproduces exactly, from the just table rather than literals.

    The prototype wrote its voicings as raw fractions — 6/5, 12/5, 9/4. Each
    one turns out to be a degree of the five-limit table plus some whole
    octaves, which is why they can be rebuilt from it. Compared as exact
    floats, not approximately: `float(Fraction(12, 5))` and `12 / 5` are the
    same double.
    """
    assert set(CHORDS) == set(PROTOTYPE_NAMES)
    for numeral, chord in CHORDS.items():
        theirs = reference_ocean.CHORDS[PROTOTYPE_NAMES[numeral]]
        ours = [float(ratio) for ratio in chord.ratios]
        assert ours == theirs, numeral


def test_transitions_match_the_prototype(reference_ocean: ModuleType) -> None:
    """
    Including the order of each chord's targets, which the seed depends on.

    Reordering the options would change which target a given random draw
    picks, so this is not a cosmetic comparison.
    """
    assert set(TRANSITIONS) == set(PROTOTYPE_NAMES)
    for numeral, moves in TRANSITIONS.items():
        prototype_moves = reference_ocean.MOVES[PROTOTYPE_NAMES[numeral]]
        theirs = [tuple(move) for move in prototype_moves]
        ours = [(PROTOTYPE_NAMES[target], weight) for target, weight in moves]
        assert ours == theirs, numeral


def test_walk_constants_match_the_prototype(reference_ocean: ModuleType) -> None:
    walk = WalkConfig()
    assert walk.crossfade == reference_ocean.XF
    assert walk.min_seconds == reference_ocean.DUR_MIN
    assert walk.max_seconds == reference_ocean.DUR_MAX


def test_ratios_are_exact_fractions() -> None:
    assert CHORDS["i"].ratios == (
        Fraction(1, 1),
        Fraction(6, 5),
        Fraction(3, 2),
        Fraction(9, 4),
    )
    assert CHORDS["VI"].ratios == (
        Fraction(8, 5),
        Fraction(2, 1),
        Fraction(12, 5),
        Fraction(3, 1),
    )


def test_every_chord_has_four_voices() -> None:
    for chord in CHORDS.values():
        assert chord.voices == 4


def test_the_voicings_span_two_octaves_above_the_root() -> None:
    """
    The register these voicings actually occupy, which is not what was claimed.

    The prototype's comment says the voices "sit between roughly 250 and
    900 Hz". They do not: ``VI`` puts its top voice at 3x the root and ``VII``
    at 18/5, which for a 335.8 Hz root is 1007 Hz and 1209 Hz. Nothing is
    wrong with the sound — a quiet voice at 1.2 kHz is bright, not harsh — but
    the range is a fifth wider at the top than advertised, and that is worth
    knowing before choosing a root.
    """
    lowest = min(min(chord.frequencies(ROOT)) for chord in CHORDS.values())
    highest = max(max(chord.frequencies(ROOT)) for chord in CHORDS.values())

    assert lowest == pytest.approx(ROOT)
    assert highest == pytest.approx(ROOT * 18 / 5)
    assert highest == pytest.approx(1208.8656)

    # Every voice stays inside the range where the percept holds up.
    for numeral, chord in CHORDS.items():
        for freq in chord.frequencies(ROOT):
            assert 250.0 <= freq <= 1250.0, (numeral, freq)


def test_chord_voices_are_in_ascending_order() -> None:
    for chord in CHORDS.values():
        ratios = [float(r) for r in chord.ratios]
        assert ratios == sorted(ratios)


def test_note_labels_recover_the_prototypes_spelling() -> None:
    """
    The E-spelling emerges as a result rather than being carried forward.

    Named against the tuning the root itself implies, so what is left is the
    just interval's own deviation and not the root's 32-cent offset on top.
    """
    expected = {
        "i": ("E4", "G4", "B4", "F#5"),
        "i7": ("E4", "G4", "B4", "D5"),
        "VI": ("C5", "E5", "G5", "B5"),
        "III": ("G4", "B4", "D5", "G5"),
        "iv": ("A4", "C5", "E5", "G5"),
        "VII": ("D5", "F#5", "A5", "D6"),
    }
    for numeral, chord in CHORDS.items():
        assert chord.note_labels(ROOT) == expected[numeral], numeral


def test_note_labels_transpose_with_the_root() -> None:
    """A different root gives different letters and the same numerals."""
    labels = CHORDS["i"].note_labels(440.0)
    assert [name.rstrip("0123456789") for name in labels] == ["A", "C", "E", "B"]

    labels = CHORDS["VI"].note_labels(440.0)
    assert [name.rstrip("0123456789") for name in labels] == ["F", "A", "C", "E"]


def test_note_labels_accept_an_explicit_reference() -> None:
    at_concert = CHORDS["i"].note_labels(ROOT, a4=440.0)
    assert at_concert[0] == "E4"


def test_chord_frequencies_scale_with_the_root() -> None:
    one = CHORDS["iv"].frequencies(300.0)
    two = CHORDS["iv"].frequencies(600.0)
    for low, high in zip(one, two, strict=True):
        assert high == pytest.approx(2.0 * low)


# ---------------------------------------------------------------------------
# the transition graph
# ---------------------------------------------------------------------------


def test_the_shipped_graph_is_valid() -> None:
    validate_transitions(TRANSITIONS, CHORDS, "i")


def test_every_chord_is_reachable_from_the_tonic() -> None:
    seen = {"i"}
    frontier = ["i"]
    while frontier:
        for target, _weight in TRANSITIONS[frontier.pop()]:
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    assert seen == set(CHORDS)


@pytest.mark.parametrize(
    ("transitions", "message"),
    [
        ({"i": ()}, "dead end"),
        ({"i": (("nope", 1),)}, "does not exist"),
        ({"i": (("i", 0),)}, "must be positive"),
        ({"nope": (("i", 1),)}, "not a known chord"),
    ],
)
def test_invalid_graphs_are_rejected(
    transitions: dict[str, tuple[tuple[str, int], ...]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_transitions(transitions, CHORDS, "i")


def test_a_graph_missing_a_chord_is_rejected() -> None:
    with pytest.raises(ValueError, match="no transitions defined"):
        validate_transitions({"i": (("i", 1),)}, CHORDS, "i")


def test_an_unknown_start_chord_is_rejected() -> None:
    with pytest.raises(ValueError, match="start chord"):
        WalkConfig(start="V")


def test_a_crossfade_longer_than_the_chord_is_rejected() -> None:
    """It would still be arriving when it started to leave."""
    with pytest.raises(ValueError, match="cannot carry"):
        WalkConfig(crossfade=40.0, min_seconds=38.0)


def test_walk_config_validates_its_lengths() -> None:
    with pytest.raises(ValueError, match="0 < min <= max"):
        WalkConfig(min_seconds=60.0, max_seconds=40.0)
    with pytest.raises(ValueError, match="crossfade must be positive"):
        WalkConfig(crossfade=0.0)


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------


def test_walk_matches_the_prototype(reference_ocean: ModuleType) -> None:
    """Same seed, same progression, to the exact second."""
    for seed in (5, 12, 2027):
        theirs = reference_ocean.plan_chords(600.0, np.random.default_rng(seed))
        ours = plan_progression(600.0, np.random.default_rng(seed))
        assert len(ours) == len(theirs)
        for event, (name, start, end) in zip(ours, theirs, strict=True):
            assert PROTOTYPE_NAMES[event.chord.numeral] == name
            assert event.start == start
            assert event.end == end


def test_the_same_seed_gives_the_same_progression() -> None:
    first = plan_progression(600.0, np.random.default_rng(5))
    second = plan_progression(600.0, np.random.default_rng(5))
    assert first == second


def test_different_seeds_give_different_progressions() -> None:
    first = plan_progression(600.0, np.random.default_rng(5))
    second = plan_progression(600.0, np.random.default_rng(6))
    assert [e.chord.numeral for e in first] != [e.chord.numeral for e in second]
    assert [e.end for e in first] != [e.end for e in second]


def test_the_walk_covers_the_render_and_overshoots_it() -> None:
    seconds = 300.0
    events = plan_progression(seconds, np.random.default_rng(5))
    assert events[-1].end > seconds
    assert events[-2].start <= seconds


def test_chords_are_contiguous() -> None:
    """
    No gaps and no overlaps in the plan.

    Equal-power crossfading depends on it: the outgoing chord's fade window
    and the incoming chord's fade window have to be the same window.
    """
    events = plan_progression(600.0, np.random.default_rng(5))
    for before, after in pairwise(events):
        assert before.end == after.start


def test_chord_lengths_stay_inside_their_bounds() -> None:
    walk = WalkConfig()
    for event in plan_progression(3600.0, np.random.default_rng(11)):
        held = event.end - event.start
        assert walk.min_seconds <= held <= walk.max_seconds


def test_the_walk_only_takes_legal_moves() -> None:
    events = plan_progression(3600.0, np.random.default_rng(11))
    for before, after in pairwise(events):
        targets = [target for target, _ in TRANSITIONS[before.chord.numeral]]
        assert after.chord.numeral in targets


def test_the_walk_starts_on_the_tonic_and_visits_everything() -> None:
    events = plan_progression(7200.0, np.random.default_rng(3))
    assert events[0].chord.numeral == "i"
    assert {event.chord.numeral for event in events} == set(CHORDS)


def test_the_first_chord_does_not_fade_in_and_the_last_does_not_fade_out() -> None:
    events = plan_progression(300.0, np.random.default_rng(5))
    assert events[0].fade_in is False
    assert events[-1].fade_out is False
    assert all(event.fade_in for event in events[1:])
    assert all(event.fade_out for event in events[:-1])


def test_a_walk_can_use_a_custom_graph() -> None:
    two_chords = {"i": CHORDS["i"], "iv": CHORDS["iv"]}
    walk = WalkConfig(
        transitions={"i": (("iv", 1),), "iv": (("i", 1),)},
        chords=two_chords,
        min_seconds=20.0,
        max_seconds=20.0,
        crossfade=8.0,
    )
    events = plan_progression(100.0, np.random.default_rng(1), walk)
    numerals = [event.chord.numeral for event in events]
    assert numerals == ["i", "iv"] * (len(numerals) // 2)


# ---------------------------------------------------------------------------
# crossfades
# ---------------------------------------------------------------------------


def crossfade_window(
    crossfade: float = 16.0,
) -> tuple[ChordEvent, ChordEvent, np.ndarray]:
    """A pair of adjoining events and the times across their shared join."""
    outgoing = ChordEvent(CHORDS["i"], 0.0, 40.0, fade_in=False, fade_out=True)
    incoming = ChordEvent(CHORDS["VI"], 40.0, 80.0, fade_in=True, fade_out=True)
    t = np.linspace(40.0 - crossfade, 40.0 + crossfade, 4001)
    return outgoing, incoming, t


def test_the_crossfade_is_equal_power() -> None:
    """
    ``g_out**2 + g_in**2`` is one all the way through the transition.

    Equal *amplitude* would sum to one instead, and dip 3 dB in the middle of
    every chord change — heard as a sag, not as harmony moving.
    """
    outgoing, incoming, t = crossfade_window()
    out_gain = crossfade_gain(t, outgoing, 16.0)
    in_gain = crossfade_gain(t, incoming, 16.0)
    power = out_gain**2 + in_gain**2
    assert np.max(np.abs(power - 1.0)) < 1e-12


def test_the_crossfade_hands_over_completely() -> None:
    outgoing, incoming, t = crossfade_window()
    out_gain = crossfade_gain(t, outgoing, 16.0)
    in_gain = crossfade_gain(t, incoming, 16.0)

    assert out_gain[0] == pytest.approx(1.0)
    assert in_gain[0] == pytest.approx(0.0)
    assert out_gain[-1] == pytest.approx(0.0)
    assert in_gain[-1] == pytest.approx(1.0)

    midpoint = len(t) // 2
    assert out_gain[midpoint] == pytest.approx(in_gain[midpoint], abs=1e-3)


def test_gain_is_exactly_zero_outside_the_support() -> None:
    """What lets a renderer skip an event without the blocks mattering."""
    event = ChordEvent(CHORDS["i"], 100.0, 140.0)
    low, high = event.support(16.0)
    assert low == 92.0
    assert high == 148.0

    t = np.linspace(0.0, 240.0, 24001)
    gain = crossfade_gain(t, event, 16.0)
    assert np.all(gain[t < low] == 0.0)
    assert np.all(gain[t >= high] == 0.0)
    assert np.all(gain[(t > low + 16.0) & (t < high - 16.0)] == 1.0)


def test_an_unfading_edge_has_unbounded_support() -> None:
    first = ChordEvent(CHORDS["i"], 0.0, 40.0, fade_in=False)
    assert first.support(16.0)[0] == -np.inf
    last = ChordEvent(CHORDS["i"], 0.0, 40.0, fade_out=False)
    assert last.support(16.0)[1] == np.inf


def test_crossfade_gain_is_a_function_of_absolute_time() -> None:
    event = ChordEvent(CHORDS["i"], 10.0, 50.0)
    whole = crossfade_gain(Span(0, SR * 60, SR).t, event, 16.0)
    part = crossfade_gain(Span(SR * 20, SR * 30, SR).t, event, 16.0)
    assert np.array_equal(whole[SR * 20 : SR * 30], part)


# ---------------------------------------------------------------------------
# constraint 3: one beat, whatever the chord
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("numeral", list(CHORDS))
@pytest.mark.parametrize("beat", [2.0, 4.0, 7.83])
def test_every_voice_of_every_chord_beats_at_the_same_rate(
    numeral: str, beat: float
) -> None:
    """
    FFT both ears, pair the peaks up, and every pair differs by the beat.

    This is the constraint the whole design exists to satisfy: eight tones
    sounding, four in each ear, and exactly one beat frequency between them.
    """
    held = ChordEvent(CHORDS[numeral], 0.0, 40.0, fade_in=False)
    bed = ChordBed(events=(held,), root=ROOT, beat=beat, level=VOICE_LEVEL)
    audio = render_array(
        [bed],
        RenderConfig(SR, duration=5.0, block_seconds=1.0, fade_seconds=0.0),
    )

    pairs = spectral.matched_pairs(
        audio[:, 0], audio[:, 1], SR, voices=4, min_separation_hz=20.0
    )
    expected = CHORDS[numeral].frequencies(ROOT)
    assert len(pairs) == 4

    for (left_hz, right_hz), centre in zip(pairs, expected, strict=True):
        assert right_hz - left_hz == pytest.approx(beat, abs=0.02)
        assert left_hz == pytest.approx(centre - beat / 2, abs=0.02)
        assert right_hz == pytest.approx(centre + beat / 2, abs=0.02)


def test_the_beat_survives_a_chord_change() -> None:
    """Mid-crossfade, with eight tones per ear, there is still one beat."""
    events = (
        ChordEvent(CHORDS["i"], 0.0, 40.0, fade_in=False),
        ChordEvent(CHORDS["VI"], 40.0, 80.0),
    )
    bed = ChordBed(events=events, root=ROOT, beat=BEAT, level=VOICE_LEVEL)

    # Two and a half seconds either side of the join, where both chords are
    # sounding at close to equal power.
    start = int((40.0 - 2.5) * SR)
    audio = render_array(
        [bed],
        RenderConfig(SR, duration=45.0, block_seconds=1.0, fade_seconds=0.0),
    )
    window = audio[start : start + 5 * SR]

    pairs = spectral.matched_pairs(
        window[:, 0], window[:, 1], SR, voices=8, min_separation_hz=15.0
    )
    for left_hz, right_hz in pairs:
        assert right_hz - left_hz == pytest.approx(BEAT, abs=0.05)


def test_a_zero_beat_chord_bed_has_identical_channels() -> None:
    """The null condition holds for a whole progression, not just one tone."""
    audio = render_array(
        [chord_bed(beat=0.0)],
        RenderConfig(SR, duration=10.0, block_seconds=2.0),
    )
    assert np.array_equal(audio[:, 0], audio[:, 1])


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_seconds", [0.25, 1.0, 3.0, 10.0])
def test_chord_bed_is_block_size_invariant(block_seconds: float) -> None:
    reference = render_array(
        [chord_bed(), Pedal(ORIGIN_HZ, 0.080)],
        RenderConfig(SR, duration=8.0, block_seconds=10.0),
    )
    other = render_array(
        [chord_bed(), Pedal(ORIGIN_HZ, 0.080)],
        RenderConfig(SR, duration=8.0, block_seconds=block_seconds),
    )
    assert np.max(np.abs(other - reference)) < 1e-6


def test_chord_bed_is_block_size_invariant_across_a_chord_change() -> None:
    """The interesting boundary case: a block that straddles a crossfade."""
    events = (
        ChordEvent(CHORDS["i"], 0.0, 6.0, fade_in=False),
        ChordEvent(CHORDS["VI"], 6.0, 46.0),
    )
    bed = ChordBed(
        events=events, root=ROOT, beat=BEAT, level=VOICE_LEVEL, crossfade=4.0
    )
    config = RenderConfig(SR, duration=10.0, block_seconds=10.0, fade_seconds=0.0)
    reference = render_array([bed], config)

    for block_seconds in (0.1, 0.33, 1.0, 2.5):
        other = render_array(
            [bed],
            RenderConfig(
                SR,
                duration=10.0,
                block_seconds=block_seconds,
                fade_seconds=0.0,
            ),
        )
        assert np.max(np.abs(other - reference)) < 1e-6, block_seconds


def test_chord_bed_declares_a_true_peak_bound() -> None:
    # Long enough to cross a chord change, which is where two chords sound
    # at once and the bound is actually tested.
    bed = chord_bed(seconds=70.0)
    audio = render_array(
        [bed], RenderConfig(SR, duration=70.0, block_seconds=10.0, gain=1.0)
    )
    assert np.max(np.abs(audio)) <= bed.peak
    assert bed.peak == pytest.approx(VOICE_LEVEL * 4 * np.sqrt(2.0))


def test_chord_bed_stays_under_the_ceiling() -> None:
    layers: list[Layer] = [chord_bed(seconds=70.0), Pedal(ORIGIN_HZ, 0.080)]
    result = render(
        layers,
        RenderConfig(SR, duration=70.0, block_seconds=10.0, gain=1.25),
        ArraySink(),
    )
    assert result.peak < 0.95
    assert result.clipped == 0


def test_chord_bed_is_stateless() -> None:
    bed = chord_bed()
    assert bed.stateful is False
    assert isinstance(bed, Layer)


def test_chord_bed_validates_itself() -> None:
    with pytest.raises(ValueError, match="level must not be negative"):
        ChordBed(events=(), root=ROOT, beat=BEAT, level=-0.1)
    with pytest.raises(ValueError, match="root must be positive"):
        ChordBed(events=(), root=0.0, beat=BEAT, level=0.07)


def test_an_empty_chord_bed_is_silent() -> None:
    bed = ChordBed(events=(), root=ROOT, beat=BEAT, level=VOICE_LEVEL)
    assert bed.peak == 0.0
    audio = render_array(
        [bed], RenderConfig(SR, duration=1.0, block_seconds=1.0, fade_seconds=0.0)
    )
    assert np.array_equal(audio, np.zeros_like(audio))


def test_different_seeds_sound_different_but_not_immediately() -> None:
    """
    A seed changes the progression, and changes nothing for half a minute.

    Every walk starts on the tonic and holds it for at least 38 seconds, so
    two seeds give bit-identical audio until the first crossfade begins. Worth
    knowing before comparing two seeds by ear, and worth knowing for the
    blinded trial: a seed is not a way to make two renders sound different
    from the start.
    """
    plan_a = plan_progression(200.0, np.random.default_rng(5))
    plan_b = plan_progression(200.0, np.random.default_rng(6))
    assert plan_a[0].chord == plan_b[0].chord
    assert plan_a[0].end != plan_b[0].end

    first_join = min(plan_a[0].end, plan_b[0].end)
    seconds = first_join + 4.0
    config = RenderConfig(SR, duration=seconds, block_seconds=10.0, fade_seconds=0.0)
    one = render_array([ChordBed(plan_a, ROOT, BEAT, VOICE_LEVEL)], config)
    two = render_array([ChordBed(plan_b, ROOT, BEAT, VOICE_LEVEL)], config)

    quiet_until = int((first_join - WalkConfig().crossfade / 2.0) * SR)
    assert np.array_equal(one[:quiet_until], two[:quiet_until])
    assert not np.allclose(one, two)


def test_the_same_seed_sounds_identical() -> None:
    config = RenderConfig(SR, duration=10.0, block_seconds=3.0)
    one = render_array([chord_bed(seconds=10.0, seed=5)], config)
    two = render_array([chord_bed(seconds=10.0, seed=5)], config)
    assert np.array_equal(one, two)


def test_voices_breathe_independently() -> None:
    """
    Each voice drifts on its own period, so the chord changes shape.

    If they shared a period the whole chord would pulse, which is a different
    and much less interesting sound.
    """
    held = ChordEvent(CHORDS["i"], 0.0, 300.0, fade_in=False, fade_out=False)
    bed = ChordBed(events=(held,), root=ROOT, beat=BEAT, level=VOICE_LEVEL)
    t = Span(0, SR * 60, SR).t
    envelopes = [bed.breathing.gain(t, voice) for voice in range(4)]

    for one, two in pairwise(envelopes):
        assert not np.allclose(one, two)
        correlation = np.corrcoef(one, two)[0, 1]
        assert abs(correlation) < 0.95


def test_chord_note_labels_are_reported_for_the_root_in_use() -> None:
    """Sanity check on the naming path a CLI would use."""
    events = plan_progression(120.0, np.random.default_rng(5))
    root = tuning.carrier_for(ORIGIN_HZ)
    for event in events[:3]:
        labels = event.chord.note_labels(root)
        assert len(labels) == event.chord.voices
        assert all(label[0] in "ABCDEFG" for label in labels)


def test_chord_is_hashable_and_comparable() -> None:
    assert CHORDS["i"] == Chord("i", CHORDS["i"].degrees)
    assert len({CHORDS["i"], CHORDS["i"], CHORDS["VI"]}) == 2
