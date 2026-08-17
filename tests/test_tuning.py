"""
Stage 2: the frequency arithmetic.

These are property tests where a property exists — the note is preserved
across an octave shift, the implied reference makes the frequency exact, the
carrier always lands in the window — and exact-value tests where the value is
the point, as with the just ratios and the origin frequency's maths.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from violet import tuning

if TYPE_CHECKING:
    from types import ModuleType

# The tone-test result the project is named for. It is an input here, never a
# constant in the package: see tests/test_architecture.py.
ORIGIN_HZ = 83.949

# Where the origin frequency lands, from the README's table.
ORIGIN_NOTE = "E2"
ORIGIN_CENTS = 32.098
ORIGIN_IMPLIED_A4 = 448.234
ORIGIN_CARRIER = 335.796
ORIGIN_OCTAVES_TO_LIGHT = 43
ORIGIN_NM = 405.990
ORIGIN_THZ = 738.423


def sweep(lo: float, hi: float, per_octave: int = 7) -> list[float]:
    """Geometric sweep from ``lo`` to ``hi``, deliberately off the semitones."""
    out: list[float] = []
    f = lo
    factor = 2.0 ** (1.0 / per_octave)
    while f <= hi:
        out.append(f)
        f *= factor
    return out


AUDIBLE = sweep(16.0, 16000.0)


# ---------------------------------------------------------------------------
# notes and cents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "label", "cents"),
    [
        (440.0, "A4", 0.0),
        (261.6255653, "C4", 0.0),
        (82.4068892282175, "E2", 0.0),
        (ORIGIN_HZ, ORIGIN_NOTE, ORIGIN_CENTS),
        (27.5, "A0", 0.0),
        (4186.009044809578, "C8", 0.0),
    ],
)
def test_note_of_known_values(freq: float, label: str, cents: float) -> None:
    note = tuning.note_of(freq)
    assert note.label == label
    assert note.cents == pytest.approx(cents, abs=1e-3)


def test_origin_frequency_is_sharp_of_e2() -> None:
    """The whole naming of the project rests on this one line."""
    note = tuning.note_of(ORIGIN_HZ)
    assert note.label == "E2"
    assert note.direction == "sharp"
    assert note.cents == pytest.approx(ORIGIN_CENTS, abs=1e-3)


def test_note_of_always_snaps_within_a_semitone() -> None:
    for freq in AUDIBLE:
        assert abs(tuning.note_of(freq).cents) <= 50.0, freq


def test_note_of_round_trips_through_midi() -> None:
    for freq in AUDIBLE:
        note = tuning.note_of(freq)
        assert tuning.midi_to_frequency(note.midi) == pytest.approx(note.exact_hz)


def test_note_of_tracks_the_naming_reference() -> None:
    """Naming against a different A4 changes the name, never the frequency."""
    at_440 = tuning.note_of(ORIGIN_HZ, a4=440.0)
    at_448 = tuning.note_of(ORIGIN_HZ, a4=ORIGIN_IMPLIED_A4)
    assert at_440.label == at_448.label == "E2"
    assert abs(at_448.cents) < abs(at_440.cents)
    assert at_448.cents == pytest.approx(0.0, abs=1e-3)


def test_note_to_frequency_known_values() -> None:
    assert tuning.note_to_frequency("A4") == pytest.approx(440.0)
    assert tuning.note_to_frequency("E2") == pytest.approx(82.40688922821748)
    flat = tuning.note_to_frequency("Bb3")
    assert flat == pytest.approx(tuning.note_to_frequency("A#3"))
    assert tuning.note_to_frequency("C-1") == pytest.approx(8.175798915643707)


def test_note_to_frequency_round_trips_the_full_keyboard() -> None:
    for midi in range(21, 109):
        freq = tuning.midi_to_frequency(midi)
        note = tuning.note_of(freq)
        assert note.midi == midi
        assert tuning.note_to_frequency(note.label) == pytest.approx(freq)


def test_note_to_frequency_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        tuning.note_to_frequency("H4")


def test_cents_round_trip() -> None:
    for freq in AUDIBLE:
        for cents in (-1200.0, -31.7, 0.0, 5.0, 701.955):
            moved = tuning.transpose_cents(freq, cents)
            assert tuning.cents_between(freq, moved) == pytest.approx(cents, abs=1e-9)


def test_an_octave_is_twelve_hundred_cents() -> None:
    for freq in AUDIBLE:
        assert tuning.cents_between(freq, freq * 2.0) == pytest.approx(1200.0)
        assert tuning.cents_between(freq, freq / 2.0) == pytest.approx(-1200.0)


# ---------------------------------------------------------------------------
# implied reference
# ---------------------------------------------------------------------------


def test_implied_reference_of_the_origin_frequency() -> None:
    """83.949 Hz reads sharp at A440 because it is in tune at A448."""
    assert tuning.implied_reference(ORIGIN_HZ) == pytest.approx(
        ORIGIN_IMPLIED_A4, abs=1e-3
    )


def test_implied_reference_makes_any_frequency_exact() -> None:
    for freq in AUDIBLE:
        implied = tuning.implied_reference(freq)
        assert tuning.note_of(freq, a4=implied).cents == pytest.approx(0.0, abs=1e-9)


def test_implied_reference_of_a_in_tune_frequency_is_the_reference() -> None:
    for midi in range(21, 109):
        implied = tuning.implied_reference(tuning.midi_to_frequency(midi))
        assert implied == pytest.approx(440.0)


# ---------------------------------------------------------------------------
# carrier selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "shift", "carrier"),
    [
        # The awkward inputs: far below, far above, and exactly on each edge.
        (20.0, 4, 320.0),
        (2000.0, -2, 500.0),
        (250.0, 0, 250.0),
        (520.0, 0, 520.0),
        # A hair either side of the edges must step, not stay.
        (249.999, 1, 499.998),
        (520.001, -1, 260.0005),
        # And the cases the README quotes.
        (ORIGIN_HZ, 2, ORIGIN_CARRIER),
        (220.0, 1, 440.0),
        (256.0, 0, 256.0),
    ],
)
def test_auto_octaves_awkward_inputs(freq: float, shift: int, carrier: float) -> None:
    assert tuning.auto_octaves(freq) == shift
    assert tuning.carrier_for(freq) == pytest.approx(carrier)


def test_carrier_always_lands_in_the_window() -> None:
    for freq in sweep(0.5, 40000.0, per_octave=13):
        carrier = tuning.carrier_for(freq)
        assert tuning.CARRIER_LO <= carrier <= tuning.CARRIER_HI, freq


def test_carrier_preserves_the_note() -> None:
    """Whole octaves only: the carrier is the same pitch class as the base."""
    for freq in sweep(0.5, 40000.0, per_octave=13):
        base_note = tuning.note_of(freq)
        carrier_note = tuning.note_of(tuning.carrier_for(freq))
        assert carrier_note.name == base_note.name, freq
        assert carrier_note.cents == pytest.approx(base_note.cents, abs=1e-9)


def test_auto_octaves_takes_the_smallest_shift() -> None:
    tol = 1e-12
    for freq in sweep(0.5, 40000.0, per_octave=13):
        landing = [
            n
            for n in range(-20, 21)
            if tuning.CARRIER_LO * (1 - tol)
            <= freq * 2.0**n
            <= tuning.CARRIER_HI * (1 + tol)
        ]
        assert tuning.auto_octaves(freq) == min(landing, key=abs), freq


def test_auto_octaves_matches_the_prototype(reference_tuning: ModuleType) -> None:
    """Parity with reference/tuning.py, over its whole usable range."""
    for freq in sweep(0.5, 40000.0, per_octave=13):
        assert tuning.auto_octaves(freq) == reference_tuning.auto_octaves(freq), freq


def test_auto_octaves_accepts_a_custom_window() -> None:
    assert tuning.auto_octaves(100.0, lo=1000.0, hi=2000.0) == 4
    assert tuning.carrier_for(100.0, lo=1000.0, hi=2000.0) == pytest.approx(1600.0)


def test_auto_octaves_rejects_a_window_it_cannot_satisfy() -> None:
    """A window narrower than an octave cannot catch every frequency."""
    with pytest.raises(ValueError, match="no whole-octave shift"):
        tuning.auto_octaves(300.0, lo=400.0, hi=500.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_frequency_validation(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        tuning.auto_octaves(bad)
    with pytest.raises(ValueError, match="finite and positive"):
        tuning.note_of(bad)


def test_auto_octaves_rejects_a_backwards_window() -> None:
    with pytest.raises(ValueError, match="0 < lo < hi"):
        tuning.auto_octaves(300.0, lo=500.0, hi=400.0)


# ---------------------------------------------------------------------------
# just intonation
# ---------------------------------------------------------------------------


def test_the_fifth_is_701_955_cents_not_700() -> None:
    """The syntonic point of the whole exercise."""
    just = tuning.ratio_cents(tuning.JUST_RATIOS["perfect_fifth"])
    assert just == pytest.approx(701.955, abs=5e-4)
    assert abs(just - 700.0) > 1.9  # audibly not equal temperament


@pytest.mark.parametrize(
    ("interval", "cents"),
    [
        ("unison", 0.0),
        ("minor_second", 111.731),
        ("major_second", 203.910),
        ("minor_third", 315.641),
        ("major_third", 386.314),
        ("perfect_fourth", 498.045),
        ("tritone", 590.224),
        ("perfect_fifth", 701.955),
        ("minor_sixth", 813.686),
        ("major_sixth", 884.359),
        ("minor_seventh", 1017.596),
        ("major_seventh", 1088.269),
        ("octave", 1200.0),
    ],
)
def test_just_interval_sizes(interval: str, cents: float) -> None:
    assert tuning.ratio_cents(tuning.just_ratio(interval)) == pytest.approx(
        cents, abs=5e-4
    )


def test_just_ratios_are_exact_fractions() -> None:
    """Held as fractions, so 3/2 is 3/2 and not 1.4999999999999998."""
    assert tuning.JUST_RATIOS["perfect_fifth"] == Fraction(3, 2)
    assert tuning.JUST_RATIOS["major_third"] == Fraction(5, 4)
    # A just minor third stacked on a just major third is exactly a fifth.
    stacked = tuning.JUST_RATIOS["minor_third"] * tuning.JUST_RATIOS["major_third"]
    assert stacked == Fraction(3, 2)
    for ratio in tuning.JUST_SEMITONES:
        assert isinstance(ratio, Fraction)


def test_just_table_is_ordered_and_spans_one_octave() -> None:
    assert len(tuning.JUST_SEMITONES) == 13
    assert tuning.JUST_SEMITONES[0] == Fraction(1, 1)
    assert tuning.JUST_SEMITONES[-1] == Fraction(2, 1)
    sizes = [tuning.ratio_cents(r) for r in tuning.JUST_SEMITONES]
    assert sizes == sorted(sizes)


def test_just_table_stays_near_equal_temperament() -> None:
    """Each ratio is the just version of its semitone, not a different note."""
    for semitones, ratio in enumerate(tuning.JUST_SEMITONES):
        error = tuning.ratio_cents(ratio) - 100.0 * semitones
        assert abs(error) < 22.0, (semitones, error)


def test_just_ratio_rejects_unknown_intervals() -> None:
    with pytest.raises(KeyError, match="unknown interval"):
        tuning.just_ratio("perfect_eleventh")


def test_ratio_cents_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        tuning.ratio_cents(0.0)


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------


def test_the_origin_frequency_maps_to_violet() -> None:
    colour = tuning.to_visible(ORIGIN_HZ)
    assert colour is not None
    assert colour.octaves == ORIGIN_OCTAVES_TO_LIGHT
    assert colour.thz == pytest.approx(ORIGIN_THZ, abs=5e-3)
    assert colour.nm == pytest.approx(ORIGIN_NM, abs=5e-3)
    assert colour.hex == "#8100CC"


def test_visible_mapping_is_consistent() -> None:
    for freq in AUDIBLE:
        colour = tuning.to_visible(freq)
        if colour is None:
            continue
        assert tuning.VISIBLE_NM[0] <= colour.nm <= tuning.VISIBLE_NM[1], freq
        shifted = freq * 2.0**colour.octaves
        assert shifted == pytest.approx(colour.thz * 1e12)
        assert tuning.C_LIGHT / shifted * 1e9 == pytest.approx(colour.nm)


def test_an_octave_apart_is_the_same_colour() -> None:
    for freq in AUDIBLE[:40]:
        here = tuning.to_visible(freq)
        up = tuning.to_visible(freq * 2.0)
        assert (here is None) == (up is None)
        if here is not None and up is not None:
            assert here.nm == pytest.approx(up.nm)
            assert up.octaves == here.octaves - 1


def test_some_frequencies_step_over_the_visible_band() -> None:
    """
    Not every frequency has a colour.

    The band spans a ratio of 1.974, less than an octave, so an octave series
    can jump from 755 nm straight to 377 nm without landing inside.
    """
    gap_hz = tuning.C_LIGHT / 755e-9 / 2**42
    assert tuning.to_visible(gap_hz) is None
    assert 380.0 / 2.0 < tuning.C_LIGHT / (gap_hz * 2**43) * 1e9 < 380.0


def test_wavelength_hex_format_and_bounds() -> None:
    assert tuning.wavelength_hex(379.0) == "#000000"
    assert tuning.wavelength_hex(781.0) == "#000000"
    for nm in range(380, 781, 5):
        assert re.fullmatch(r"#[0-9A-F]{6}", tuning.wavelength_hex(float(nm)))


def test_wavelength_hex_is_recognisably_coloured() -> None:
    def channels(nm: float) -> tuple[int, int, int]:
        h = tuning.wavelength_hex(nm)
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)

    r, g, b = channels(406.0)  # violet: blue dominant, some red, no green
    assert b > r > g == 0

    r, g, b = channels(520.0)  # green
    assert g > r
    assert g > b

    r, g, b = channels(680.0)  # red
    assert r > g
    assert r > b


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_report_agrees_with_its_parts() -> None:
    r = tuning.report(ORIGIN_HZ)
    assert r.base_hz == ORIGIN_HZ
    assert r.note == tuning.note_of(ORIGIN_HZ)
    assert r.implied_a4 == tuning.implied_reference(ORIGIN_HZ)
    assert r.carrier_hz == pytest.approx(tuning.carrier_for(ORIGIN_HZ))
    assert r.octave_shift == tuning.auto_octaves(ORIGIN_HZ)
    assert r.colour == tuning.to_visible(ORIGIN_HZ)


def test_describe_reports_the_origin_maths() -> None:
    text = tuning.describe(ORIGIN_HZ)
    assert "E2 = 82.407 Hz" in text
    assert "32.1 cents sharp" in text
    assert "448.23 Hz" in text
    assert "335.796 Hz (+2 octaves, same note)" in text
    assert "406.0 nm" in text
    assert "#8100CC" in text
    assert "out of sRGB gamut" in text


def test_describe_handles_a_frequency_with_no_colour() -> None:
    text = tuning.describe(tuning.C_LIGHT / 755e-9 / 2**42)
    assert "steps over the visible band" in text


def test_report_matches_the_prototype(reference_tuning: ModuleType) -> None:
    """Parity with reference/tuning.py for the quantities it also computes."""
    for freq in sweep(20.0, 4000.0, per_octave=11):
        ours = tuning.report(freq)
        theirs = reference_tuning.note_of(freq)
        assert ours.note.name == theirs["name"]
        assert ours.note.octave == theirs["octave"]
        assert ours.note.cents == pytest.approx(theirs["cents"])
        their_a4 = reference_tuning.implied_reference(freq)
        assert ours.implied_a4 == pytest.approx(their_a4)

        vis = reference_tuning.to_visible(freq)
        if vis["octaves"] is None:
            assert ours.colour is None
        else:
            assert ours.colour is not None
            assert ours.colour.octaves == vis["octaves"]
            assert ours.colour.nm == pytest.approx(vis["nm"])
            assert ours.colour.hex == vis["hex"]
