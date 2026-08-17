"""
Frequency arithmetic: notes, cents, tuning references, carriers, colour.

Pure functions over floats and exact :class:`~fractions.Fraction` ratios. No
audio, no numpy, no state. Everything the rest of the package needs to turn a
base frequency into musical and visual quantities lives here:

* the nearest equal-tempered note, and the deviation from it in cents
* the tuning reference implied by treating a frequency as exactly in tune
* :func:`auto_octaves` — the whole-octave shift that lands a base frequency
  in the usable binaural carrier window, preserving the note
* just-intonation ratio tables, so chords can be spelled as exact ratios
  rather than equal-tempered approximations
* the octave-arithmetic mapping from frequency to wavelength and an
  approximate sRGB hex

No base frequency appears anywhere in this module, or anywhere else in the
package. It is configuration, and it enters from there.

Why just intonation
-------------------
An equal-tempered major third is 400 cents; the just 5/4 is 386.31. On a
sustained drone that 13.7-cent error is audible as roughness, because the
partials of the two tones beat against each other a few times a second.
Exact small-integer ratios lock instead: their partials coincide rather than
collide. Over a 45-minute render the difference is the whole character of the
sound, so ratios are held as exact fractions and only converted to float at
the point where a frequency is needed.

The colour mapping
------------------
Doubling an audio frequency enough times lands it in the band we see. That is
octave arithmetic and nothing more — there is no physical process connecting a
sound at 84 Hz to light at 406 nm, and none is claimed. It is a naming
convention that happens to be pleasing.

Spectral colours are also outside the sRGB gamut, so no screen can display the
result. :func:`wavelength_hex` returns the nearest in-gamut approximation, and
that limitation is more interesting than pretending otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor, isfinite, log2

__all__ = [
    "A4_STANDARD",
    "CARRIER_HI",
    "CARRIER_LO",
    "C_LIGHT",
    "JUST_RATIOS",
    "JUST_SEMITONES",
    "NOTES",
    "VISIBLE_NM",
    "Colour",
    "Note",
    "TuningReport",
    "auto_octaves",
    "carrier_for",
    "cents_between",
    "describe",
    "implied_reference",
    "just_ratio",
    "midi_to_frequency",
    "note_of",
    "note_to_frequency",
    "ratio_cents",
    "report",
    "to_visible",
    "transpose_cents",
    "wavelength_hex",
]

NOTES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

#: Speed of light in m/s — exact, by the definition of the metre.
C_LIGHT = 299_792_458.0

#: The band we see, in nanometres. Its span is a ratio of 1.974, slightly
#: less than an octave, so a frequency's octave series can step over it.
VISIBLE_NM = (380.0, 750.0)

#: Comfortable register for a binaural carrier. The percept is formed from
#: interaural phase differences and weakens sharply below roughly 200 Hz,
#: where headphones also roll off; above roughly 600 Hz a sustained tone
#: becomes fatiguing over a long session.
CARRIER_LO = 250.0
CARRIER_HI = 520.0

#: The conventional tuning reference. Used for *naming* only — it is the
#: yardstick a frequency is measured against, never something imposed on it.
A4_STANDARD = 440.0

#: Five-limit just intonation, one entry per semitone of the octave. The
#: tritone has no good five-limit spelling; 45/32 is the conventional choice.
JUST_SEMITONES: tuple[Fraction, ...] = (
    Fraction(1, 1),  # 0   unison            0.00 cents
    Fraction(16, 15),  # 1   minor second    111.73
    Fraction(9, 8),  # 2   major second      203.91
    Fraction(6, 5),  # 3   minor third       315.64
    Fraction(5, 4),  # 4   major third       386.31
    Fraction(4, 3),  # 5   perfect fourth    498.04
    Fraction(45, 32),  # 6  tritone          590.22
    Fraction(3, 2),  # 7   perfect fifth     701.96
    Fraction(8, 5),  # 8   minor sixth       813.69
    Fraction(5, 3),  # 9   major sixth       884.36
    Fraction(9, 5),  # 10  minor seventh     1017.60
    Fraction(15, 8),  # 11 major seventh     1088.27
    Fraction(2, 1),  # 12  octave            1200.00
)

#: The same table by interval name.
JUST_RATIOS: dict[str, Fraction] = {
    "unison": JUST_SEMITONES[0],
    "minor_second": JUST_SEMITONES[1],
    "major_second": JUST_SEMITONES[2],
    "minor_third": JUST_SEMITONES[3],
    "major_third": JUST_SEMITONES[4],
    "perfect_fourth": JUST_SEMITONES[5],
    "tritone": JUST_SEMITONES[6],
    "perfect_fifth": JUST_SEMITONES[7],
    "minor_sixth": JUST_SEMITONES[8],
    "major_sixth": JUST_SEMITONES[9],
    "minor_seventh": JUST_SEMITONES[10],
    "major_seventh": JUST_SEMITONES[11],
    "octave": JUST_SEMITONES[12],
}

# Guard for whole-octave arithmetic. One picocent, far below anything
# audible or measurable, but enough that log2 rounding cannot push a
# frequency sitting exactly on a window edge into the wrong octave.
_OCTAVE_EPS = 1e-12

_NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Note:
    """The equal-tempered note nearest a frequency, and the error against it."""

    name: str
    octave: int
    exact_hz: float
    cents: float
    midi: int

    @property
    def label(self) -> str:
        """Scientific pitch notation, e.g. ``"E2"``."""
        return f"{self.name}{self.octave}"

    @property
    def direction(self) -> str:
        """``"sharp"``, ``"flat"``, or ``"exact"`` — where the frequency sits."""
        if abs(self.cents) < 0.005:  # noqa: PLR2004 - half a hundredth of a cent
            return "exact"
        return "sharp" if self.cents > 0 else "flat"


@dataclass(frozen=True, slots=True)
class Colour:
    """A frequency octave-shifted into the visible band."""

    octaves: int
    thz: float
    nm: float
    hex: str


@dataclass(frozen=True, slots=True)
class TuningReport:
    """Everything derivable from a base frequency, in one value."""

    base_hz: float
    a4: float
    note: Note
    implied_a4: float
    octave_shift: int
    carrier_hz: float
    colour: Colour | None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _check_frequency(freq: float, label: str = "frequency") -> None:
    if not isfinite(freq) or freq <= 0.0:
        msg = f"{label} must be finite and positive, got {freq!r}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# notes and cents
# ---------------------------------------------------------------------------


def note_of(freq: float, a4: float = A4_STANDARD) -> Note:
    """
    Nearest equal-tempered note to ``freq``, and how far off it sits.

    ``a4`` is the reference the note is *named* against; changing it changes
    the name and the cents, never the frequency.
    """
    _check_frequency(freq)
    _check_frequency(a4, "a4")
    midi_exact = 69.0 + 12.0 * log2(freq / a4)
    midi = round(midi_exact)
    exact = midi_to_frequency(midi, a4)
    return Note(
        name=NOTES[midi % 12],
        octave=midi // 12 - 1,
        exact_hz=exact,
        cents=1200.0 * log2(freq / exact),
        midi=midi,
    )


def midi_to_frequency(midi: int, a4: float = A4_STANDARD) -> float:
    """Frequency of an equal-tempered MIDI note number."""
    _check_frequency(a4, "a4")
    return float(a4 * 2.0 ** ((midi - 69) / 12.0))


def note_to_frequency(name: str, a4: float = A4_STANDARD) -> float:
    """
    Frequency of a note in scientific pitch notation, e.g. ``"E2"``, ``"Bb3"``.

    Sharps and flats are both accepted; the octave number may be negative.
    """
    match = _NOTE_PATTERN.match(name.strip())
    if match is None:
        msg = f"cannot parse {name!r} as a note; expected something like 'E2'"
        raise ValueError(msg)
    letter, accidental, octave = match.groups()
    semitone = _LETTER_SEMITONE[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = semitone + 12 * (int(octave) + 1)
    return midi_to_frequency(midi, a4)


def cents_between(f_from: float, f_to: float) -> float:
    """Interval from ``f_from`` up to ``f_to``, in cents. Negative if down."""
    _check_frequency(f_from, "f_from")
    _check_frequency(f_to, "f_to")
    return 1200.0 * log2(f_to / f_from)


def transpose_cents(freq: float, cents: float) -> float:
    """``freq`` shifted by ``cents``. The inverse of :func:`cents_between`."""
    _check_frequency(freq)
    return float(freq * 2.0 ** (cents / 1200.0))


def implied_reference(freq: float, a4: float = A4_STANDARD) -> float:
    """
    The A4 reference that would make ``freq`` exactly in tune.

    A frequency reading 32 cents sharp of E2 at A440 is exactly E2 at some
    other reference; this is that reference. ``a4`` only selects which note
    the frequency is snapped to, which for anything within a quartertone of a
    note is the obvious one.
    """
    return a4 * freq / note_of(freq, a4).exact_hz


def ratio_cents(ratio: Fraction | float) -> float:
    """Size of a frequency ratio in cents. ``3/2`` is 701.955, not 700."""
    value = float(ratio)
    if not isfinite(value) or value <= 0.0:
        msg = f"ratio must be finite and positive, got {ratio!r}"
        raise ValueError(msg)
    return 1200.0 * log2(value)


def just_ratio(interval: str) -> Fraction:
    """Exact five-limit ratio for a named interval, e.g. ``"perfect_fifth"``."""
    try:
        return JUST_RATIOS[interval]
    except KeyError:
        known = ", ".join(sorted(JUST_RATIOS))
        msg = f"unknown interval {interval!r}; known intervals are: {known}"
        raise KeyError(msg) from None


# ---------------------------------------------------------------------------
# carrier selection
# ---------------------------------------------------------------------------


def auto_octaves(
    freq: float,
    lo: float = CARRIER_LO,
    hi: float = CARRIER_HI,
) -> int:
    """
    Whole-octave shift landing ``freq`` in the carrier window ``[lo, hi]``.

    Whole octaves, so the note is preserved: a carrier chosen this way is the
    same pitch class as the base frequency, an octave or three up. Of the
    shifts that land in the window, the one of smallest magnitude wins, so a
    frequency already in range is left alone.

    Raises ``ValueError`` if no whole-octave shift lands inside, which can
    only happen for a window narrower than an octave.
    """
    _check_frequency(freq)
    if not 0.0 < lo < hi:
        msg = f"carrier window must satisfy 0 < lo < hi, got lo={lo!r}, hi={hi!r}"
        raise ValueError(msg)

    lowest = ceil(log2(lo / freq) - _OCTAVE_EPS)
    highest = floor(log2(hi / freq) + _OCTAVE_EPS)
    if lowest > highest:
        msg = (
            f"no whole-octave shift of {freq:g} Hz lands in {lo:g}-{hi:g} Hz; "
            f"a window spanning less than an octave "
            f"({hi / lo:.3f}x) cannot catch every frequency"
        )
        raise ValueError(msg)
    return min(max(0, lowest), highest)


def carrier_for(
    freq: float,
    lo: float = CARRIER_LO,
    hi: float = CARRIER_HI,
) -> float:
    """``freq`` shifted by :func:`auto_octaves` — the binaural carrier."""
    return freq * 2.0 ** auto_octaves(freq, lo, hi)


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------


def to_visible(freq: float) -> Colour | None:
    """
    Octave-shift ``freq`` upward until it lands in the visible band.

    Returns ``None`` when no octave lands inside. The visible band spans a
    ratio of 1.974, a shade less than an octave, so roughly one frequency in
    eighty has an octave series that steps over it. That is a real outcome,
    not an error.
    """
    _check_frequency(freq)
    lo_nm, hi_nm = VISIBLE_NM
    octaves = 0
    f = freq
    while True:
        nm = C_LIGHT / f * 1e9
        if nm < lo_nm:
            return None
        if nm <= hi_nm:
            return Colour(
                octaves=octaves,
                thz=f / 1e12,
                nm=nm,
                hex=wavelength_hex(nm),
            )
        f *= 2.0
        octaves += 1


def wavelength_hex(nm: float) -> str:
    """
    Approximate sRGB for a spectral wavelength, by Bruton's piecewise fit.

    Spectral colours lie outside the sRGB gamut, so this is the nearest
    in-gamut approximation rather than the colour itself. Wavelengths outside
    380-780 nm return black.
    """
    if nm < 380.0 or nm > 780.0:  # noqa: PLR2004 - the visible band, by definition
        return "#000000"

    if nm < 440.0:  # noqa: PLR2004
        r, g, b = -(nm - 440.0) / 60.0, 0.0, 1.0
    elif nm < 490.0:  # noqa: PLR2004
        r, g, b = 0.0, (nm - 440.0) / 50.0, 1.0
    elif nm < 510.0:  # noqa: PLR2004
        r, g, b = 0.0, 1.0, -(nm - 510.0) / 20.0
    elif nm < 580.0:  # noqa: PLR2004
        r, g, b = (nm - 510.0) / 70.0, 1.0, 0.0
    elif nm < 645.0:  # noqa: PLR2004
        r, g, b = 1.0, -(nm - 645.0) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    # The eye's response falls away at both ends of the band.
    if nm < 420.0:  # noqa: PLR2004
        fade = 0.3 + 0.7 * (nm - 380.0) / 40.0
    elif nm > 700.0:  # noqa: PLR2004
        fade = 0.3 + 0.7 * (780.0 - nm) / 80.0
    else:
        fade = 1.0

    # 0.8 is a gamma approximation, matching the reference implementation.
    channels = (round(255 * (c * fade) ** 0.8) for c in (r, g, b))
    return "#" + "".join(f"{v:02X}" for v in channels)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(freq: float, a4: float = A4_STANDARD) -> TuningReport:
    """Everything this module can say about a base frequency."""
    shift = auto_octaves(freq)
    return TuningReport(
        base_hz=freq,
        a4=a4,
        note=note_of(freq, a4),
        implied_a4=implied_reference(freq, a4),
        octave_shift=shift,
        carrier_hz=freq * 2.0**shift,
        colour=to_visible(freq),
    )


def describe(freq: float, a4: float = A4_STANDARD) -> str:
    """Human-readable summary of :func:`report`, for the CLI."""
    r = report(freq, a4)
    n = r.note
    lines = [
        f"base frequency      {r.base_hz:.4f} Hz",
        f"nearest note        {n.label} = {n.exact_hz:.3f} Hz (A4 = {r.a4:g})",
        f"deviation           {abs(n.cents):.1f} cents {n.direction}",
        f"implied A4 ref      {r.implied_a4:.2f} Hz",
        f"suggested carrier   {r.carrier_hz:.3f} Hz "
        f"({r.octave_shift:+d} octaves, same note)",
    ]
    if r.colour is None:
        lines.append(
            "colour octave       none - the octave series steps over the visible band"
        )
    else:
        c = r.colour
        lines += [
            f"colour octave       +{c.octaves} octaves = {c.thz:.2f} THz",
            f"wavelength          {c.nm:.1f} nm",
            f"approx screen hex   {c.hex}  (out of sRGB gamut - approximation only)",
        ]
    return "\n".join(lines)
