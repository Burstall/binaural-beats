#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Tuning and colour mapping for an arbitrary base frequency.

Stdlib only, so it imports cleanly from the render scripts and also runs
standalone:

    uv run reference/tuning.py 83.949
    uv run reference/tuning.py 136.10 --a4 432
"""

from __future__ import annotations

import argparse
from math import floor, log2

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
C_LIGHT = 299_792_458.0
VISIBLE_NM = (380.0, 750.0)

# Comfortable register for a binaural carrier. Below ~200 Hz the percept
# weakens and headphones roll off; above ~600 Hz sustained tones get
# fatiguing over a long session.
CARRIER_LO = 250.0
CARRIER_HI = 520.0


def note_of(freq: float, a4: float = 440.0) -> dict:
    """Nearest equal-tempered note, and how far off it the frequency sits."""
    midi_exact = 69.0 + 12.0 * log2(freq / a4)
    midi = round(midi_exact)
    exact = a4 * 2.0 ** ((midi - 69) / 12.0)
    return {
        "name": NOTES[midi % 12],
        "octave": midi // 12 - 1,
        "exact_hz": exact,
        "cents": 1200.0 * log2(freq / exact),
        "midi": midi,
    }


def implied_reference(freq: float, a4: float = 440.0) -> float:
    """The A4 reference that would make this frequency exactly in tune."""
    n = note_of(freq, a4)
    return a4 * freq / n["exact_hz"]


def auto_octaves(freq: float, lo: float = CARRIER_LO,
                 hi: float = CARRIER_HI) -> int:
    """Whole octave shift landing `freq` in the usable carrier window."""
    n = 0
    f = freq
    while f < lo and n < 12:
        f *= 2.0
        n += 1
    while f > hi and n > -12:
        f /= 2.0
        n -= 1
    return n


def to_visible(freq: float) -> dict:
    """Octave-shift upward until the frequency lands in the visible band."""
    n = 0
    f = freq
    while n < 80:
        nm = C_LIGHT / f * 1e9
        if VISIBLE_NM[0] <= nm <= VISIBLE_NM[1]:
            return {"octaves": n, "thz": f / 1e12, "nm": nm,
                    "hex": wavelength_hex(nm)}
        if nm < VISIBLE_NM[0]:
            break
        f *= 2.0
        n += 1
    return {"octaves": None, "thz": f / 1e12, "nm": C_LIGHT / f * 1e9,
            "hex": None}


def wavelength_hex(nm: float) -> str:
    """
    Approximate sRGB for a spectral wavelength (Bruton's piecewise fit).

    Spectral colours are outside the sRGB gamut - no screen can display a
    true 406 nm violet. This returns the nearest in-gamut approximation,
    which is a compromise rather than the real thing.
    """
    if nm < 380 or nm > 780:
        return "#000000"
    if nm < 440:
        r, g, b = -(nm - 440) / 60.0, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50.0, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20.0
    elif nm < 580:
        r, g, b = (nm - 510) / 70.0, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    if nm < 420:
        f = 0.3 + 0.7 * (nm - 380) / 40.0
    elif nm > 700:
        f = 0.3 + 0.7 * (780 - nm) / 80.0
    else:
        f = 1.0

    vals = [int(round(255 * (c * f) ** 0.8)) for c in (r, g, b)]
    return "#" + "".join(f"{v:02X}" for v in vals)


def describe(freq: float, a4: float = 440.0) -> str:
    n = note_of(freq, a4)
    ref = implied_reference(freq, a4)
    oct_shift = auto_octaves(freq)
    carrier = freq * 2.0 ** oct_shift
    vis = to_visible(freq)
    sign = "sharp" if n["cents"] > 0 else "flat"

    lines = [
        f"base frequency      {freq:.4f} Hz",
        f"nearest note        {n['name']}{n['octave']} "
        f"= {n['exact_hz']:.3f} Hz (A4 = {a4:g})",
        f"deviation           {abs(n['cents']):.1f} cents {sign}",
        f"implied A4 ref      {ref:.2f} Hz",
        f"suggested carrier   {carrier:.3f} Hz "
        f"({oct_shift:+d} octaves, same note)",
    ]
    if vis["octaves"] is not None:
        lines += [
            f"colour octave       +{vis['octaves']} octaves "
            f"= {vis['thz']:.2f} THz",
            f"wavelength          {vis['nm']:.1f} nm",
            f"approx screen hex   {vis['hex']}  "
            f"(out of sRGB gamut - approximation only)",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frequency", type=float)
    p.add_argument("--a4", type=float, default=440.0,
                   help="tuning reference for note naming (default 440)")
    a = p.parse_args()
    print(describe(a.frequency, a.a4))
