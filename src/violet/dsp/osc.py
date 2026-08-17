"""
Oscillators: sines and the detuned pairs that carry the beat.

Everything here is a pure function of absolute time. Given ``t`` for a span
of absolute sample indices, phase follows from the time value itself, so a
tone rendered across ten blocks is the same tone rendered in one. There is no
per-block phase to reset and therefore no click to accidentally introduce.

Time-varying frequency is the exception, and cannot be handled this way:
``sin(2*pi*f(t)*t)`` is not a tone that sweeps, it is a tone whose phase jumps
about. That needs an integrated phase accumulator carried across blocks, and
arrives with the beat automation curves (roadmap item 2).

One beat per mix
----------------
:func:`sine_pair` is the only way a tonal voice should be built. A voice at
ratio ``r`` off the root emits ``r*root - beat/2`` to the left ear and
``r*root + beat/2`` to the right, so every voice in a chord beats at exactly
the same rate and the percept stays single.

The temptation is to use a richer waveform for a fuller sound. Do not:
harmonic *n* of a pair detuned by *b* beats at *n*b*, so a sawtooth carrier
produces beats at *b*, *2b*, *3b* simultaneously and the percept smears. Add
voices instead — that is what :mod:`violet.harmony` is for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from violet._types import TAU

if TYPE_CHECKING:
    from violet._types import FloatArray, Stereo

__all__ = ["sine", "sine_pair"]


def sine(
    t: FloatArray,
    freq: float,
    level: float = 1.0,
    phase: float = 0.0,
) -> FloatArray:
    """
    A sine of ``freq`` Hz sampled at absolute times ``t``.

    ``phase`` is in radians and is a fixed offset, not a per-block state.
    """
    if freq < 0.0:
        msg = f"frequency must not be negative, got {freq!r}"
        raise ValueError(msg)
    out: FloatArray = level * np.sin(TAU * freq * t + phase)
    return out


def sine_pair(
    t: FloatArray,
    freq: float,
    beat: float,
    level: float = 1.0,
) -> Stereo:
    """
    One binaural voice: ``freq`` split into ``freq -/+ beat/2`` across the ears.

    A ``beat`` of zero is legal and meaningful — it is the null condition for
    a blinded trial, identical in every respect except that there is no
    interaural difference and therefore no beat.
    """
    if beat < 0.0:
        msg = f"beat must not be negative, got {beat!r}"
        raise ValueError(msg)
    half = beat / 2.0
    if freq - half <= 0.0:
        msg = (
            f"a {beat:g} Hz beat around {freq:g} Hz would put the left ear at "
            f"{freq - half:g} Hz; the carrier must exceed half the beat"
        )
        raise ValueError(msg)
    return sine(t, freq - half, level), sine(t, freq + half, level)
