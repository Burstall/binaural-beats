"""
Oscillators: sines and the detuned pairs that carry the beat.

Everything here is a pure function of absolute time. Given ``t`` for a span
of absolute sample indices, phase follows from the time value itself, so a
tone rendered across ten blocks is the same tone rendered in one. There is no
per-block phase to reset and therefore no click to accidentally introduce.

Time-varying frequency is the exception, and cannot be handled the same way.
``sin(2*pi*f(t)*t)`` does not sweep at ``f(t)``; it sweeps at
``f(t) + t*f'(t)``, which is a different rate and sounds entirely plausible
while being wrong. Phase is the *integral* of frequency, and :func:`swept_pair`
gets it from :class:`violet.dsp.curves.BeatCurve`, which evaluates that integral
in closed form — so a sweeping tone stays as stateless and as exactly block
invariant as a fixed one.

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
    from violet.dsp.curves import BeatCurve

__all__ = ["sine", "sine_pair", "swept_pair"]


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


def swept_pair(
    t: FloatArray,
    freq: float,
    curve: BeatCurve,
    level: float = 1.0,
) -> Stereo:
    """
    One binaural voice whose beat rate follows ``curve``.

    The carrier stays put and the two ears move symmetrically around it, so the
    pitch you hear does not drift while the pulse slows. Each ear's phase is the
    carrier's phase plus or minus half the beat's *integrated* phase::

        left = sin(2 * pi * (freq * t - cycles(t) / 2))
        right = sin(2 * pi * (freq * t + cycles(t) / 2))

    Not ``freq -/+ curve.at(t)/2`` inside the usual expression. That is the
    mistake this function exists to prevent, and it is a quiet one: for a smooth
    curve it produces a perfectly smooth signal at ``f(t) + t*f'(t)``, which is
    not the rate you asked for and eventually not even the right sign. See
    :mod:`violet.dsp.curves`.
    """
    low, high = curve.span
    if freq - high / 2.0 <= 0.0:
        msg = (
            f"a beat reaching {high:g} Hz around {freq:g} Hz would put the left "
            f"ear at {freq - high / 2.0:g} Hz; the carrier must exceed half the "
            f"fastest rate in the curve"
        )
        raise ValueError(msg)
    del low

    carrier_phase = TAU * freq * t
    half_beat_phase = TAU * curve.cycles(t) / 2.0
    left: FloatArray = level * np.sin(carrier_phase - half_beat_phase)
    right: FloatArray = level * np.sin(carrier_phase + half_beat_phase)
    return left, right
