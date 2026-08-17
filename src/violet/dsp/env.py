"""
Envelopes: fades and slow LFOs.

Every envelope here is a function of absolute time or absolute sample index.
That is what makes them safe to evaluate block by block: the value at sample
*i* depends on *i* alone, so splitting a render into different blocks cannot
change it.

The swell envelopes that give the ocean its shape arrive in stage 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from violet._types import TAU

if TYPE_CHECKING:
    from violet._types import FloatArray, IntArray

__all__ = ["fade_envelope", "fade_length", "slow_lfo"]


def fade_length(
    total: int,
    sample_rate: int,
    seconds: float,
    max_denominator: int = 4,
) -> int:
    """
    Fade length in samples, capped at ``total // max_denominator``.

    The cap matters for short renders: an 8-second fade either end of a
    10-second file is not a fade, it is the whole file. Capping at a quarter
    of the length keeps a recognisable body in the middle, and keeps the head
    and tail masks from overlapping.
    """
    if seconds <= 0.0 or total <= 0:
        return 0
    if max_denominator < 2:  # noqa: PLR2004 - head and tail need half each
        msg = f"max_denominator must be at least 2, got {max_denominator!r}"
        raise ValueError(msg)
    return min(int(sample_rate * seconds), total // max_denominator)


def fade_envelope(indices: IntArray, total: int, fade: int) -> FloatArray:
    """
    Raised-sine fade in and out, evaluated at absolute sample ``indices``.

    ``sin^2`` rather than a straight line: a linear fade on a sustained tone
    has an audible corner at each end where the slope changes discontinuously.
    The envelope reaches exactly zero at sample 0 and at sample ``total - 1``,
    so a rendered file starts and ends in silence rather than at a step.
    """
    env = np.ones(len(indices), dtype=np.float64)
    if fade <= 0:
        return env
    head = indices < fade
    tail = indices >= total - fade
    env[head] = np.sin(indices[head] / fade * np.pi / 2.0) ** 2
    env[tail] = np.sin((total - 1 - indices[tail]) / fade * np.pi / 2.0) ** 2
    return env


def slow_lfo(
    t: FloatArray,
    period: float,
    low: float = 0.55,
    high: float = 1.0,
    phase: float = 0.0,
) -> FloatArray:
    """
    A sine oscillating between ``low`` and ``high`` once every ``period``.

    Used to make layers breathe. Periods are deliberately long and mutually
    prime-ish — 47 s against 70 s against 110 s — so the layers drift in and
    out of alignment instead of pulsing together, and the texture never
    settles into a pattern the ear can predict.
    """
    if period <= 0.0:
        msg = f"period must be positive, got {period!r}"
        raise ValueError(msg)
    swing = 0.5 + 0.5 * np.sin(TAU * t / period + phase)
    out: FloatArray = low + (high - low) * swing
    return out
