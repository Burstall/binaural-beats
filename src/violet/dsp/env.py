"""
Envelopes: fades, slow LFOs, and the asymmetric swells that make waves.

Every envelope here is a function of absolute time or absolute sample index.
That is what makes them safe to evaluate block by block: the value at sample
*i* depends on *i* alone, so splitting a render into different blocks cannot
change it. It sounds obvious written down, and it is the single thing the
prototype's wave envelope got wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from violet._types import TAU

if TYPE_CHECKING:
    from violet._types import FloatArray, IntArray

__all__ = [
    "SWELL_TAIL_SECONDS",
    "Breathing",
    "Swell",
    "fade_envelope",
    "fade_length",
    "plan_swells",
    "slow_lfo",
    "swell_envelope",
]

#: How long a swell's decay is followed before it is treated as exactly zero.
#:
#: This constant is a correctness fix, not a tuning choice. The prototype
#: decided which swells to include from the *block's* start time — ``t[0] -
#: 14`` — so a decaying wave was in or out depending on where the block
#: boundaries happened to fall. At a 10-second block a sample near the end of
#: the block saw waves up to 24 seconds old; the same sample in a 1-second
#: block saw only 14. The envelope differed by around 0.04 between the two,
#: four orders of magnitude past the block-invariance tolerance.
#:
#: Here the window is absolute and the tail is gated to exactly zero at the end
#: of it, so inclusion depends on the sample's own time and nothing else. With
#: the longest decay in use the residual at 120 seconds is about 1e-12.
SWELL_TAIL_SECONDS = 120.0


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


@dataclass(frozen=True, slots=True)
class Breathing:
    """
    Per-voice slow amplitude drift, so a held chord does not sit still.

    Each voice gets its own period and its own starting phase, both derived
    from its index. The periods are deliberately incommensurate — 47, 66, 85,
    104 seconds — so the voices drift in and out of alignment and the chord
    keeps changing shape without ever repeating a pattern. Give them all the
    same period and the chord pumps instead of breathing.

    The floor is well above zero: this is a swell, not a gate. A voice that
    disappears and comes back is heard as an event; one that varies between
    62% and 100% is heard as a texture.
    """

    base_period: float = 47.0
    period_step: float = 19.0
    phase_step: float = 1.9
    low: float = 0.62
    high: float = 1.0

    #: Explicit periods, one per voice, overriding the arithmetic sequence.
    #: Set by :meth:`snapped_to_loop`, which cannot express its result as a
    #: base and a step — each period is rounded independently.
    periods: tuple[float, ...] | None = None

    def period_for(self, voice: int) -> float:
        """The drift period of one voice."""
        if self.periods is not None:
            return self.periods[voice % len(self.periods)]
        return self.base_period + self.period_step * voice

    def gain(self, t: FloatArray, voice: int) -> FloatArray:
        """Envelope for one voice of a chord, at absolute times ``t``."""
        return slow_lfo(
            t,
            period=self.period_for(voice),
            low=self.low,
            high=self.high,
            phase=self.phase_step * voice,
        )

    def snapped_to_loop(self, loop_seconds: float, voices: int = 8) -> Breathing:
        """
        Periods adjusted so each divides the loop a whole number of times.

        An envelope caught mid-swing at the join is as much of a step as a tone
        caught mid-cycle, just slower and therefore heard as a lurch in the
        texture rather than a click. Making each period an exact divisor
        removes it. The periods move by a few percent, which is well inside the
        range they were picked from in the first place.

        The result is only approximately incommensurate afterwards, which is
        the point of the original choice — but with periods near 47 and 104
        seconds and a loop of minutes, the divisors stay distinct.
        """
        if loop_seconds <= 0.0:
            msg = f"loop length must be positive, got {loop_seconds!r}"
            raise ValueError(msg)

        wanted = (self.period_for(voice) for voice in range(max(1, voices)))
        snapped = tuple(
            loop_seconds / max(1, round(loop_seconds / period)) for period in wanted
        )
        return replace(self, periods=snapped)


@dataclass(frozen=True, slots=True)
class Swell:
    """
    One asymmetric swell: a fast rise to a peak, then a long exponential fall.

    The asymmetry is the whole point. A wave arrives in a second or two and
    takes eight to drain away, and an envelope that rises and falls at the same
    rate does not sound like water at all — it sounds like a siren. Sharp in,
    slow out is what makes it read as surf.

    ``decay`` is the audible length of the fall; the exponential time constant
    is ``decay / 2.4``, which puts the tail at roughly a tenth of peak by the
    time ``decay`` has elapsed.
    """

    peak_time: float
    rise: float
    decay: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.rise <= 0.0 or self.decay <= 0.0:
            msg = (
                f"swell rise and decay must be positive, got rise={self.rise!r}, "
                f"decay={self.decay!r}"
            )
            raise ValueError(msg)

    @property
    def tau(self) -> float:
        """Exponential time constant of the decay."""
        return self.decay / 2.4

    def support(self, tail: float = SWELL_TAIL_SECONDS) -> tuple[float, float]:
        """The window outside which this swell contributes exactly nothing."""
        return self.peak_time - self.rise, self.peak_time + tail

    def gain(self, t: FloatArray, tail: float = SWELL_TAIL_SECONDS) -> FloatArray:
        """The swell's contribution at absolute times ``t``."""
        since_peak = t - self.peak_time
        out = np.zeros_like(t)

        rising = (since_peak >= -self.rise) & (since_peak < 0.0)
        out[rising] = (
            np.sin((since_peak[rising] + self.rise) / self.rise * np.pi / 2.0) ** 2
        )

        falling = (since_peak >= 0.0) & (since_peak <= tail)
        out[falling] = np.exp(-since_peak[falling] / self.tau)

        return self.amplitude * out


def plan_swells(
    total_seconds: float,
    rng: np.random.Generator,
    first_at: float = 4.0,
    rise: tuple[float, float] = (1.3, 2.8),
    decay: tuple[float, float] = (5.0, 10.5),
    amplitude: tuple[float, float] = (0.55, 1.0),
    gap: tuple[float, float] = (6.5, 13.0),
) -> tuple[Swell, ...]:
    """
    Plan a sequence of swells covering ``total_seconds`` and a little past it.

    Four draws per swell, in this order: rise, decay, amplitude, then the gap
    to the next one. That ordering is part of the seed contract.

    The first swell is a few seconds in rather than at zero, so a render opens
    on quiet water instead of mid-wave.
    """
    swells: list[Swell] = []
    clock = first_at
    while clock < total_seconds + 20.0:
        swells.append(
            Swell(
                peak_time=clock,
                rise=float(rng.uniform(*rise)),
                decay=float(rng.uniform(*decay)),
                amplitude=float(rng.uniform(*amplitude)),
            )
        )
        clock += float(rng.uniform(*gap))
    return tuple(swells)


def swell_envelope(
    t: FloatArray,
    swells: tuple[Swell, ...],
    set_period: float,
    ceiling: float = 2.2,
    tail: float = SWELL_TAIL_SECONDS,
    low: float = 0.55,
    high: float = 1.0,
) -> FloatArray:
    """
    Sum of overlapping swells, modulated by a slow set cycle.

    Swells overlap and add, which is what gives an occasional big one where two
    coincide, and the sum is clipped so those coincidences cannot run away. The
    set cycle is a very slow LFO over the whole thing — real surf comes in sets,
    a couple of minutes of larger waves and then a lull, and without it a
    render sounds mechanically even over half an hour.
    """
    if not math.isfinite(set_period) or set_period <= 0.0:
        msg = f"set_period must be finite and positive, got {set_period!r}"
        raise ValueError(msg)

    first, last = float(t[0]), float(t[-1])
    total = np.zeros_like(t)
    for swell in swells:
        audible_from, audible_to = swell.support(tail)
        if audible_to < first or audible_from > last:
            continue
        total += swell.gain(t, tail)

    return np.clip(total, 0.0, ceiling) * slow_lfo(t, set_period, low=low, high=high)
