"""
Beat rates that change over time, and the phase integral they need.

A constant tone is ``sin(2*pi*f*t)``. A tone whose frequency changes is *not*
``sin(2*pi*f(t)*t)``, and reaching for that is the single easiest way to ruin a
long render. That expression says "the phase now is the frequency now times the
whole elapsed time", which is only true if the frequency has always been that
value. Phase is the *integral* of frequency::

    phase(t) = 2*pi * integral of f from 0 to t

What the mistake actually sounds like
-------------------------------------
Not, as one might guess, a click — that only happens where the rate has a step
in it, and there it is a loud one, seventeen times the largest legitimate
sample-to-sample move. For a smooth glide the naive expression is perfectly
smooth as well, which is why it survives a listen and ships.

It is playing the wrong rate. Differentiating the naive phase gives an
instantaneous rate of ``f(t) + t*f'(t)``, so the error grows with elapsed time
and with how fast the curve is moving. Gliding from 10 Hz to 2 Hz over a minute,
at thirty seconds in you ask for 6 Hz and get 2 Hz; at forty seconds it passes
through zero and goes *negative*, which swaps which ear leads. Forty-five
minutes of that is not a descent, it is a rate nobody chose.

Why this is a closed form and not an accumulator
------------------------------------------------
The usual fix is to accumulate: ``phase += 2*pi*f/sr`` each sample, carried
across blocks. That works, and it makes the oscillator stateful, which costs
three things this package has been careful to keep. It has to be rendered
strictly in order. It accumulates rounding error over an hour. And rendering at
two block sizes no longer gives identical samples, because summing a million
numbers in chunks of a thousand is not the same arithmetic as summing them in
chunks of ten thousand.

The curves here are piecewise, and a piecewise-linear or piecewise-exponential
function has an integral in closed form. So the integral is evaluated directly
from absolute time, exactly like everything else in this package: no state, no
drift, no ordering requirement, and block-size invariance that is exact rather
than merely close.

Shapes
------
``linear`` moves at a constant number of hertz per second. ``exponential``
moves at a constant *ratio* per second, which is how pitch is heard — halving
from 8 Hz to 4 Hz sounds like the same size of step as halving from 4 to 2,
where a linear ramp would spend most of its time up at the top and then plunge.
For a descent through the EEG bands, exponential is usually the one you want.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from violet._types import FloatArray

__all__ = ["SHAPES", "BeatCurve", "Shape"]

Shape = Literal["linear", "exponential"]

#: The interpolation shapes a curve can use.
SHAPES: tuple[Shape, ...] = ("linear", "exponential")


@dataclass(frozen=True, slots=True)
class BeatCurve:
    """
    A beat rate that changes over time, as breakpoints in seconds and hertz.

    Held flat before the first breakpoint and after the last, so the curve is
    defined for all time and a render can start or overrun without the rate
    doing anything surprising at the edges.

    A curve is a value: two curves with the same points are equal, and it is
    hashable, so it can sit inside a frozen config like any other field.
    """

    points: tuple[tuple[float, float], ...]
    shape: Shape = "linear"

    _times: FloatArray = field(init=False, repr=False, compare=False)
    _freqs: FloatArray = field(init=False, repr=False, compare=False)
    _cycles: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.points:
            msg = "a beat curve needs at least one point"
            raise ValueError(msg)
        if self.shape not in SHAPES:
            msg = f"shape must be one of {SHAPES}, got {self.shape!r}"
            raise ValueError(msg)

        times = [float(at) for at, _ in self.points]
        freqs = [float(hz) for _, hz in self.points]

        if times[0] < 0.0:
            msg = f"beat curve times must not be negative, got {times[0]!r}"
            raise ValueError(msg)
        if any(later <= earlier for earlier, later in pairwise(times)):
            msg = f"beat curve times must increase, got {times!r}"
            raise ValueError(msg)
        if any(hz < 0.0 for hz in freqs):
            msg = f"beat rates must not be negative, got {freqs!r}"
            raise ValueError(msg)
        if self.shape == "exponential" and any(hz <= 0.0 for hz in freqs):
            msg = (
                "an exponential curve cannot pass through zero — there is no "
                "constant ratio from 0 Hz to anything. Use shape='linear' to "
                "sweep down to silence."
            )
            raise ValueError(msg)

        object.__setattr__(self, "_times", np.asarray(times, dtype=np.float64))
        object.__setattr__(self, "_freqs", np.asarray(freqs, dtype=np.float64))
        object.__setattr__(self, "_cycles", self._breakpoint_cycles())

    # -- construction -------------------------------------------------------

    @classmethod
    def constant(cls, hz: float) -> BeatCurve:
        """A curve that never moves. Mostly useful in tests."""
        return cls(points=((0.0, hz),))

    @classmethod
    def from_minutes(
        cls,
        at_minutes: tuple[float, ...],
        hz: tuple[float, ...],
        shape: Shape = "linear",
    ) -> BeatCurve:
        """
        Build from breakpoints in *minutes*, which is how presets read.

        A forty-five minute descent written in seconds is a wall of four-digit
        numbers nobody can check by eye.
        """
        if len(at_minutes) != len(hz):
            msg = (
                f"a curve needs one rate per time: got {len(at_minutes)} times "
                f"and {len(hz)} rates"
            )
            raise ValueError(msg)
        points = tuple(
            (at * 60.0, rate) for at, rate in zip(at_minutes, hz, strict=True)
        )
        return cls(points=points, shape=shape)

    # -- description --------------------------------------------------------

    @property
    def start_hz(self) -> float:
        """The rate before the first breakpoint."""
        return float(self._freqs[0])

    @property
    def end_hz(self) -> float:
        """The rate after the last."""
        return float(self._freqs[-1])

    @property
    def moves(self) -> bool:
        """Whether the rate ever changes. A one-point curve does not."""
        return bool(np.any(self._freqs != self._freqs[0]))

    @property
    def span(self) -> tuple[float, float]:
        """Lowest and highest rate the curve reaches."""
        return float(self._freqs.min()), float(self._freqs.max())

    def describe(self) -> str:
        """One line, for the CLI."""
        if not self.moves:
            return f"{self.start_hz:g} Hz, constant"
        legs = ", ".join(
            f"{at / 60.0:g} min {hz:g} Hz"
            for at, hz in zip(self._times, self._freqs, strict=True)
        )
        return f"{self.shape}: {legs}"

    # -- the maths ----------------------------------------------------------

    def _segment_cycles(self, index: int, u: FloatArray | float) -> FloatArray | float:
        """
        Cycles completed a fraction ``u`` of the way through one segment.

        Linear: the trapezoid, which is exact for a straight line rather than an
        approximation of one. Exponential: the integral of ``f0 * r**u``, which
        is ``f0 * (r**u - 1) / ln r`` — and degenerates to ``f0 * u`` when the
        two ends are equal, where the ratio is 1 and the logarithm is 0.
        """
        span = float(self._times[index + 1] - self._times[index])
        low = float(self._freqs[index])
        high = float(self._freqs[index + 1])

        if self.shape == "linear":
            return span * u * (2.0 * low + (high - low) * u) / 2.0
        if low == high:
            return span * low * u
        ratio = high / low
        return span * low * (ratio**u - 1.0) / math.log(ratio)

    def _breakpoint_cycles(self) -> FloatArray:
        """Cumulative cycles at each breakpoint, counting from t = 0."""
        # Before the first breakpoint the rate is held, so the cycles completed
        # by the time we reach it are simply rate times elapsed time.
        cycles = [float(self._freqs[0] * self._times[0])]
        for index in range(len(self._times) - 1):
            whole = float(self._segment_cycles(index, 1.0))
            cycles.append(cycles[-1] + whole)
        return np.asarray(cycles, dtype=np.float64)

    def at(self, t: FloatArray) -> FloatArray:
        """The instantaneous beat rate at absolute times ``t``."""
        index = np.searchsorted(self._times, t, side="right") - 1
        out = np.empty_like(t)

        before = index < 0
        out[before] = self._freqs[0]

        last = len(self._times) - 1
        after = index >= last
        out[after] = self._freqs[-1]

        inside = ~before & ~after
        if np.any(inside):
            here = index[inside]
            span = self._times[here + 1] - self._times[here]
            u = (t[inside] - self._times[here]) / span
            low, high = self._freqs[here], self._freqs[here + 1]
            if self.shape == "linear":
                out[inside] = low + (high - low) * u
            else:
                out[inside] = low * (high / low) ** u

        return out

    def cycles(self, t: FloatArray) -> FloatArray:
        """
        Cycles of beat completed by absolute times ``t``.

        The integral of :meth:`at`, in cycles rather than radians — multiply by
        two pi where a phase is wanted. Evaluated in closed form from ``t``
        alone, which is what makes it stateless and exactly block invariant.
        """
        index = np.searchsorted(self._times, t, side="right") - 1
        out = np.empty_like(t)

        before = index < 0
        out[before] = self._freqs[0] * t[before]

        last = len(self._times) - 1
        after = index >= last
        out[after] = self._cycles[-1] + self._freqs[-1] * (t[after] - self._times[-1])

        inside = ~before & ~after
        if np.any(inside):
            here = index[inside]
            span = self._times[here + 1] - self._times[here]
            u = (t[inside] - self._times[here]) / span
            low = self._freqs[here]
            high = self._freqs[here + 1]
            if self.shape == "linear":
                partial = span * u * (2.0 * low + (high - low) * u) / 2.0
            else:
                ratio = np.where(low > 0.0, high / np.where(low > 0.0, low, 1.0), 1.0)
                flat = ratio == 1.0
                logs = np.log(np.where(flat, np.e, ratio))
                partial = np.where(
                    flat,
                    span * low * u,
                    span * low * (ratio**u - 1.0) / logs,
                )
            out[inside] = self._cycles[here] + partial

        return out
