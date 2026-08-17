"""
Typed wrappers over the scipy IIR filters this package uses.

Two objects. A :class:`FilterDesign` is the coefficients — immutable, cheap to
share, safe to reuse. A :class:`FilterStream` is one running instance of that
design with its own ``zi`` state, and there must be exactly one per channel.

Why the state matters
---------------------
``lfilter`` without ``zi`` starts from silence. Called once per block that
means every block begins with the filter's step response, which is a click at
every boundary and a comb filter over the whole render. Carrying ``zi``
forward makes the blocks invisible: the filter sees one continuous signal and
does not know or care where the block edges were.

One stream per channel, never one shared. Two channels through one stream
would interleave their histories, which is both wrong and — worse — quietly
correlates the two ears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from violet._types import FloatArray

__all__ = ["FilterDesign", "FilterStream"]

#: A first-order denominator is ``[1, a1]`` — two coefficients.
_FIRST_ORDER = 2


@dataclass(frozen=True, slots=True)
class FilterDesign:
    """Numerator and denominator coefficients of an IIR filter."""

    b: tuple[float, ...]
    a: tuple[float, ...]
    label: str = "iir"

    def __post_init__(self) -> None:
        if not self.b or len(self.a) < _FIRST_ORDER:
            msg = f"{self.label}: need at least a first-order filter"
            raise ValueError(msg)
        if self.a[0] == 0.0:
            msg = f"{self.label}: a[0] must not be zero"
            raise ValueError(msg)

    @property
    def order(self) -> int:
        """Filter order, which is also the length of the state vector."""
        return max(len(self.a), len(self.b)) - 1

    def stream(self) -> FilterStream:
        """A running instance of this design, with its own state."""
        return FilterStream(self)

    @classmethod
    def from_arrays(
        cls,
        b: FloatArray,
        a: FloatArray,
        label: str,
    ) -> FilterDesign:
        """Wrap the pair of arrays a scipy design function returns."""
        return cls(
            b=tuple(float(v) for v in b),
            a=tuple(float(v) for v in a),
            label=label,
        )

    @classmethod
    def butterworth(
        cls,
        sample_rate: int,
        cutoff: float | tuple[float, float],
        btype: str = "low",
        order: int = 2,
    ) -> FilterDesign:
        """
        Butterworth low-pass, high-pass or band-pass.

        Cutoffs are in Hz and normalised against Nyquist here rather than
        passed as ``fs``, which is the form the prototypes used and keeps the
        coefficients bit-identical to theirs.
        """
        nyquist = sample_rate / 2.0
        edges = cutoff if isinstance(cutoff, tuple) else (cutoff,)
        for edge in edges:
            if not 0.0 < edge < nyquist:
                msg = (
                    f"butterworth {btype}: cutoff {edge:g} Hz must sit between 0 "
                    f"and Nyquist ({nyquist:g} Hz)"
                )
                raise ValueError(msg)

        normalised: float | list[float] = (
            [edge / nyquist for edge in edges]
            if isinstance(cutoff, tuple)
            else edges[0] / nyquist
        )
        b, a = signal.butter(order, normalised, btype)
        return cls.from_arrays(b, a, f"butterworth-{btype}")

    @classmethod
    def notch(cls, sample_rate: int, freq: float, q: float = 2.0) -> FilterDesign:
        """
        A narrow dip at ``freq``.

        Used to carve the carrier out of a broadband noise bed so the noise
        cannot mask the beat. ``q`` sets how narrow: 2 is wide enough to catch
        a carrier that drifts and narrow enough to be inaudible as a change in
        the noise's colour.
        """
        if not 0.0 < freq < sample_rate / 2.0:
            msg = (
                f"notch: {freq:g} Hz must sit between 0 and Nyquist "
                f"({sample_rate / 2.0:g} Hz)"
            )
            raise ValueError(msg)
        if q <= 0.0:
            msg = f"notch: q must be positive, got {q!r}"
            raise ValueError(msg)
        b, a = signal.iirnotch(freq, q, sample_rate)
        return cls.from_arrays(b, a, "notch")


class FilterStream:
    """One running instance of a :class:`FilterDesign`, for one channel."""

    __slots__ = ("_a", "_b", "_zi", "design")

    def __init__(self, design: FilterDesign) -> None:
        self.design = design
        self._b = np.asarray(design.b, dtype=np.float64)
        self._a = np.asarray(design.a, dtype=np.float64)
        self._zi = np.zeros(design.order, dtype=np.float64)

    def reset(self) -> None:
        """Forget the history. Only correct at the start of a render."""
        self._zi = np.zeros(self.design.order, dtype=np.float64)

    def process(self, x: FloatArray) -> FloatArray:
        """Filter one block, carrying the state forward to the next."""
        y, self._zi = signal.lfilter(self._b, self._a, x, zi=self._zi)
        out: FloatArray = np.asarray(y, dtype=np.float64)
        return out
