"""
Noise generators with persistent state.

White noise filtered to pink, with both the generator's state and the filter's
state carried forward block to block, so the output is one continuous stream
rather than a sequence of independently started ones.

One generator per stream
------------------------
Each stream owns its own :class:`numpy.random.Generator`, seeded from a shared
seed sequence. The prototype drew both ocean channels from a single generator —
``n`` samples for the left, then ``n`` for the right, block after block —
which quietly makes the noise depend on the block size. Render the same
configuration in 1-second blocks instead of 10 and the two channels' draws
interleave differently, and you get different noise. The statistics are the
same either way, so nothing sounds wrong; it is simply not the same file, and
a rendering engine that cannot reproduce its own output is not much use.

Spawning from a seed sequence also guarantees the two channels are
independent, which matters more than it looks. Decorrelated noise cannot carry
a binaural beat at all — the percept needs correlated input at both ears — so
the ocean is texture only and the beat lives entirely in the harmony. Two
streams accidentally drawn from overlapping state would blur that line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from violet.dsp.filters import FilterDesign

if TYPE_CHECKING:
    from violet._types import FloatArray

__all__ = [
    "KELLET",
    "PINKING",
    "KelletPink",
    "OnePole",
    "PinkNoise",
    "WhiteNoise",
    "spawn_seeds",
]

#: Paul Kellet's economy pink filter: three one-poles in parallel plus a
#: little of the white straight through. Coarser than :data:`PINKING` and
#: warmer for it, with more weight at the bottom. Kept because it is what the
#: layered prototype's air bed used, and it does not sound the same.
KELLET: tuple[tuple[float, float], ...] = (
    (0.99765, 0.0990460),
    (0.96300, 0.2965164),
    (0.57000, 1.0526913),
)
KELLET_DIRECT = 0.1848

#: A three-pole, three-zero fit to -3 dB/octave, accurate to about 0.3 dB from
#: 10 Hz to 20 kHz. Cheaper than a proper octave-band cascade and, unlike the
#: economy running-sum filters, expressible as a single ``lfilter`` call — which
#: is what lets the state carry forward as one small vector.
PINKING = FilterDesign(
    b=(0.049922, -0.095993, 0.050612, -0.004408),
    a=(1.0, -2.494956, 2.017265, -0.522189),
    label="pinking",
)


def spawn_seeds(seed: int, count: int) -> list[np.random.SeedSequence]:
    """
    ``count`` independent seed sequences derived from one seed.

    Independent in the strong sense: spawned sequences are designed not to
    overlap, unlike seeds picked by hand as ``seed``, ``seed + 1``.
    """
    if count < 1:
        msg = f"count must be at least 1, got {count!r}"
        raise ValueError(msg)
    return np.random.SeedSequence(seed).spawn(count)


class WhiteNoise:
    """Gaussian white noise from one generator, resettable."""

    __slots__ = ("_rng", "_seed", "scale")

    def __init__(
        self,
        seed: int | np.random.SeedSequence,
        scale: float = 1.0,
    ) -> None:
        self._seed = seed
        self.scale = scale
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Restart the stream from its seed."""
        self._rng = np.random.default_rng(self._seed)

    def block(self, n: int) -> FloatArray:
        """The next ``n`` samples."""
        out: FloatArray = self._rng.standard_normal(n) * self.scale
        return out


class PinkNoise:
    """
    White noise through the pinking filter: -3 dB per octave.

    Pink rather than white because white noise is brutal to sit inside for
    forty minutes — its energy per octave rises, so it reads as hiss. Pink has
    equal energy per octave, which is the distribution most natural broadband
    sound has, surf included.
    """

    __slots__ = ("_filter", "_white")

    def __init__(
        self,
        seed: int | np.random.SeedSequence,
        scale: float = 1.0,
        design: FilterDesign = PINKING,
    ) -> None:
        self._white = WhiteNoise(seed, scale)
        self._filter = design.stream()

    @property
    def scale(self) -> float:
        """Amplitude of the white noise going in."""
        return self._white.scale

    def reset(self) -> None:
        """Restart the generator and clear the filter state."""
        self._white.reset()
        self._filter.reset()

    def block(self, n: int) -> FloatArray:
        """The next ``n`` samples."""
        return self._filter.process(self._white.block(n))


class KelletPink:
    """
    The economy pink filter, as parallel one-poles rather than a sample loop.

    The prototype ran this as a Python loop over every sample, which is the
    same arithmetic done sixty thousand times a second in the interpreter.
    Three poles in parallel is exactly the same filter — each line of that loop
    *is* a one-pole recursion — and it runs as three vectorised calls with
    their state carried forward like any other filter here.
    """

    __slots__ = ("_direct", "_poles", "_white", "scale")

    def __init__(
        self,
        seed: int | np.random.SeedSequence,
        scale: float = 1.0,
        poles: tuple[tuple[float, float], ...] = KELLET,
        direct: float = KELLET_DIRECT,
    ) -> None:
        self._white = WhiteNoise(seed)
        self._direct = direct
        self._poles = [
            FilterDesign(b=(gain,), a=(1.0, -pole), label="kellet").stream()
            for pole, gain in poles
        ]
        self.scale = scale

    def reset(self) -> None:
        """Restart the generator and clear every pole."""
        self._white.reset()
        for pole in self._poles:
            pole.reset()

    def block(self, n: int) -> FloatArray:
        """The next ``n`` samples."""
        white = self._white.block(n)
        out = white * self._direct
        for pole in self._poles:
            out = out + pole.process(white)
        return out * self.scale


class OnePole:
    """
    A one-pole low-pass, parameterised by its corner frequency.

    The gentlest filter there is: 6 dB per octave, no resonance, no ripple.
    Used to take the top off a noise bed, where a steeper filter would be
    audible as a colour rather than as a softening.
    """

    __slots__ = ("_stream", "coefficient")

    def __init__(self, sample_rate: int, cutoff_hz: float) -> None:
        if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate / 2.0:
            msg = (
                f"one-pole cutoff {cutoff_hz:g} Hz must sit between 0 and Nyquist "
                f"({sample_rate / 2.0:g} Hz)"
            )
            raise ValueError(msg)
        self.coefficient = 1.0 - float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
        self._stream = FilterDesign(
            b=(self.coefficient,),
            a=(1.0, self.coefficient - 1.0),
            label="one-pole",
        ).stream()

    def reset(self) -> None:
        """Clear the filter state."""
        self._stream.reset()

    def process(self, x: FloatArray) -> FloatArray:
        """Filter one block, carrying state forward."""
        return self._stream.process(x)
