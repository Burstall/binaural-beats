"""
The ``Layer`` protocol and its implementations.

A layer renders a stereo pair for a span of *absolute* sample indices. That
is the whole contract, and the reason it is expressed as a :class:`Span`
rather than a sample count: a layer is never told how many samples to
produce, it is told *which* samples to produce. There is no way to write
``arange(0, n)`` and get away with it, because the layer does not know ``n``
until it has already been given the indices those samples belong to.

Adding a sound source means writing one class. The engine never changes.

State
-----
Most layers are pure functions of time and say so with
``stateful = False``. Layers that carry state across blocks — noise
generators, IIR filters, phase accumulators — set ``stateful = True`` and
implement :meth:`StatefulLayer.reset`. The engine resets them before every
render, so a second render of the same configuration is identical to the
first. A stateful layer must be given contiguous spans in order; it has no
way to seek.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np

from violet.dsp.env import Breathing
from violet.dsp.osc import sine, sine_pair
from violet.harmony import crossfade_gain

if TYPE_CHECKING:
    from violet._types import FloatArray, IntArray, Stereo
    from violet.harmony import ChordEvent

__all__ = [
    "BinauralPair",
    "ChordBed",
    "Layer",
    "Pedal",
    "Span",
    "StatefulLayer",
]


@dataclass(frozen=True)
class Span:
    """
    A half-open range of absolute sample indices, ``[start, stop)``.

    The arrays are built once per block and shared by every layer, so a
    twenty-layer mix computes its time base once rather than twenty times.
    """

    start: int
    stop: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.start < 0:
            msg = f"span start must not be negative, got {self.start!r}"
            raise ValueError(msg)
        if self.stop <= self.start:
            msg = f"span must be non-empty, got [{self.start}, {self.stop})"
            raise ValueError(msg)
        if self.sample_rate <= 0:
            msg = f"sample rate must be positive, got {self.sample_rate!r}"
            raise ValueError(msg)

    @property
    def n(self) -> int:
        """Number of samples in the span."""
        return self.stop - self.start

    @cached_property
    def indices(self) -> IntArray:
        """Absolute sample indices."""
        out: IntArray = np.arange(self.start, self.stop, dtype=np.int64)
        return out

    @cached_property
    def t(self) -> FloatArray:
        """Absolute time in seconds, one value per sample."""
        out: FloatArray = (
            np.arange(self.start, self.stop, dtype=np.float64) / self.sample_rate
        )
        return out

    @property
    def t0(self) -> float:
        """Time of the first sample."""
        return self.start / self.sample_rate

    @property
    def t1(self) -> float:
        """Time of the last sample."""
        return (self.stop - 1) / self.sample_rate


@runtime_checkable
class Layer(Protocol):
    """One sound source in a mix."""

    #: Whether :meth:`render` depends on previous calls.
    stateful: ClassVar[bool]

    @property
    def peak(self) -> float:
        """
        Largest absolute value this layer can contribute to either channel.

        An upper bound, used to work out master headroom before a single
        sample is rendered. Layers whose peak is not analytically known —
        noise — return a practical bound rather than an infinite one.
        """
        ...

    def render(self, span: Span) -> Stereo:
        """
        Render the layer over ``span``.

        The returned arrays belong to the caller to read, not to modify, and
        the two may be the same array when a layer is mono.
        """
        ...


@runtime_checkable
class StatefulLayer(Layer, Protocol):
    """A layer that carries state from one block to the next."""

    def reset(self) -> None:
        """Return to the state the layer had before any block was rendered."""
        ...


@dataclass(frozen=True, slots=True)
class BinauralPair:
    """
    One binaural voice: a sine pair detuned by ``beat`` around ``carrier``.

    This is where the beat comes from, and the only place it should come
    from. The carrier belongs in roughly the 250-520 Hz window
    (:func:`violet.tuning.auto_octaves` will find it) — the percept is formed
    from interaural phase differences and weakens sharply below about 200 Hz,
    where headphones roll off as well.
    """

    carrier: float
    beat: float
    level: float

    stateful: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)

    @property
    def peak(self) -> float:
        """One sine per ear, so the bound is the level itself."""
        return self.level

    def render(self, span: Span) -> Stereo:
        """Render the pair over ``span``."""
        return sine_pair(span.t, self.carrier, self.beat, self.level)


@dataclass(frozen=True, slots=True)
class Pedal:
    """
    A mono drone, identical in both ears.

    Two jobs. It puts the base frequency physically in the mix, at the pitch
    it was actually given rather than the octave-shifted carrier. And because
    it never moves, chords above it read as modal colour over one tonal
    centre rather than as key changes — the tanpura trick.

    Identical in both ears means no interaural difference, so it contributes
    nothing to the beat. That is deliberate: the beat lives in the harmony.
    """

    freq: float
    level: float

    stateful: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)

    @property
    def peak(self) -> float:
        """A single sine, so the bound is the level itself."""
        return self.level

    def render(self, span: Span) -> Stereo:
        """Render the drone over ``span``, the same array to both ears."""
        drone = sine(span.t, self.freq, self.level)
        return drone, drone


@dataclass(frozen=True, slots=True)
class ChordBed:
    """
    A progression of chords, every voice its own binaural pair.

    This is the layer that makes moving harmony and a single beat compatible.
    A voice at ratio ``r`` off the root sounds ``r*root - beat/2`` in one ear
    and ``r*root + beat/2`` in the other, so *every* voice of *every* chord
    beats at exactly the same rate. The harmony can go where it likes; the
    pulse does not move.

    Events are planned in advance by :func:`violet.harmony.plan_progression`,
    which is what keeps this layer stateless: the whole progression is a
    function of the seed, decided before the first sample is rendered, so the
    layer only has to evaluate it.
    """

    events: tuple[ChordEvent, ...]
    root: float
    beat: float
    level: float
    crossfade: float = 16.0
    breathing: Breathing = field(default_factory=Breathing)

    stateful: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)
        if self.root <= 0.0:
            msg = f"root must be positive, got {self.root!r}"
            raise ValueError(msg)

    @property
    def peak(self) -> float:
        """
        Upper bound: the widest chord, doubled by root two, not by two.

        Two chords sound at once during a crossfade, but they are equal power,
        so their gains sum to at most ``sqrt(2)`` rather than to 2. Three
        chords never overlap — a chord is always held for at least one
        crossfade, which :class:`violet.harmony.WalkConfig` enforces.
        """
        widest = max((event.chord.voices for event in self.events), default=0)
        return self.level * widest * self.breathing.high * math.sqrt(2.0)

    def render(self, span: Span) -> Stereo:
        """Render every chord audible over ``span``, and no others."""
        t = span.t
        left = np.zeros(span.n, dtype=np.float64)
        right = np.zeros(span.n, dtype=np.float64)

        for event in self.events:
            audible_from, audible_to = event.support(self.crossfade)
            if audible_to < span.t0 or audible_from > span.t1:
                continue

            gain = crossfade_gain(t, event, self.crossfade)
            if not gain.any():
                continue

            for voice, freq in enumerate(event.chord.frequencies(self.root)):
                envelope = self.level * gain * self.breathing.gain(t, voice)
                voice_left, voice_right = sine_pair(t, freq, self.beat)
                left += envelope * voice_left
                right += envelope * voice_right

        return left, right
