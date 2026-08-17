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

from violet.dsp.env import Breathing, Swell, swell_envelope
from violet.dsp.filters import FilterDesign
from violet.dsp.noise import PinkNoise, spawn_seeds
from violet.dsp.osc import sine, sine_pair
from violet.harmony import crossfade_gain, tiles_loop

if TYPE_CHECKING:
    from violet._types import FloatArray, IntArray, Stereo
    from violet.dsp.filters import FilterStream
    from violet.harmony import ChordEvent

__all__ = [
    "BinauralPair",
    "ChordBed",
    "Layer",
    "LoopableLayer",
    "Ocean",
    "Pedal",
    "Span",
    "StatefulLayer",
    "snap_frequency",
]


def snap_frequency(freq: float, loop_seconds: float) -> float:
    """
    Nearest frequency completing a whole number of cycles in ``loop_seconds``.

    What makes a tone loop. A tone that does not complete whole cycles arrives
    back at the join part-way through a cycle, and the step from the last
    sample to the first is a click. Rounding the frequency to the nearest
    multiple of ``1 / loop_seconds`` removes the step entirely.

    The cost is a frequency error of at most half a cycle over the whole loop:
    0.0017 Hz for a five-minute loop, which at 336 Hz is 0.009 cents. Nothing
    hears that. Everything hears the click.
    """
    if loop_seconds <= 0.0:
        msg = f"loop length must be positive, got {loop_seconds!r}"
        raise ValueError(msg)
    return max(1.0, round(freq * loop_seconds)) / loop_seconds


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


@runtime_checkable
class LoopableLayer(Layer, Protocol):
    """
    A layer that can be made exactly periodic, and so needs no crossfade.

    Implementing this is a claim: *give me a loop length and I will return a
    version of myself whose output at sample N is the natural continuation of
    sample N-1 wrapping to sample 0*. Fixed-frequency layers can do that by
    rounding their frequencies. Noise cannot, and a chord progression cannot
    without being planned for it, so neither implements this and the engine
    crossfades them instead.
    """

    def snapped_to_loop(
        self, loop_seconds: float
    ) -> tuple[Layer, tuple[str, ...]] | None:
        """
        A periodic version of this layer, and notes on what was changed.

        ``None`` declines: the layer could in principle loop but this
        particular instance cannot, and the engine should crossfade it. The
        notes are for reporting — a render that quietly moved a frequency
        should say so.
        """
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

    def snapped_to_loop(self, loop_seconds: float) -> tuple[Layer, tuple[str, ...]]:
        """
        Move both ears onto whole cycles, keeping the beat between them.

        Both ears have to land on whole cycles, not just the carrier, so the
        beat is snapped first — to an *even* multiple of ``1 / loop_seconds``,
        which is what makes ``carrier -/+ beat/2`` land on whole cycles too.
        """
        half_cycles = max(1, round(self.beat * loop_seconds / 2.0))
        beat = 2.0 * half_cycles / loop_seconds
        carrier = snap_frequency(self.carrier, loop_seconds)

        notes: list[str] = []
        if beat != self.beat:
            notes.append(f"beat {self.beat:.6g} -> {beat:.6g} Hz to close the loop")
        if carrier != self.carrier:
            notes.append(
                f"carrier {self.carrier:.6g} -> {carrier:.6g} Hz to close the loop"
            )
        snapped = BinauralPair(carrier=carrier, beat=beat, level=self.level)
        return snapped, tuple(notes)


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

    def snapped_to_loop(self, loop_seconds: float) -> tuple[Layer, tuple[str, ...]]:
        """Move the drone onto whole cycles."""
        freq = snap_frequency(self.freq, loop_seconds)
        notes = (
            ()
            if freq == self.freq
            else (f"pedal {self.freq:.6g} -> {freq:.6g} Hz to close the loop",)
        )
        return Pedal(freq=freq, level=self.level), notes


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

    #: Set when the progression tiles a loop exactly. Voice frequencies are
    #: then snapped to whole cycles and the events are evaluated cyclically, so
    #: the bed is periodic and needs no crossfade at the join.
    loop_seconds: float | None = None

    stateful: ClassVar[bool] = False

    _voices: tuple[tuple[float, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)
        if self.root <= 0.0:
            msg = f"root must be positive, got {self.root!r}"
            raise ValueError(msg)
        if self.loop_seconds is not None and not tiles_loop(
            self.events, self.loop_seconds
        ):
            msg = (
                "a looping ChordBed needs a progression that tiles the loop; "
                "plan it with harmony.plan_loop_progression"
            )
            raise ValueError(msg)

        loop = self.loop_seconds
        voices = tuple(
            tuple(
                snap_frequency(freq, loop) if loop is not None else freq
                for freq in event.chord.frequencies(self.root)
            )
            for event in self.events
        )
        object.__setattr__(self, "_voices", voices)

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
        breaths = [self.breathing.gain(t, voice) for voice in range(self._widest)]

        # When looping, the progression is a cycle rather than a line: time is
        # taken modulo the loop, and each event is also evaluated one loop
        # either side so the chords that straddle the join are heard on both
        # sides of it. Only the chord *gains* wrap. The tones themselves stay
        # on absolute time, which is what keeps their phase continuous — and
        # their frequencies have been snapped, so a whole loop of absolute time
        # brings them back to where they started anyway.
        loop = self.loop_seconds
        if loop is None:
            clock = t
            offsets: tuple[float, ...] = (0.0,)
        else:
            clock = np.mod(t, loop)
            offsets = (-loop, 0.0, loop)
        first, last = float(clock.min()), float(clock.max())

        for offset in offsets:
            for event, frequencies in zip(self.events, self._voices, strict=True):
                audible_from, audible_to = event.support(self.crossfade)
                if audible_to + offset < first or audible_from + offset > last:
                    continue

                gain = crossfade_gain(clock - offset, event, self.crossfade)
                if not gain.any():
                    continue

                for voice, freq in enumerate(frequencies):
                    envelope = self.level * gain * breaths[voice]
                    voice_left, voice_right = sine_pair(t, freq, self.beat)
                    left += envelope * voice_left
                    right += envelope * voice_right

        return left, right

    @property
    def _widest(self) -> int:
        return max((event.chord.voices for event in self.events), default=0)

    def snapped_to_loop(
        self, loop_seconds: float
    ) -> tuple[Layer, tuple[str, ...]] | None:
        """
        A periodic version of this bed, if the progression closes.

        Returns ``None`` when the progression was not planned for a loop, in
        which case the engine crossfades the bed instead. That fallback works,
        but it is second best: two chords a whole loop apart usually share
        voices, and crossfading a tone against a phase-shifted copy of itself
        makes it swell and dip as the phases fight. A closed progression has no
        join to crossfade.
        """
        if not tiles_loop(self.events, loop_seconds):
            return None

        half_cycles = max(1, round(self.beat * loop_seconds / 2.0))
        beat = 2.0 * half_cycles / loop_seconds
        breathing = self.breathing.snapped_to_loop(loop_seconds)

        notes: list[str] = []
        if beat != self.beat:
            notes.append(f"beat {self.beat:.6g} -> {beat:.6g} Hz to close the loop")
        notes.append("chord voices snapped to whole cycles to close the loop")

        snapped = ChordBed(
            events=self.events,
            root=self.root,
            beat=beat,
            level=self.level,
            crossfade=self.crossfade,
            breathing=breathing,
            loop_seconds=loop_seconds,
        )
        return snapped, tuple(notes)


@dataclass(slots=True)
class Ocean:
    """
    Two decorrelated streams of pink noise, shaped into waves.

    Stateful: three IIR filters and one noise generator per channel, all of
    which carry their history forward. It must be rendered in order from the
    start; there is no way to seek into the middle of a filter's memory. The
    layer checks that the spans it is handed are contiguous rather than
    trusting the caller.

    Decorrelated, and that is the point
    -----------------------------------
    The two channels come from independent generators, so the noise is
    uncorrelated between the ears. Uncorrelated noise *cannot* carry a binaural
    beat — the percept is built from interaural phase differences and there is
    no consistent phase relationship to find. So the ocean is texture and the
    harmony is the beat, cleanly separated. Feeding one noise stream to both
    ears would put a wide correlated wash right on top of the thing the beat
    needs to be legible in.

    The notch
    ---------
    Broadband noise sitting over the carrier masks the beat. There is a narrow
    dip carved at the carrier frequency for exactly that reason. You do not
    hear the notch — a 2-Q dip in surf is nothing — and the pulse stays audible
    underneath a bed loud enough to hide it otherwise.

    Two bands
    ---------
    A dark band, low-passed, carries the body of the wave and follows the swell
    envelope directly. A bright band, band-passed, carries the hiss of the
    break and follows a steeper version of the same envelope, so it only shows
    up on the bigger waves. One band alone sounds like a hiss or a rumble;
    together they sound like water.
    """

    swells: tuple[Swell, ...]
    set_period: float
    notch_hz: float
    level: float
    sample_rate: int

    seed: int = 2027
    q: float = 2.0
    white_scale: float = 0.35
    dark_hz: float = 520.0
    bright_hz: tuple[float, float] = (900.0, 6500.0)
    bright_mix: float = 0.55
    bright_shape: float = 1.6
    bright_ceiling: float = 1.6

    #: Practical amplitude bound for one channel of the notched, enveloped
    #: noise at ``level = 1``. Gaussian noise has no true maximum, so this is
    #: measured rather than analytic: five-minute renders across six seeds peak
    #: between 0.077 and 0.095, and this leaves half again on top. There is a
    #: test that renders the bed and checks nothing exceeds it.
    peak_estimate: float = 0.15

    stateful: ClassVar[bool] = True

    _pink: list[PinkNoise] = field(init=False, repr=False)
    _dark: list[FilterStream] = field(init=False, repr=False)
    _bright: list[FilterStream] = field(init=False, repr=False)
    _notch: list[FilterStream] = field(init=False, repr=False)
    _next_index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)
        if self.sample_rate <= 0:
            msg = f"sample rate must be positive, got {self.sample_rate!r}"
            raise ValueError(msg)

        dark = FilterDesign.butterworth(self.sample_rate, self.dark_hz, "low")
        bright = FilterDesign.butterworth(self.sample_rate, self.bright_hz, "band")
        notch = FilterDesign.notch(self.sample_rate, self.notch_hz, self.q)

        seeds = spawn_seeds(self.seed, 2)
        self._pink = [PinkNoise(seed, self.white_scale) for seed in seeds]
        self._dark = [dark.stream(), dark.stream()]
        self._bright = [bright.stream(), bright.stream()]
        self._notch = [notch.stream(), notch.stream()]
        self._next_index = 0

    @property
    def peak(self) -> float:
        """Practical bound on one channel's contribution."""
        return self.level * self.peak_estimate

    def reset(self) -> None:
        """Restart both noise streams and clear every filter."""
        for noise in self._pink:
            noise.reset()
        for bank in (self._dark, self._bright, self._notch):
            for filtered in bank:
                filtered.reset()
        self._next_index = 0

    def render(self, span: Span) -> Stereo:
        """Render the bed over ``span``, which must follow the previous one."""
        if span.sample_rate != self.sample_rate:
            msg = (
                f"Ocean was built for {self.sample_rate} Hz but asked to render "
                f"at {span.sample_rate} Hz; its filters are designed for one rate"
            )
            raise ValueError(msg)
        if span.start != self._next_index:
            msg = (
                f"Ocean must be rendered in order: expected the block starting at "
                f"sample {self._next_index}, got {span.start}. Its filter state "
                f"cannot be seeked; call reset() to start again."
            )
            raise ValueError(msg)

        envelope = swell_envelope(span.t, self.swells, self.set_period)
        bright_envelope = (
            np.clip(envelope, 0.0, self.bright_ceiling) ** self.bright_shape
        )

        channels: list[FloatArray] = []
        for index in range(2):
            pink = self._pink[index].block(span.n)
            dark = self._dark[index].process(pink)
            bright = self._bright[index].process(pink)
            mixed = dark * envelope + bright * self.bright_mix * bright_envelope
            channels.append(self._notch[index].process(mixed) * self.level)

        self._next_index = span.stop
        return channels[0], channels[1]
