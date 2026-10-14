"""
The ``Layer`` protocol and its implementations.

A layer renders a stereo pair for a span of *absolute* sample indices. That
is the whole contract, and the reason it is expressed as a :class:`Span`
rather than a sample count: a layer is never told how many samples to
produce, it is told *which* samples to produce. There is no way to write
``arange(0, n)`` and get away with it, because the layer does not know ``n``
until it has already been given the indices those samples belong to.

Adding a sound source means writing one class. The engine never changes.
That holds for processing too: :class:`Modulated` takes other layers and hands
back their sum with a pulse applied, and to the engine it is one more layer.

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
import soundfile as sf

from violet.dsp.curves import BeatCurve
from violet.dsp.env import Breathing, Lfo, Swell, swell_envelope
from violet.dsp.filters import FilterDesign
from violet.dsp.modulation import band_gain_bound
from violet.dsp.noise import KelletPink, OnePole, PinkNoise, spawn_seeds
from violet.dsp.osc import sine, sine_pair, swept_pair
from violet.harmony import crossfade_gain, tiles_loop

if TYPE_CHECKING:
    from pathlib import Path

    from violet._types import FloatArray, IntArray, Stereo
    from violet.dsp.filters import FilterStream
    from violet.dsp.modulation import Modulation
    from violet.harmony import ChordEvent

__all__ = [
    "Air",
    "BinauralPair",
    "ChordBed",
    "Layer",
    "LoopableLayer",
    "Modulated",
    "Ocean",
    "Pedal",
    "Span",
    "StatefulLayer",
    "Track",
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

    #: A fixed rate, or a :class:`~violet.dsp.curves.BeatCurve` that moves.
    beat: float | BeatCurve
    level: float

    #: An optional slow swell, so a layer above the root breathes rather than
    #: sitting still. Long periods, and never all the way down to silence.
    swell: Lfo | None = None

    stateful: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)

    @property
    def peak(self) -> float:
        """One sine per ear, so the bound is the level at its loudest."""
        return self.level * (1.0 if self.swell is None else self.swell.high)

    def render(self, span: Span) -> Stereo:
        """Render the pair over ``span``."""
        if isinstance(self.beat, BeatCurve):
            left, right = swept_pair(span.t, self.carrier, self.beat, self.level)
        else:
            left, right = sine_pair(span.t, self.carrier, self.beat, self.level)
        if self.swell is None:
            return left, right
        gain = self.swell.gain(span.t)
        return left * gain, right * gain

    def snapped_to_loop(
        self, loop_seconds: float
    ) -> tuple[Layer, tuple[str, ...]] | None:
        """
        Move both ears onto whole cycles, keeping the beat between them.

        Both ears have to land on whole cycles, not just the carrier, so the
        beat is snapped first — to an *even* multiple of ``1 / loop_seconds``,
        which is what makes ``carrier -/+ beat/2`` land on whole cycles too.

        A swept beat declines: see :class:`ChordBed.snapped_to_loop` for why a
        moving rate and a seamless loop do not go together.
        """
        if isinstance(self.beat, BeatCurve):
            return None
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
        swell = None if self.swell is None else self.swell.snapped_to_loop(loop_seconds)
        snapped = BinauralPair(
            carrier=carrier, beat=beat, level=self.level, swell=swell
        )
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

    #: A fixed rate, or a :class:`~violet.dsp.curves.BeatCurve` that moves.
    #: Every voice follows the same one, which is what keeps the mix to a
    #: single beat however the harmony moves.
    beat: float | BeatCurve
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
                    if isinstance(self.beat, BeatCurve):
                        voice_left, voice_right = swept_pair(t, freq, self.beat)
                    else:
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
        if isinstance(self.beat, BeatCurve):
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
class Air:
    """
    A quiet low-passed noise bed, the same in both ears.

    Not the ocean. This is the sound of a room rather than the sound of water:
    a soft floor under the tones that stops a mix of pure sines from feeling
    like a test signal. Low-passed hard, so it sits below the harmony instead
    of over it.

    Mono, and unlike the ocean that is fine. Identical in both ears means no
    interaural difference at all, so it adds nothing the brainstem can read as
    a beat and nothing that competes with the one in the harmony. It is not
    notched, because at a 140 Hz corner there is nothing left up at the carrier
    to notch.
    """

    level: float
    sample_rate: int
    cutoff_hz: float = 141.7974
    seed: int = 7

    #: Measured bound at ``level = 1``, the same way the ocean's is: five-minute
    #: renders across seven seeds and three sample rates peak between 8.9 and
    #: 11.2, and this leaves half again on top. Large, because Kellet's slowest
    #: pole has a DC gain of about 42 and the low-pass keeps all of it — this
    #: bed is nearly all rumble. It once said 0.35, which no render ever
    #: respected; nothing noticed because the one preset using it set its gain
    #: by hand. There is now a test.
    peak_estimate: float = 17.0

    stateful: ClassVar[bool] = True

    _pink: KelletPink = field(init=False, repr=False)
    _lowpass: OnePole = field(init=False, repr=False)
    _next_index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)
        self._pink = KelletPink(self.seed, self.level)
        self._lowpass = OnePole(self.sample_rate, self.cutoff_hz)
        self._next_index = 0

    @property
    def peak(self) -> float:
        """Practical bound on the bed's contribution."""
        return self.level * self.peak_estimate

    def reset(self) -> None:
        """Restart the generator and clear the filter."""
        self._pink.reset()
        self._lowpass.reset()
        self._next_index = 0

    def render(self, span: Span) -> Stereo:
        """Render the bed over ``span``, the same array to both ears."""
        if span.start != self._next_index:
            msg = (
                f"Air must be rendered in order: expected the block starting at "
                f"sample {self._next_index}, got {span.start}"
            )
            raise ValueError(msg)
        bed = self._lowpass.process(self._pink.block(span.n))
        self._next_index = span.stop
        return bed, bed


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


#: Frames read per chunk when measuring a track's peak. A few seconds at a
#: time keeps memory flat however long the file is.
_MEASURE_FRAMES = 1 << 18

#: Containers libsndfile cannot seek to the exact sample. Measured, not
#: assumed: seeking an Ogg Vorbis file lands within a packet of the target, so
#: reading it in 0.37 s blocks and in 10 s blocks gave samples up to 0.47
#: apart. WAV, FLAC and MP3 seek exactly and read back identically either way.
_SEQUENTIAL_FORMATS = frozenset({"OGG"})


class _Reader:
    """
    An open audio file and how far into it the last read went.

    Reading straight on from where the previous block stopped needs no seek
    at all, and a render always asks for its blocks in order — so in practice
    the only seek a render makes is back to the start when a track repeats.
    For a container that cannot seek exactly, that one is done by reopening
    the file, which is exact by definition, and any other jump is refused
    rather than quietly misread.
    """

    __slots__ = ("_handle", "path", "position", "sequential")

    def __init__(self, path: Path, *, sequential: bool) -> None:
        self.path = path
        self.sequential = sequential
        self.position = 0
        self._handle: sf.SoundFile | None = None

    def read(self, start: int, count: int) -> FloatArray:
        """``count`` frames from frame ``start``, as ``(count, channels)``."""
        if self._handle is None or start != self.position:
            if self._handle is not None and start == 0 and self.sequential:
                self.close()
            if self._handle is None:
                self._handle = sf.SoundFile(str(self.path))
                self.position = 0
            if start != self.position:
                if self.sequential:
                    msg = (
                        f"{self.path.name} cannot be seeked to the sample, so it "
                        f"has to be read in order: asked for frame {start} with "
                        f"the file at {self.position}. Convert it to FLAC to "
                        f"read it in any order."
                    )
                    raise ValueError(msg)
                self._handle.seek(start)
        out: FloatArray = self._handle.read(count, dtype="float64", always_2d=True)
        self.position = start + count
        return out

    def close(self) -> None:
        """Let go of the file. The next read opens it again."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.position = 0


@dataclass(frozen=True, slots=True)
class Track:
    """
    A piece of recorded music, streamed from a file.

    Stateless for WAV, FLAC and MP3, because those can be seeked to the
    sample: the layer is handed a span of absolute indices and reads exactly
    those frames, so it renders at any block size and in any order and gives
    the same samples. Only the file handle is kept between blocks, and only so
    that reading straight on needs no seek. Ogg files cannot be seeked
    exactly and must be read in order — which is how the engine reads them
    anyway — and the layer refuses a jump rather than misreading one.

    Mono files go to both ears. Anything over two channels is refused rather
    than guessed at.

    The file has to be at the render's sample rate. Resampling block by block
    without seams is a project of its own, and a render at the track's own rate
    loses nothing, so the layer says so and leaves the rate to whoever builds
    the render.

    ``repeat`` plays the file again from the top when it runs out, for
    sessions longer than the track. The join is whatever the file's own end and
    start make it — a track that fades out and in again is heard doing so.
    Without ``repeat`` the layer is silent past the end of the file.
    """

    path: Path
    level: float = 1.0
    repeat: bool = False

    stateful: ClassVar[bool] = False

    frames: int = field(init=False, default=0)
    sample_rate: int = field(init=False, default=0)
    channels: int = field(init=False, default=0)

    #: The file's own peak, measured once when the layer is built. Exact rather
    #: than estimated: this is recorded audio, and it can simply be read.
    file_peak: float = field(init=False, default=0.0)

    _reader: _Reader = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.level < 0.0:
            msg = f"level must not be negative, got {self.level!r}"
            raise ValueError(msg)
        if not self.path.is_file():
            msg = f"no audio file at {self.path}"
            raise FileNotFoundError(msg)
        info = sf.info(str(self.path))
        if info.channels not in {1, 2}:
            msg = (
                f"{self.path.name} has {info.channels} channels; a track must be "
                f"mono or stereo"
            )
            raise ValueError(msg)
        if info.frames <= 0:
            msg = f"{self.path.name} has no audio in it"
            raise ValueError(msg)
        object.__setattr__(self, "frames", int(info.frames))
        object.__setattr__(self, "sample_rate", int(info.samplerate))
        object.__setattr__(self, "channels", int(info.channels))
        object.__setattr__(self, "file_peak", self._measure_peak())
        sequential = info.format in _SEQUENTIAL_FORMATS
        object.__setattr__(self, "_reader", _Reader(self.path, sequential=sequential))

    @property
    def seconds(self) -> float:
        """Length of the file."""
        return self.frames / self.sample_rate

    @property
    def peak(self) -> float:
        """The file's measured peak, at this layer's level."""
        return self.level * self.file_peak

    def _measure_peak(self) -> float:
        peak = 0.0
        with sf.SoundFile(str(self.path)) as handle:
            for chunk in handle.blocks(blocksize=_MEASURE_FRAMES, dtype="float64"):
                peak = max(peak, float(np.abs(chunk).max()))
        return peak

    def render(self, span: Span) -> Stereo:
        """Read the frames ``span`` names, wrapping if the track repeats."""
        if span.sample_rate != self.sample_rate:
            msg = (
                f"{self.path.name} is at {self.sample_rate} Hz but the render is "
                f"at {span.sample_rate} Hz; render at the track's own rate"
            )
            raise ValueError(msg)

        block = np.zeros((span.n, self.channels), dtype=np.float64)
        if self.repeat:
            filled = 0
            position = span.start % self.frames
            while filled < span.n:
                count = min(span.n - filled, self.frames - position)
                block[filled : filled + count] = self._reader.read(position, count)
                filled += count
                position = 0
        elif span.start < self.frames:
            count = min(span.stop, self.frames) - span.start
            block[:count] = self._reader.read(span.start, count)

        if self._reader.position >= self.frames:
            self._reader.close()

        block *= self.level
        if self.channels == 1:
            return block[:, 0], block[:, 0]
        return block[:, 0], block[:, 1]


@dataclass(slots=True)
class Modulated:
    """
    Other layers, summed, with an amplitude modulation applied to the result.

    The one layer that processes rather than generates. It renders what it
    wraps over the span it is given, adds them up, and dips a band of the sum
    by :meth:`~violet.dsp.modulation.Modulation.dip` — the same in both ears,
    so the pulse is in the loudness and not in the interaural phase. See
    :mod:`violet.dsp.modulation` for what that is and what is known about it.

    Stateful, because the band-pass that isolates the modulated band carries
    its history from block to block: one filter per ear, never shared. That
    makes it the same kind of layer as the ocean — rendered in order, reset
    before every render — whatever it wraps.

    At a depth of zero the output is the sum of the wrapped layers, exactly.
    No filter runs and nothing is subtracted, which is what lets a render
    without modulation serve as the placebo for one with it.

    Not loopable: a filter's memory cannot be made periodic by rounding a
    frequency. The presets refuse a modulated loop rather than let the engine
    crossfade the whole mix, which would comb-filter every tone in it.
    """

    layers: tuple[Layer, ...]
    modulation: Modulation
    sample_rate: int

    stateful: ClassVar[bool] = True

    _streams: list[FilterStream] = field(init=False, repr=False)
    _band_bound: float = field(init=False, repr=False)
    _next_index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.layers:
            msg = "a modulated group needs at least one layer to modulate"
            raise ValueError(msg)
        design = self.modulation.design(self.sample_rate)
        self._streams = [design.stream(), design.stream()]
        self._band_bound = band_gain_bound(design, self.sample_rate)
        self._next_index = 0

    @property
    def peak(self) -> float:
        """
        What the wrapped layers can reach, plus what the dip can add.

        ``x - d*m*band`` is at most ``|x| + d*|band|``, and the band can reach
        the filter's L1 gain times the input. Pessimistic, because it assumes
        the band peaks against the full signal at the bottom of a pulse, but it
        is a bound and not a guess. At depth zero it is the wrapped layers' own
        peak, as it should be.
        """
        inner = sum(layer.peak for layer in self.layers)
        return inner * (1.0 + self.modulation.depth * self._band_bound)

    def reset(self) -> None:
        """Reset everything wrapped, and both band filters."""
        for layer in self.layers:
            if isinstance(layer, StatefulLayer):
                layer.reset()
        for stream in self._streams:
            stream.reset()
        self._next_index = 0

    def render(self, span: Span) -> Stereo:
        """Render the wrapped layers over ``span`` and modulate their sum."""
        if span.sample_rate != self.sample_rate:
            msg = (
                f"Modulated was built for {self.sample_rate} Hz but asked to "
                f"render at {span.sample_rate} Hz; its filters are designed for "
                f"one rate"
            )
            raise ValueError(msg)
        if span.start != self._next_index:
            msg = (
                f"Modulated must be rendered in order: expected the block "
                f"starting at sample {self._next_index}, got {span.start}. Its "
                f"band filters cannot be seeked; call reset() to start again."
            )
            raise ValueError(msg)

        # Fresh arrays: what a layer returns belongs to it, and a mono layer
        # hands back the same array for both ears.
        left = np.zeros(span.n, dtype=np.float64)
        right = np.zeros(span.n, dtype=np.float64)
        for layer in self.layers:
            layer_left, layer_right = layer.render(span)
            left += layer_left
            right += layer_right
        self._next_index = span.stop

        if not self.modulation.active:
            return left, right

        dip = self.modulation.dip(span.t)
        left -= dip * self._streams[0].process(left)
        right -= dip * self._streams[1].process(right)
        return left, right
