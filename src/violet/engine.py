"""
The block-streaming renderer.

The engine owns the loop and nothing else. It walks absolute sample indices
in blocks, asks each layer for its stereo contribution over that span, sums
them, applies the master fade and gain, and hands the block to a sink. A
90-minute render costs the same working set as a 5-second one.

Block boundaries are an implementation detail and must never be audible.
Everything the engine computes — the time base, the fade — is derived from
absolute sample indices for exactly that reason, and layers are handed those
indices rather than a sample count so they cannot do otherwise. Rendering the
same configuration at a 1-second block size and a 10-second block size gives
the same samples, to within float rounding, and there is a test that says so.

Headroom
--------
Master gain can be given explicitly, or left to the engine. Left to the
engine, it sums the analytic peak of every layer and scales the mix so that
sum lands on ``ceiling``. It only ever attenuates, never boosts: a quiet mix
stays quiet rather than being pumped up to the ceiling.

The bound is per channel and is usually pessimistic, because it assumes every
layer hits its maximum in the same sample and in the same direction. Sines at
unrelated frequencies do not. That is the right way round for a safety
margin.

Seamless loops
--------------
A file that is going to be played on repeat has to join onto itself without a
click, and different layers need different treatment to get there.

Fixed-frequency layers are made *exactly* periodic: rounding a frequency to the
nearest whole number of cycles over the loop length moves it by a fraction of a
cent and removes the join entirely. Layers that cannot be made periodic —
noise, and a chord progression that was not planned to close — are handled with
a tail crossfade: render past the end, then fold that overhang back over the
opening with equal-power gains, so the first sample of the file continues the
signal that the last sample was in the middle of.

The two mechanisms are deliberately not interchangeable. Crossfading a tone
against a phase-shifted copy of itself makes it comb-filter, dipping and
swelling as the phases fight; snapping noise to a frequency is meaningless.
Each layer declares which it can do by implementing
:class:`~violet.layers.LoopableLayer` or not.

No ffmpeg
---------
Files are written by libsndfile through ``soundfile``, block by block, as WAV
or FLAC. Nothing is buffered in memory and nothing is shelled out to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from violet.dsp.env import fade_envelope, fade_length
from violet.layers import LoopableLayer, Span, StatefulLayer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from violet._types import FloatArray, Stereo
    from violet.layers import Layer

__all__ = [
    "ArraySink",
    "LoopConfig",
    "RenderConfig",
    "RenderResult",
    "Sink",
    "SoundFileSink",
    "headroom_gain",
    "loop_length",
    "render",
    "render_to_file",
]


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """
    Close a render into a loop that can be played on repeat indefinitely.

    ``crossfade`` only applies to layers that cannot be made periodic. A render
    made entirely of fixed-frequency layers is snapped instead and no crossfade
    happens at all, which is both cheaper and exact.

    Twelve seconds is a long crossfade by any normal standard, and the right
    order of magnitude here: the ocean's wave events run to ten seconds and the
    chords move over forty, so a shorter fold would be heard as an event rather
    than as the texture continuing.
    """

    crossfade: float = 12.0

    def __post_init__(self) -> None:
        if self.crossfade <= 0.0:
            msg = f"loop crossfade must be positive, got {self.crossfade!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """How to render, as opposed to what to render."""

    sample_rate: int
    duration: float

    #: Block size in seconds. Affects memory and nothing else — any value
    #: gives the same output.
    block_seconds: float = 10.0

    #: Master gain. ``None`` derives it from the layers' analytic peaks.
    gain: float | None = None

    #: Target for the derived gain. Ignored when ``gain`` is given.
    ceiling: float = 0.89

    #: Fade in and out, in seconds. Zero for a seamless loop.
    fade_seconds: float = 8.0

    #: The fade is capped at ``total // fade_max_denominator`` samples.
    fade_max_denominator: int = 4

    #: Set to close the render into a seamless loop. Requires no master fade.
    loop: LoopConfig | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            msg = f"sample rate must be positive, got {self.sample_rate!r}"
            raise ValueError(msg)
        if self.duration <= 0.0:
            msg = f"duration must be positive, got {self.duration!r}"
            raise ValueError(msg)
        if self.block_seconds <= 0.0:
            msg = f"block_seconds must be positive, got {self.block_seconds!r}"
            raise ValueError(msg)
        if self.gain is not None and self.gain <= 0.0:
            msg = f"gain must be positive when given, got {self.gain!r}"
            raise ValueError(msg)
        if self.ceiling <= 0.0:
            msg = f"ceiling must be positive, got {self.ceiling!r}"
            raise ValueError(msg)
        if self.loop is not None and self.fade_seconds != 0.0:
            msg = (
                "a looping render cannot have a master fade: fading to silence "
                "and starting again is exactly the break a loop is meant to "
                "avoid. Set fade_seconds=0.0."
            )
            raise ValueError(msg)

    @property
    def frames(self) -> int:
        """Total number of samples the render will produce."""
        return round(self.sample_rate * self.duration)

    @property
    def block_frames(self) -> int:
        """Block size in samples, at least one."""
        return max(1, round(self.sample_rate * self.block_seconds))


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What came out, and whether it was safe."""

    frames: int
    sample_rate: int
    blocks: int
    gain: float
    peak: float
    clipped: int

    #: Length of the loop crossfade actually applied, in samples. Zero when
    #: not looping, and also zero when every layer could be made periodic and
    #: no crossfade was needed.
    loop_crossfade_frames: int = 0

    #: Anything the render changed on its own, for reporting. Snapping a
    #: frequency to close a loop is a change to the sound, however small, and
    #: a render that does it silently is lying.
    notes: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        """Length of the render."""
        return self.frames / self.sample_rate

    @property
    def peak_dbfs(self) -> float:
        """Peak in dBFS, or negative infinity for silence."""
        return -np.inf if self.peak <= 0.0 else float(20.0 * np.log10(self.peak))


@runtime_checkable
class Sink(Protocol):
    """Somewhere for finished blocks to go."""

    def write(self, block: FloatArray) -> None:
        """Accept one ``(frames, 2)`` block of float samples."""
        ...


class ArraySink:
    """Collects blocks in memory. For tests and for short renders."""

    def __init__(self) -> None:
        self._blocks: list[FloatArray] = []

    def write(self, block: FloatArray) -> None:
        """Keep a copy of the block."""
        self._blocks.append(np.asarray(block, dtype=np.float64).copy())

    @property
    def result(self) -> FloatArray:
        """Everything written so far, as one ``(frames, 2)`` array."""
        if not self._blocks:
            return np.zeros((0, 2), dtype=np.float64)
        out: FloatArray = np.concatenate(self._blocks, axis=0)
        return out


class SoundFileSink:
    """
    Streams blocks straight to a WAV or FLAC file through libsndfile.

    The format follows the file extension and the sample format follows the
    format's default — 16-bit for both WAV and FLAC — unless ``subtype`` says
    otherwise. Quantisation happens here and nowhere else.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        channels: int = 2,
        subtype: str | None = None,
    ) -> None:
        self.path = path
        self._file: Any = sf.SoundFile(
            str(path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
        )

    def write(self, block: FloatArray) -> None:
        """Write one block."""
        self._file.write(block)

    def close(self) -> None:
        """Close the file. Safe to call twice."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> SoundFileSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def loop_length(seconds: float, beat: float) -> float:
    """
    Round ``seconds`` to a whole number of beat cycles.

    This is what the prototype does, and it is only half of a seamless loop.
    It makes the *beat* meet itself at the join: the interaural phase
    difference comes back to where it started, so the pulse does not stumble.
    It says nothing about the carrier. A 333.796 Hz tone fits 1001.388 cycles
    into three seconds, and that leftover 0.388 of a cycle is a step in the
    waveform at the join — a click.

    A loop is seamless only when the carrier completes whole cycles over the
    same length, which means nudging the carrier off the note by a fraction
    of a cent. Both behaviours are pinned by tests; choosing between them is
    a decision for the preset, not for this function.

    Fades must be off either way, which is why loop presets set
    ``fade_seconds`` to zero.
    """
    if beat <= 0.0:
        return seconds
    cycles = max(round(seconds * beat), 1)
    return cycles / beat


def headroom_gain(
    layers: Iterable[Layer],
    ceiling: float,
    extra: Iterable[Layer] = (),
    *,
    boost: bool = False,
) -> float:
    """
    Master gain that keeps the summed analytic peak at or under ``ceiling``.

    Attenuates or does nothing; never boosts.

    ``extra`` is the group being folded back on itself by a loop crossfade,
    which for the length of the fold contributes twice. Equal-power gains sum
    to at most root two, so that is the allowance made when ``boost`` is set.
    """
    total = sum(layer.peak for layer in layers)
    overlap = sum(layer.peak for layer in extra)
    total += overlap * (math.sqrt(2.0) if boost else 1.0)
    if total <= 0.0:
        return 1.0
    return min(1.0, ceiling / total)


def _reset(layers: Iterable[Layer]) -> None:
    for layer in layers:
        if isinstance(layer, StatefulLayer):
            layer.reset()


def _accumulate(layers: Iterable[Layer], span: Span) -> Stereo:
    """Sum every layer's contribution over one span."""
    left = np.zeros(span.n, dtype=np.float64)
    right = np.zeros(span.n, dtype=np.float64)
    for layer in layers:
        layer_left, layer_right = layer.render(span)
        left += layer_left
        right += layer_right
    return left, right


def _prepare_loop(
    layers: Sequence[Layer], loop_seconds: float
) -> tuple[list[Layer], list[Layer], tuple[str, ...]]:
    """
    Split the layers into those that can be made periodic and those that cannot.

    The periodic ones come back snapped to the loop length. The rest are
    returned untouched, for the tail crossfade to deal with.
    """
    periodic: list[Layer] = []
    aperiodic: list[Layer] = []
    notes: list[str] = []

    for layer in layers:
        result = (
            layer.snapped_to_loop(loop_seconds)
            if isinstance(layer, LoopableLayer)
            else None
        )
        if result is None:
            aperiodic.append(layer)
        else:
            snapped, layer_notes = result
            periodic.append(snapped)
            notes.extend(layer_notes)

    return periodic, aperiodic, tuple(notes)


def _render_overhang(
    layers: Sequence[Layer],
    config: RenderConfig,
    frames: int,
    overhang: int,
) -> FloatArray:
    """
    Render past the end of the loop and keep the overhang.

    A first pass, needed because a stateful layer's value at sample ``frames``
    depends on every sample before it — there is no way to jump there. Only the
    overhang is kept, so the cost is a second pass over the aperiodic layers
    and ``overhang`` samples of memory, not a second copy of the render.
    """
    _reset(layers)
    tail = np.zeros((overhang, 2), dtype=np.float64)
    total = frames + overhang
    block_frames = config.block_frames

    for start in range(0, total, block_frames):
        span = Span(start, min(start + block_frames, total), config.sample_rate)
        left, right = _accumulate(layers, span)
        if span.stop <= frames:
            continue
        keep_from = max(span.start, frames)
        source = slice(keep_from - span.start, span.n)
        target = slice(keep_from - frames, span.stop - frames)
        tail[target, 0] = left[source]
        tail[target, 1] = right[source]

    return tail


def _loop_gains(indices: FloatArray, overhang: int) -> tuple[FloatArray, FloatArray]:
    """
    Equal-power gains for folding the overhang back over the opening.

    Equal power, not equal amplitude, because the two sides of this fold are
    uncorrelated — different noise, a different chord. Their powers add, so
    ``sin`` against ``cos`` keeps the total steady. Equal amplitude would sag
    3 dB in the middle.
    """
    phase = indices / overhang * np.pi / 2.0
    return np.sin(phase), np.cos(phase)


def render(
    layers: Sequence[Layer],
    config: RenderConfig,
    sink: Sink,
) -> RenderResult:
    """
    Render ``layers`` into ``sink``, one block at a time.

    Stateful layers are reset first, so the result depends on the
    configuration and nothing else.
    """
    frames = config.frames
    if frames <= 0:
        msg = f"nothing to render: {config.duration!r} seconds is zero samples"
        raise ValueError(msg)

    periodic: Sequence[Layer] = layers
    aperiodic: Sequence[Layer] = ()
    notes: tuple[str, ...] = ()
    overhang = 0
    tail: FloatArray | None = None

    if config.loop is not None:
        periodic, aperiodic, notes = _prepare_loop(layers, frames / config.sample_rate)
        if aperiodic:
            overhang = max(1, round(config.loop.crossfade * config.sample_rate))
            if overhang > frames:
                msg = (
                    f"loop crossfade of {config.loop.crossfade:g} s does not fit "
                    f"in a {config.duration:g} s render"
                )
                raise ValueError(msg)
            tail = _render_overhang(aperiodic, config, frames, overhang)

    _reset(layers)
    gain = (
        config.gain
        if config.gain is not None
        else headroom_gain(
            periodic, config.ceiling, extra=aperiodic, boost=overhang > 0
        )
    )
    fade = fade_length(
        frames,
        config.sample_rate,
        config.fade_seconds,
        config.fade_max_denominator,
    )

    block_frames = config.block_frames
    peak = 0.0
    clipped = 0
    blocks = 0

    for start in range(0, frames, block_frames):
        span = Span(start, min(start + block_frames, frames), config.sample_rate)
        left, right = _accumulate(periodic, span)

        if aperiodic:
            wrap_left, wrap_right = _accumulate(aperiodic, span)
            if tail is not None and span.start < overhang:
                inside = span.indices < overhang
                position = span.indices[inside].astype(np.float64)
                rise, fall = _loop_gains(position, overhang)
                folded = span.indices[inside]
                wrap_left[inside] = wrap_left[inside] * rise + tail[folded, 0] * fall
                wrap_right[inside] = wrap_right[inside] * rise + tail[folded, 1] * fall
            left += wrap_left
            right += wrap_right

        if fade > 0:
            env = fade_envelope(span.indices, frames, fade)
            left *= env
            right *= env

        stereo = np.stack([left, right], axis=1)
        stereo *= gain

        magnitude = np.abs(stereo)
        peak = max(peak, float(magnitude.max()))
        clipped += int(np.count_nonzero(magnitude >= 1.0))
        blocks += 1

        sink.write(stereo)

    return RenderResult(
        frames=frames,
        sample_rate=config.sample_rate,
        blocks=blocks,
        gain=gain,
        peak=peak,
        clipped=clipped,
        loop_crossfade_frames=overhang,
        notes=notes,
    )


def render_to_file(
    layers: Sequence[Layer],
    config: RenderConfig,
    path: str | Path,
    subtype: str | None = None,
) -> RenderResult:
    """Render straight to a WAV or FLAC file, streaming as it goes."""
    with SoundFileSink(path, config.sample_rate, subtype=subtype) as sink:
        return render(layers, config, sink)
