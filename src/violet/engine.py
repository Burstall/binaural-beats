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

No ffmpeg
---------
Files are written by libsndfile through ``soundfile``, block by block, as WAV
or FLAC. Nothing is buffered in memory and nothing is shelled out to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from violet.dsp.env import fade_envelope, fade_length
from violet.layers import Span, StatefulLayer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from violet._types import FloatArray
    from violet.layers import Layer

__all__ = [
    "ArraySink",
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


def headroom_gain(layers: Iterable[Layer], ceiling: float) -> float:
    """
    Master gain that keeps the summed analytic peak at or under ``ceiling``.

    Attenuates or does nothing; never boosts.
    """
    total = sum(layer.peak for layer in layers)
    if total <= 0.0:
        return 1.0
    return min(1.0, ceiling / total)


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

    for layer in layers:
        if isinstance(layer, StatefulLayer):
            layer.reset()

    gain = (
        config.gain
        if config.gain is not None
        else headroom_gain(layers, config.ceiling)
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
        left = np.zeros(span.n, dtype=np.float64)
        right = np.zeros(span.n, dtype=np.float64)

        for layer in layers:
            layer_left, layer_right = layer.render(span)
            left += layer_left
            right += layer_right

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
