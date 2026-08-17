"""
Spectrum helpers for the beat-recovery tests.

Recovering a beat frequency means measuring two peaks a few hertz apart and
subtracting, so the measurement has to be better than the bin spacing. A
5-second block at 44.1 kHz has 0.2 Hz bins, and a 4 Hz beat has to come back
as 4 Hz and not 4.2.

Two things get us there. A Hann window, so a tone that does not sit exactly
on a bin centre leaks into its neighbours in a predictable shape rather than
smearing across the whole spectrum. And parabolic interpolation over the log
magnitudes of the three bins around each peak, which recovers the true peak
position of that shape to a small fraction of a bin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from violet._types import FloatArray


def spectrum(x: FloatArray, sample_rate: int) -> tuple[FloatArray, FloatArray]:
    """Hann-windowed magnitude spectrum and its frequency axis."""
    windowed = x * np.hanning(len(x))
    magnitude = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    return freqs, magnitude


def refine(magnitude: FloatArray, bin_index: int, bin_hz: float) -> float:
    """Sub-bin peak position, by parabolic interpolation on log magnitudes."""
    if bin_index <= 0 or bin_index >= len(magnitude) - 1:
        return bin_index * bin_hz
    a, b, c = (np.log(magnitude[bin_index + k] + 1e-300) for k in (-1, 0, 1))
    denominator = a - 2.0 * b + c
    offset = 0.0 if denominator == 0.0 else 0.5 * (a - c) / denominator
    return (bin_index + offset) * bin_hz


def peak_frequencies(
    x: FloatArray,
    sample_rate: int,
    count: int,
    min_separation_hz: float = 1.0,
    floor_ratio: float = 1e-3,
) -> list[float]:
    """
    The ``count`` strongest spectral peaks, strongest first, refined sub-bin.

    Peaks closer together than ``min_separation_hz`` are treated as one, so a
    tone's leakage into neighbouring bins cannot be mistaken for a second
    tone. Peaks weaker than ``floor_ratio`` of the strongest are discarded,
    which is what stops noise floor wobble from being counted as a voice.
    """
    freqs, magnitude = spectrum(x, sample_rate)
    bin_hz = float(freqs[1] - freqs[0])
    separation_bins = max(1, round(min_separation_hz / bin_hz))

    order = np.argsort(magnitude)[::-1]
    threshold = magnitude[order[0]] * floor_ratio if len(order) else 0.0

    chosen: list[int] = []
    for index in order:
        if magnitude[index] < threshold:
            break
        if any(abs(int(index) - taken) < separation_bins for taken in chosen):
            continue
        chosen.append(int(index))
        if len(chosen) == count:
            break

    return [refine(magnitude, index, bin_hz) for index in chosen]


def matched_pairs(
    left: FloatArray,
    right: FloatArray,
    sample_rate: int,
    voices: int,
    min_separation_hz: float = 1.0,
) -> list[tuple[float, float]]:
    """
    Pair up the peaks of the two channels by frequency, low to high.

    One pair per tonal voice: the left ear's tone and the right ear's tone
    for the same voice, whose difference should be the beat.
    """
    lefts = sorted(peak_frequencies(left, sample_rate, voices, min_separation_hz))
    rights = sorted(peak_frequencies(right, sample_rate, voices, min_separation_hz))
    if len(lefts) != voices or len(rights) != voices:
        msg = (
            f"expected {voices} peaks per channel, found {len(lefts)} left and "
            f"{len(rights)} right"
        )
        raise AssertionError(msg)
    return list(zip(lefts, rights, strict=True))
