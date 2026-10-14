"""
A stand-in for a real music track, for the tests that need one.

Built, not shipped: twenty seconds of something shaped like the music the
modulation is for — a bass note under the band, a chord in the middle of it,
noise over the top, and a kick on every beat at 120 bpm — written once to the
system temporary directory and reused. Seeded, so its content and therefore
every hash taken downstream of it is the same on every machine.

Real music would be a licensing question in a repository. This has no
author, and it is generated from a dozen lines that anyone can read.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
SECONDS = 20.0
BPM = 120.0

#: Bump when the recipe below changes, so a stale file is never reused.
VERSION = 1


def track_path(sample_rate: int = SAMPLE_RATE, channels: int = 2) -> Path:
    """The test track at ``sample_rate``, written on first use."""
    folder = Path(tempfile.gettempdir()) / "violet-tests"
    folder.mkdir(exist_ok=True)
    path = folder / f"track-v{VERSION}-{sample_rate}-{channels}ch.wav"
    if not path.is_file():
        sf.write(str(path), synthesise(sample_rate, channels), sample_rate, "PCM_16")
    return path


def synthesise(sample_rate: int = SAMPLE_RATE, channels: int = 2) -> np.ndarray:
    """The track itself, as ``(frames, channels)`` floats peaking near 0.6."""
    t = np.arange(int(SECONDS * sample_rate)) / sample_rate
    rng = np.random.default_rng(1729)

    bass = 0.18 * np.sin(2 * np.pi * 55.0 * t)
    chord = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (330.0, 415.3, 494.0))
    hiss = 0.01 * rng.standard_normal(t.size)

    since_beat = np.mod(t, 60.0 / BPM)
    kick = 0.25 * np.exp(-since_beat * 30.0) * np.sin(2 * np.pi * 60.0 * since_beat)

    mono = bass + chord + hiss + kick
    if channels == 1:
        return mono[:, None]
    right = mono + 0.01 * rng.standard_normal(t.size)
    return np.stack([mono, right], axis=1)
