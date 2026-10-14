"""
Track analysis: loudness against the standard, tempo against known grids.

The loudness tests use the BS.1770-4 filter table and the EBU Tech 3341
minimum-requirement test signals; the rhythm tests use synthetic tracks whose
tempo and first beat are known exactly, and tracks with no beat at all —
noise, a held chord, a few slow chord changes — which must say so.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

import music
from violet import analysis
from violet.analysis import (
    NO_BEAT,
    TARGET_LUFS,
    analyse,
    estimate_tempo,
    flux_bands,
    gated_loudness,
    k_weighting,
    playback_gain_db,
)

if TYPE_CHECKING:
    from pathlib import Path

    from violet._types import FloatArray


def write(path: Path, data: FloatArray, sample_rate: int) -> Path:
    sf.write(str(path), data, sample_rate, "FLOAT")
    return path


def tone(dbfs: float, seconds: float, sample_rate: int = 48000) -> FloatArray:
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    wave: FloatArray = 10.0 ** (dbfs / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    return wave


def stereo(x: FloatArray) -> FloatArray:
    return np.stack([x, x], axis=1)


def beat_track(
    bpm: float,
    *,
    seconds: float = 40.0,
    start: float = 0.0,
    hats: bool = True,
    sample_rate: int = 44100,
) -> FloatArray:
    """Kick on every beat, hats on the off-beats, a held chord underneath."""
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    period = 60.0 / bpm
    since = np.mod(t - start, period)
    since[t < start] = 10.0
    kick = 0.4 * np.exp(-since * 25.0) * np.sin(2.0 * np.pi * 55.0 * since)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(t.size)
    offbeat = np.mod(t - start - period / 2.0, period)
    offbeat[t < start + period / 2.0] = 10.0
    hat = 0.1 * np.exp(-offbeat * 80.0) * noise if hats else 0.0
    chord = sum(0.05 * np.sin(2.0 * np.pi * f * t) for f in (261.63, 329.63, 392.0))
    return stereo(kick + hat + chord)


def circular_gap(a: float, b: float, period: float) -> float:
    d = (a - b) % period
    return min(d, period - d)


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------


def test_k_weighting_reproduces_the_bs1770_table_at_48k() -> None:
    """ITU-R BS.1770-4, Tables 1 and 2: the coefficients at 48 kHz."""
    shelf, highpass = k_weighting(48000)
    np.testing.assert_allclose(
        shelf,
        [
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            1.0,
            -1.69065929318241,
            0.73248077421585,
        ],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        highpass,
        [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
        atol=1e-9,
    )


@pytest.mark.parametrize("sample_rate", [48000, 44100])
@pytest.mark.parametrize("dbfs", [-23.0, -33.0])
def test_ebu_3341_steady_sine(tmp_path: Path, sample_rate: int, dbfs: float) -> None:
    """EBU Tech 3341 cases 1 and 2: a stereo 1 kHz sine reads as its own level."""
    path = write(tmp_path / "s.wav", stereo(tone(dbfs, 20.0, sample_rate)), sample_rate)
    loudness = analyse(path).loudness
    assert loudness is not None
    assert loudness == pytest.approx(dbfs, abs=0.1)


def test_ebu_3341_relative_gate(tmp_path: Path) -> None:
    """
    EBU Tech 3341 case 3: -36, -23, -36 dBFS reads -23.

    The quiet ends are 13 LU down, past the relative gate, so they do not
    pull the figure down. At the test's own 10/60/10 s: shortened, the blocks
    straddling each step weigh enough to show.
    """
    x = np.concatenate([tone(-36.0, 10.0), tone(-23.0, 60.0), tone(-36.0, 10.0)])
    loudness = analyse(write(tmp_path / "c3.wav", stereo(x), 48000)).loudness
    assert loudness == pytest.approx(-23.0, abs=0.1)


def test_ebu_3341_absolute_gate(tmp_path: Path) -> None:
    """EBU Tech 3341 case 4: -72 dBFS lead-in and tail do not count."""
    parts = [(-72.0, 10.0), (-36.0, 10.0), (-23.0, 20.0), (-36.0, 10.0), (-72.0, 10.0)]
    x = np.concatenate([tone(db, s) for db, s in parts])
    loudness = analyse(write(tmp_path / "c4.wav", stereo(x), 48000)).loudness
    assert loudness == pytest.approx(-23.0, abs=0.1)


def test_mono_is_measured_as_dual_mono(tmp_path: Path) -> None:
    """A mono file plays in both ears, so it measures like the stereo copy."""
    x = tone(-20.0, 10.0)
    mono = analyse(write(tmp_path / "m.wav", x, 48000)).loudness
    both = analyse(write(tmp_path / "s.wav", stereo(x), 48000)).loudness
    assert mono == pytest.approx(both, abs=1e-6)


def test_silence_has_no_loudness_and_no_gain(tmp_path: Path) -> None:
    measured = analyse(write(tmp_path / "z.wav", np.zeros((48000 * 3, 2)), 48000))
    assert measured.loudness is None
    assert measured.gain_db == 0.0
    assert measured.bpm is None
    assert measured.beat_strength == 0.0


def test_gating_needs_one_whole_block() -> None:
    assert gated_loudness(np.ones((3, 2))) is None
    assert gated_loudness(np.zeros((10, 2))) is None


def test_gain_reaches_the_target_when_the_peak_allows() -> None:
    assert playback_gain_db(-10.0, 0.9) == pytest.approx(TARGET_LUFS + 10.0)


def test_gain_is_capped_by_the_peak() -> None:
    """-30 LUFS wants +14 dB, but a 0.5 peak only has 5 dB to -1 dBFS."""
    gain = playback_gain_db(-30.0, 0.5)
    assert gain == pytest.approx(-1.0 - 20.0 * math.log10(0.5), abs=0.01)
    assert gain < 14.0


def test_more_than_two_channels_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "q.wav", np.zeros((4800, 4)), 48000)
    with pytest.raises(ValueError, match="4 channels"):
        analyse(path)


# ---------------------------------------------------------------------------
# rhythm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000])
def test_the_test_track_is_120_bpm_on_the_grid(sample_rate: int) -> None:
    measured = analyse(music.track_path(sample_rate))
    assert measured.bpm == 120.0
    assert measured.beat_offset is not None
    assert circular_gap(measured.beat_offset, 0.0, 0.5) < 0.01
    assert measured.beat_strength >= 0.6


@pytest.mark.parametrize("shift", [0.123, 0.31])
def test_a_later_start_moves_the_offset(tmp_path: Path, shift: float) -> None:
    data, rate = sf.read(str(music.track_path()), always_2d=True)
    padded = np.concatenate([np.zeros((round(shift * rate), 2)), data])
    measured = analyse(write(tmp_path / "late.wav", padded, rate))
    assert measured.bpm == 120.0
    assert measured.beat_offset is not None
    assert circular_gap(measured.beat_offset, shift, 0.5) < 0.01


@pytest.mark.parametrize(
    ("bpm", "expected"),
    [
        (84.0, 84.0),  # whole bpm, snapped
        (93.5, 93.5),  # half bpm, snapped
        (72.0, 72.0),  # slow, hats at double time do not double it
        (170.0, 85.0),  # folded into 60-140
    ],
)
def test_tempos(tmp_path: Path, bpm: float, expected: float) -> None:
    measured = analyse(write(tmp_path / "b.wav", beat_track(bpm), 44100))
    assert measured.bpm == expected
    assert measured.beat_strength >= 0.5


def test_offset_of_a_synthetic_groove(tmp_path: Path) -> None:
    measured = analyse(write(tmp_path / "g.wav", beat_track(84.0, start=0.2), 44100))
    assert measured.beat_offset is not None
    assert circular_gap(measured.beat_offset, 0.2, 60.0 / 84.0) < 0.015


def test_an_unrounded_tempo_is_left_alone(tmp_path: Path) -> None:
    """A tempo 0.2 bpm off a whole number is real, not measurement error."""
    measured = analyse(write(tmp_path / "u.wav", beat_track(88.2, seconds=60.0), 44100))
    assert measured.bpm == pytest.approx(88.2, abs=0.03)


def no_beat_signals() -> dict[str, FloatArray]:
    sr = 44100
    t = np.arange(40 * sr) / sr
    rng = np.random.default_rng(3)
    swell = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.05 * t)
    # Whole-hertz partials beat inside a band and ripple with a period of
    # exactly one second: the trap a scale-blind periodicity measure falls in.
    pad = sum(0.1 * np.sin(2.0 * np.pi * f * t) for f in (220.0, 277.0, 330.0)) * swell
    chords = np.zeros_like(t)
    for i, start in enumerate(np.arange(0.0, 40.0, 7.3)):
        span = (t >= start) & (t < start + 7.3)
        f = (220.0, 247.0, 262.0, 294.0)[i % 4]
        decay = np.exp(-(t[span] - start) * 0.3)
        chords[span] = 0.2 * np.sin(2.0 * np.pi * f * t[span]) * decay
    return {
        "noise": 0.1 * rng.standard_normal((t.size, 2)),
        "pad": stereo(pad),
        "chords": stereo(chords),
    }


@pytest.mark.parametrize("name", ["noise", "pad", "chords"])
def test_no_beat_means_no_tempo(tmp_path: Path, name: str) -> None:
    measured = analyse(write(tmp_path / f"{name}.wav", no_beat_signals()[name], 44100))
    assert measured.beat_strength < NO_BEAT
    assert measured.bpm is None
    assert measured.beat_offset is None


def test_too_short_to_judge() -> None:
    assert estimate_tempo(np.ones(500)) == analysis.Tempo(None, None, 0.0)
    assert estimate_tempo(np.zeros(5000)).bpm is None


def test_energy_does_not_follow_level(tmp_path: Path) -> None:
    """Energy is measured at target loudness: turning a track down changes nothing."""
    data = beat_track(90.0)
    loud = analyse(write(tmp_path / "loud.wav", data, 44100))
    quiet = analyse(write(tmp_path / "quiet.wav", data * 0.25, 44100))
    assert quiet.energy == pytest.approx(loud.energy, rel=0.01)
    assert quiet.brightness == pytest.approx(loud.brightness, rel=0.01)
    assert quiet.gain_db == pytest.approx(loud.gain_db + 12.04, abs=0.05)


def test_busy_is_more_energetic_than_still(tmp_path: Path) -> None:
    groove = analyse(write(tmp_path / "g.wav", beat_track(90.0), 44100))
    pad = analyse(write(tmp_path / "p.wav", no_beat_signals()["pad"], 44100))
    assert groove.energy > 3.0 * pad.energy


def test_hats_are_brighter_than_none(tmp_path: Path) -> None:
    with_hats = analyse(write(tmp_path / "h.wav", beat_track(90.0), 44100))
    without = analyse(write(tmp_path / "n.wav", beat_track(90.0, hats=False), 44100))
    assert with_hats.brightness > without.brightness


def test_read_size_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """READ_SECONDS is memory only: the stream's block size cannot show."""
    path = music.track_path()
    before = analyse(path)
    monkeypatch.setattr(analysis, "READ_SECONDS", 3.7)
    after = analyse(path)
    assert after == before


def test_flux_bands_partition_the_bins() -> None:
    bands = flux_bands(25.0, 321)
    assert bands.shape[1] == 321
    assert (bands.sum(axis=1) >= 1).all()
    assert (bands.sum(axis=0) <= 1).all()
    assert bands[:, 0].sum() == 0  # DC is below 30 Hz


def test_a_slow_coarse_tempo_is_folded_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A coarse pick under 60 bpm is doubled into range, not trusted."""
    monkeypatch.setattr(analysis, "SEARCH_BPM", (40.0, 59.0))
    spikes = np.zeros(40 * analysis.FRAME_RATE)
    spikes[:: round(analysis.FRAME_RATE * 60.0 / 104.0)] = 30.0
    # Onsets are a few frames wide in real envelopes; a one-frame spike falls
    # between the coarse grid's lags.
    tempo = estimate_tempo(gaussian_filter1d(spikes, 2.0))
    assert tempo.bpm is not None
    assert 100.0 <= tempo.bpm <= 108.0


def test_a_file_shorter_than_one_frame(tmp_path: Path) -> None:
    measured = analyse(write(tmp_path / "blip.wav", np.zeros((400, 2)), 44100))
    assert measured.duration == pytest.approx(400 / 44100, abs=1e-3)
    assert measured.energy == 0.0
    assert measured.brightness == 0.0
