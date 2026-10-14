"""
What a music track is like, measured: loudness, tempo, beat and feel.

Everything the phone app needs to know about a track before it plays it, and
everything the library needs to sort it into modes. Two streamed passes over
the decoded file, a block at a time, so an hour-long mix costs the same memory
as a three-minute song:

1. **Loudness and peak.** Integrated loudness per ITU-R BS.1770-4: the
   K-weighting filter, mean square over 400 ms blocks overlapping by 75%, an
   absolute gate at -70 LUFS and a relative gate 10 LU below the absolutely
   gated loudness. From that and the sample peak, the playback gain that
   brings the track to the library's target without pushing its peak past
   the ceiling.
2. **Rhythm and feel.** A spectral-flux onset envelope at 100 frames a
   second, measured on the track *after* it has been brought to the target
   loudness, so that a quiet track and a loud one with the same drums read
   the same. Tempo, the phase of the beat grid, how clearly there is a beat at
   all, how busy the track is, and how bright.

The thresholds that turn these numbers into tags live in
:mod:`violet.library`; this module only measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

if TYPE_CHECKING:
    from pathlib import Path

    from violet._types import FloatArray

# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------

#: Every track is brought to this integrated loudness on playback. -16 LUFS is
#: the usual target for listening on phones: loud enough to need no extra
#: volume, quiet enough that most masters only ever come *down*.
TARGET_LUFS = -16.0

#: ...but never so far that a sample peak passes this.
CEILING_DBFS = -1.0

#: BS.1770 gating: blocks quieter than this are silence and never count.
ABSOLUTE_GATE_LUFS = -70.0

#: And blocks this far below the absolutely gated loudness are pauses.
RELATIVE_GATE_LU = -10.0

#: Read size for both passes. Memory only; results do not depend on it.
READ_SECONDS = 10.0

# The K-weighting filter's two stages, as analogue prototypes. These are the
# parameters that reproduce the coefficients BS.1770-4 tabulates for 48 kHz
# (Table 1 and Table 2 of the recommendation) and give the same filter at any
# other rate: Brecht De Man's derivation, as used by libebur128.
_SHELF_HZ = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_HIGHPASS_HZ = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773


def k_weighting(sample_rate: int) -> FloatArray:
    """
    BS.1770's K-weighting filter at ``sample_rate``, as second-order sections.

    A high shelf of about +4 dB above 1.7 kHz, modelling the head, then a
    high-pass at 38 Hz, the "revised low-frequency B-curve". Returned in
    :func:`scipy.signal.sosfilt` form, shelf first.
    """
    k = math.tan(math.pi * _SHELF_HZ / sample_rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / _SHELF_Q + k * k
    shelf = [
        (vh + vb * k / _SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / _SHELF_Q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / _SHELF_Q + k * k) / a0,
    ]
    k = math.tan(math.pi * _HIGHPASS_HZ / sample_rate)
    a0 = 1.0 + k / _HIGHPASS_Q + k * k
    highpass = [
        1.0,
        -2.0,
        1.0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / _HIGHPASS_Q + k * k) / a0,
    ]
    return np.array([shelf, highpass])


def gated_loudness(mean_squares: FloatArray) -> float | None:
    """
    Integrated loudness from per-100 ms K-weighted mean squares.

    ``mean_squares`` is ``(sub_blocks, channels)``, already channel-weighted.
    Each 400 ms gating block is four consecutive sub-blocks, so consecutive
    blocks overlap by 75% exactly as BS.1770-4 specifies. Returns ``None``
    when nothing survives the absolute gate: silence has no loudness.
    """
    subs = mean_squares.sum(axis=1)
    if subs.size < 4:  # noqa: PLR2004 - four 100 ms sub-blocks make one block
        return None
    blocks = (subs[:-3] + subs[1:-2] + subs[2:-1] + subs[3:]) / 4.0
    with np.errstate(divide="ignore"):
        levels = -0.691 + 10.0 * np.log10(blocks)
    audible = blocks[levels > ABSOLUTE_GATE_LUFS]
    if audible.size == 0:
        return None
    relative = -0.691 + 10.0 * math.log10(float(audible.mean())) + RELATIVE_GATE_LU
    kept = blocks[(levels > ABSOLUTE_GATE_LUFS) & (levels > relative)]
    return -0.691 + 10.0 * math.log10(float(kept.mean()))


def playback_gain_db(loudness: float | None, peak: float) -> float:
    """
    The gain that brings a track to :data:`TARGET_LUFS`, capped by its peak.

    A quiet track with a hard transient is only raised until the transient
    reaches :data:`CEILING_DBFS`; it ends up a little under target rather
    than clipped. Silence gets no gain at all.
    """
    if loudness is None:
        return 0.0
    gain = TARGET_LUFS - loudness
    if peak > 0.0:
        gain = min(gain, CEILING_DBFS - 20.0 * math.log10(peak))
    return round(gain, 2)


# ---------------------------------------------------------------------------
# rhythm
# ---------------------------------------------------------------------------

#: Onset-envelope frames per second. The FFT is four hops long, so every bin
#: is 25 Hz wide whatever the sample rate and the measures below mean the
#: same thing for a 32 kHz file as for a 48 kHz one.
FRAME_RATE = 100

#: Spectral flux is summed over bins up to here: above it, MP3 encoders at
#: different bitrates disagree about whether there is anything at all.
FLUX_TOP_HZ = 8000.0

#: Brightness is a centroid over bins up to here.
CENTROID_TOP_HZ = 11000.0

#: Magnitudes are compressed with ``log1p(COMPRESSION * amplitude)`` before
#: differencing, so a hi-hat counts for something next to a kick.
COMPRESSION = 1000.0

#: Flux is taken over this many log-spaced bands between 30 Hz and
#: :data:`FLUX_TOP_HZ`, not over raw bins. Hiss and vinyl crackle make every
#: bin flicker from frame to frame; summed over hundreds of bins that flicker
#: buries the drums. Pooled into bands, it averages out, and a kick still owns
#: its band.
FLUX_BANDS = 24

#: Below this beat strength a track has no steady beat and no tempo.
NO_BEAT = 0.2

#: Bands entirely below this are "the kick" for placing the beat grid.
KICK_TOP_HZ = 200.0

#: How lined-up the kick's onsets must be (phase coherence at the beat,
#: 0..1) for the grid to be placed by the kick rather than by everything.
KICK_COHERENCE = 0.3

#: Tempo search range for the first, coarse estimate.
SEARCH_BPM = (50.0, 200.0)

#: Where an octave-ambiguous tempo is folded to. Lo-fi hats run at double
#: time over a 75 bpm groove; the groove is the tempo.
FOLD_BPM = (60.0, 140.0)

#: A refined tempo this close to an integer or half-integer bpm is snapped to
#: it: music made in a sequencer is at a round tempo, and the residue is
#: measurement error that would otherwise slide the pulse off the beat.
SNAP_BPM = 0.05

#: Beat strength is judged over windows this long, so a live band drifting
#: slowly off the grid still reads as having a beat.
STRENGTH_WINDOW_SECONDS = 20.0

#: The size of onsets (standard deviation of the smoothed, detrended onset
#: envelope, at :data:`TARGET_LUFS`) at and above which a repeating pattern
#: counts in full. Measured: real music 0.5-3, a stable synthetic pad's
#: ripple about 0.06.
AUDIBLE_ONSETS = 0.4


@dataclass(frozen=True)
class Tempo:
    """What :func:`estimate_tempo` found: a tempo, or ``None`` and why not."""

    bpm: float | None
    offset: float | None
    strength: float


def onset_envelope(flux: FloatArray) -> FloatArray:
    """
    Spectral flux with its slow trend removed, ready for periodicity.

    The flux of a sustained passage is not zero (there is always some
    change) and it rises and falls with the arrangement; subtracting a
    one-second moving average and keeping what pokes above it leaves the
    onsets.
    """
    trend: FloatArray = uniform_filter1d(flux, size=FRAME_RATE + 1, mode="nearest")
    return np.maximum(flux - trend, 0.0)


def _comb_scores(envelope: FloatArray, bpms: FloatArray) -> FloatArray:
    """Autocorrelation at the first four multiples of each tempo's period."""
    x = envelope - envelope.mean()
    n = x.size
    spectrum = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    ac = ac / ac[0]
    lags = 60.0 * FRAME_RATE / bpms
    multiples = lags[:, None] * np.arange(1, 5)[None, :]
    values = np.interp(multiples, np.arange(n), ac, right=0.0)
    scores: FloatArray = values.mean(axis=1)
    return scores


def _harmonic_power(
    envelope: FloatArray, times: FloatArray, bpms: FloatArray
) -> FloatArray:
    """Periodogram power summed over the first four harmonics of each tempo."""
    power = np.zeros_like(bpms)
    for k in range(1, 5):
        hz = k * bpms / 60.0
        phases = np.exp(-2j * np.pi * hz[:, None] * times[None, :])
        power += np.abs(phases @ envelope) ** 2
    return power


def _strength(envelope: FloatArray, bpm: float) -> float:
    """
    How clearly, and how audibly, the envelope repeats at ``bpm``.

    Judged in windows, then averaged. In each window the onset envelope has
    its quarter-second trend taken out and is smoothed by 20 ms (so swing
    and a human drummer's timing still line up), and its autocorrelation is
    read at one beat, two beats and a bar, a couple of frames either side,
    taking the best. Loops repeat at the bar even when the hats are swung
    off the beat, which is why phase coherence at the beat frequency — tried
    first — scored obvious lo-fi at 0.3 and a handful of sparse chord
    changes at 0.8.

    Autocorrelation is blind to scale, so a stable drone's tiny ripple (its
    partials beating inside a band) can repeat almost perfectly and read as a
    beat. So each window's score is also scaled by its onsets' size, at the
    target loudness, up to :data:`AUDIBLE_ONSETS`: a beat has to be heard.
    """
    window = int(STRENGTH_WINDOW_SECONDS * FRAME_RATE)
    step = window // 2
    period = 60.0 * FRAME_RATE / bpm
    detrended = envelope - uniform_filter1d(envelope, size=25, mode="nearest")
    smooth: FloatArray = gaussian_filter1d(detrended, 2.0)
    scores: list[float] = []
    for start in range(0, max(1, smooth.size - window + 1), step):
        x = smooth[start : start + window]
        x = x - x.mean()
        n = x.size
        variance = float(x @ x) / n
        if variance <= 1e-12:  # noqa: PLR2004 - silence has no beat
            continue
        best = 0.0
        for beats in (1, 2, 4):
            for nudge in (-2, -1, 0, 1, 2):
                lag = round(beats * period) + nudge
                if 0 < lag < n // 2:
                    r = float(x[:-lag] @ x[lag:]) / (n - lag) / variance
                    best = max(best, r)
        audible = min(1.0, math.sqrt(variance) / AUDIBLE_ONSETS)
        scores.append(min(1.0, best) * audible)
    return float(np.mean(scores)) if scores else 0.0


def estimate_tempo(envelope: FloatArray, kick: FloatArray | None = None) -> Tempo:
    """
    Tempo, beat phase and beat strength from an onset envelope.

    1. **Coarse.** Autocorrelation, scored at the first four multiples of
       each candidate period between 50 and 200 bpm and weighted by a
       one-octave log-normal prior around 100 bpm, which settles the
       otherwise exact tie between a tempo and its half.
    2. **Folded** into 60-140 bpm by octaves.
    3. **Refined** to a hundredth of a bpm on the periodogram of the whole
       envelope, its first four harmonics summed, and **snapped** to a whole
       or half bpm when within :data:`SNAP_BPM` of one.
    4. **Phase.** The first beat is where the envelope's component at the
       beat frequency peaks: ``offset = -arg(E(f)) / 2 pi f``, wrapped into
       one beat. Taken from ``kick`` — the onsets below
       :data:`KICK_TOP_HZ` — when it is given and lines up with the beat at
       all clearly. Hats are broadband and outweigh a kick in a summed
       envelope, and a groove with hats on the off-beat would otherwise put
       the "first beat" half a beat late.

    A track whose beat strength is under :data:`NO_BEAT` gets no tempo and no
    offset — a guessed grid is worse than none, because the pulse would then
    land on the music's own onsets at random.
    """
    n = envelope.size
    if n < 10 * FRAME_RATE or envelope.sum() <= 0.0:
        return Tempo(None, None, 0.0)
    times = np.arange(n) / FRAME_RATE

    grid = np.arange(SEARCH_BPM[0], SEARCH_BPM[1] + 0.25, 0.5)
    prior = np.exp(-0.5 * np.log2(grid / 100.0) ** 2)
    bpm = float(grid[np.argmax(np.maximum(_comb_scores(envelope, grid), 0.0) * prior)])
    while bpm > FOLD_BPM[1] and bpm / 2.0 >= FOLD_BPM[0]:
        bpm /= 2.0
    while bpm < FOLD_BPM[0]:
        bpm *= 2.0

    for span, step in ((0.03 * bpm, 0.05), (0.1, 0.005)):
        fine = np.arange(bpm - span, bpm + span + step / 2, step)
        bpm = float(fine[np.argmax(_harmonic_power(envelope, times, fine))])
    snapped = round(bpm * 2.0) / 2.0
    if abs(bpm - snapped) <= SNAP_BPM:
        bpm = snapped

    strength = _strength(envelope, bpm)
    if strength < NO_BEAT:
        return Tempo(None, None, round(strength, 3))
    hz = bpm / 60.0
    rotation = np.exp(-2j * np.pi * hz * times)
    component = rotation @ envelope
    if kick is not None and kick.sum() > 0.0:
        low = rotation @ kick
        if abs(low) / kick.sum() >= KICK_COHERENCE:
            component = low
    offset = (-float(np.angle(component)) / (2.0 * np.pi * hz)) % (1.0 / hz)
    return Tempo(round(bpm, 2), round(offset, 3), round(strength, 3))


# ---------------------------------------------------------------------------
# the whole track
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """
    Everything measured about one track.

    ``energy`` is the mean spectral flux of the loudness-normalised track: how
    much is *happening*, not how loud it is. It has no unit worth naming and
    only means something next to other tracks' values, which is how the
    library uses it (terciles). ``brightness`` is the spectral centroid in Hz,
    weighted by frame energy so silences do not drag it down.
    """

    duration: float
    sample_rate: int
    channels: int
    loudness: float | None
    peak: float
    gain_db: float
    bpm: float | None
    beat_offset: float | None
    beat_strength: float
    energy: float
    brightness: float


@dataclass
class _Levels:
    sample_rate: int
    channels: int
    frames: int
    peak: float
    loudness: float | None


def _levels(path: Path) -> _Levels:
    """Pass one: duration, sample peak and BS.1770 integrated loudness."""
    with sf.SoundFile(str(path)) as f:
        sr, channels = int(f.samplerate), int(f.channels)
        if channels > 2:  # noqa: PLR2004
            msg = f"{path.name} has {channels} channels; only 1 or 2 are supported"
            raise ValueError(msg)
        sos = k_weighting(sr)
        zi = np.zeros((sos.shape[0], 2, channels))
        sub = round(sr / 10)
        frames, peak = 0, 0.0
        squares: list[FloatArray] = []
        for block in f.blocks(
            blocksize=sub * int(READ_SECONDS * 10), dtype="float64", always_2d=True
        ):
            frames += len(block)
            peak = max(peak, float(np.abs(block).max(initial=0.0)))
            weighted, zi = signal.sosfilt(sos, block, axis=0, zi=zi)
            whole = len(weighted) // sub
            squares.append(
                (weighted[: whole * sub] ** 2)
                .reshape(whole, sub, channels)
                .mean(axis=1)
            )
    mean_squares = np.concatenate(squares) if squares else np.zeros((0, channels))
    if channels == 1:
        # Played to both ears, so measured as both: dual mono, +3 dB on the
        # single-channel figure. Otherwise a mono track would be turned up 3 dB
        # past every stereo one.
        mean_squares = mean_squares * 2.0
    return _Levels(sr, channels, frames, peak, gated_loudness(mean_squares))


@dataclass
class _Rhythm:
    flux: FloatArray
    kick: FloatArray
    energy: float
    brightness: float


def flux_bands(bin_hz: float, bins: int) -> FloatArray:
    """
    The ``(bands, bins)`` 0/1 matrix pooling FFT bins into log-spaced bands.

    Edges are spaced evenly in log frequency from 30 Hz to
    :data:`FLUX_TOP_HZ`; each bin goes to the band its centre falls in. At the
    bottom a band can be narrower than a bin and catch nothing; those rows
    are dropped, so every band returned holds at least one bin.
    """
    edges = np.geomspace(30.0, FLUX_TOP_HZ, FLUX_BANDS + 1)
    centres = np.arange(bins) * bin_hz
    which = np.searchsorted(edges, centres, side="right") - 1
    matrix = np.zeros((FLUX_BANDS, bins))
    inside = (which >= 0) & (which < FLUX_BANDS)
    matrix[which[inside], np.arange(bins)[inside]] = 1.0
    return matrix[matrix.sum(axis=1) > 0.0]


def _rhythm(path: Path, gain: float) -> _Rhythm:
    """Pass two: the onset envelope, energy and brightness, at target loudness."""
    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate)
        hop = round(sr / FRAME_RATE)
        n_fft = 4 * hop
        window = signal.get_window("hann", n_fft)
        scale = gain * 2.0 / window.sum()
        bin_hz = sr / n_fft
        flux_bins = int(FLUX_TOP_HZ / bin_hz) + 1
        bands = flux_bands(bin_hz, flux_bins)
        tops = (bands * (np.arange(flux_bins) * bin_hz)).max(axis=1)
        low = tops < KICK_TOP_HZ
        centroid_bins = min(n_fft // 2 + 1, int(CENTROID_TOP_HZ / bin_hz) + 1)
        freqs = np.arange(centroid_bins) * bin_hz
        # Half a frame of silence in front, so frame j is centred on sample j*hop.
        carry = np.zeros(n_fft // 2)
        previous: FloatArray | None = None
        flux: list[FloatArray] = []
        kick: list[FloatArray] = []
        weighted_centroid, total_weight = 0.0, 0.0
        for block in f.blocks(
            blocksize=int(READ_SECONDS * sr), dtype="float64", always_2d=True
        ):
            carry = np.concatenate([carry, block.mean(axis=1)])
            count = (carry.size - n_fft) // hop + 1
            if count <= 0:
                continue
            index = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
            amplitude = np.abs(np.fft.rfft(carry[index] * window, axis=1)) * scale
            carry = carry[count * hop :]
            pooled = np.sqrt((amplitude[:, :flux_bins] ** 2) @ bands.T)
            compressed = np.log1p(COMPRESSION * pooled)
            before = compressed[:1] if previous is None else previous
            rises = np.maximum(
                np.diff(np.concatenate([before, compressed]), axis=0), 0.0
            )
            flux.append(rises.sum(axis=1))
            kick.append(rises[:, low].sum(axis=1))
            previous = compressed[-1:]
            band = amplitude[:, :centroid_bins]
            weight = band.sum(axis=1)
            weighted_centroid += float((band @ freqs).sum())
            total_weight += float(weight.sum())
    envelope = np.concatenate(flux) if flux else np.zeros(0)
    kicks = np.concatenate(kick) if kick else np.zeros(0)
    energy = float(envelope.mean()) if envelope.size else 0.0
    brightness = weighted_centroid / total_weight if total_weight > 0.0 else 0.0
    return _Rhythm(envelope, kicks, energy, brightness)


def analyse(path: Path) -> Analysis:
    """
    Measure one audio file: anything soundfile can decode, mono or stereo.

    Raises :class:`ValueError` for more than two channels, and whatever
    soundfile raises for a file it cannot read.
    """
    levels = _levels(path)
    gain_db = playback_gain_db(levels.loudness, levels.peak)
    # Features are measured at target loudness, uncapped: only the level is
    # being normalised here, not protected.
    feature_gain = (
        1.0
        if levels.loudness is None
        else 10.0 ** ((TARGET_LUFS - levels.loudness) / 20.0)
    )
    rhythm = _rhythm(path, feature_gain)
    tempo = estimate_tempo(onset_envelope(rhythm.flux), onset_envelope(rhythm.kick))
    return Analysis(
        duration=round(levels.frames / levels.sample_rate, 3),
        sample_rate=levels.sample_rate,
        channels=levels.channels,
        loudness=None if levels.loudness is None else round(levels.loudness, 2),
        peak=round(levels.peak, 4),
        gain_db=gain_db,
        bpm=tempo.bpm,
        beat_offset=tempo.offset,
        beat_strength=tempo.strength,
        energy=round(rhythm.energy, 4),
        brightness=round(rhythm.brightness, 1),
    )
