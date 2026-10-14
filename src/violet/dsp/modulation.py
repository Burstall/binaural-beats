"""
Amplitude modulation: a pulse in the loudness, in each ear on its own.

A binaural beat lives *between* the ears. It exists only once the brainstem
has compared one ear's phase with the other's, it needs headphones, and the
evidence that it does anything is thin. Amplitude modulation is the other way
to put a rate into a sound: the level itself rises and falls, the same in both
ears, so the pulse is physically present in the air and works on a speaker.

This is the technique Brain.fm describes, and the one its published study
tested: Woods et al. (2024), *Communications Biology* 7:1376. Two of the
authors work for the company, the music was the company's own, and the
benefit it found was small and concentrated in listeners who rated themselves
high on attention difficulties. It is not nothing, and it is not settled. The
README has the details and this package makes no claim beyond them. What it
does provide is the null condition: a depth of zero gives back the input
*sample for sample*, so a blinded comparison of the two is exact.

What gets modulated
-------------------
Not the whole signal. The study confined its modulation to roughly 200 Hz to
1 kHz, and there is a good reason to: modulate the bass and a kick drum starts
pumping, modulate the top and a hi-hat starts fluttering. Both are heard as
the music being broken rather than as texture.

So the signal is split. A band-pass picks out the band, and the output is::

    out = x - depth * m(t) * band(x)

where ``m`` runs from 0 to 1 and back once per cycle. Written that way rather
than as ``rest + gain * band``, it has a property worth more than any other:
with ``depth = 0`` the subtraction is of exactly zero and the output *is* the
input, bit for bit. The placebo is not an approximation of the original; it is
the original.

The band edges are soft. Near the corners the band-pass shifts phase as well
as level, so there the dip is shallower than ``depth`` and can even lift the
level very slightly. That is the price of a filter that can run block by block
with its state carried forward, and it is inaudible next to the modulation
itself.

The shape
---------
A raised cosine: full level at the start of each cycle, down to ``1 - depth``
half a cycle later, back up. Smooth, so the modulation adds no clicks of its
own, and with its maximum *on* the cycle boundary. When the rate is locked to a
tempo that boundary is the beat, so the drum hits land at full level and the
dips fall between them.

The phase comes from :class:`~violet.dsp.curves.BeatCurve`, the same closed-form
integral the binaural beat uses, so a modulation rate can move over a render
exactly as a beat rate can, and it is evaluated from absolute time with no
state at all.

Locking to the tempo
--------------------
A 16 Hz pulse laid over music at 84 bpm drifts against the beat, and the
drift is audible as a slow phasing that has nothing to do with the music. The
study aligned its modulation to the metrical grid instead. :func:`tempo_locked`
does the same: it picks the beat subdivision — duple or triple, up to 32 per
beat — that lands closest to the rate asked for, so every beat begins at the
top of a cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from violet._types import TAU
from violet.dsp.curves import BeatCurve
from violet.dsp.filters import FilterDesign

if TYPE_CHECKING:
    from violet._types import FloatArray

__all__ = [
    "SUBDIVISIONS",
    "Modulation",
    "band_gain_bound",
    "tempo_locked",
]

#: Pulses per beat that :func:`tempo_locked` may choose from. Duple and triple
#: subdivisions only: those are the grids music is written on, and an 11-tuplet
#: lands on every beat but on nothing in between.
SUBDIVISIONS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def tempo_locked(target_hz: float, bpm: float) -> float:
    """
    The rate nearest ``target_hz`` that divides a beat at ``bpm`` evenly.

    Nearest by ratio, not by hertz, for the same reason the curves glide
    exponentially: 12 Hz is as far below 16 as 21.3 is above it.

    At 120 bpm a beat is 2 Hz and 16 Hz is eight pulses per beat, exactly. At
    84 bpm a beat is 1.4 Hz, and the choice is between 12 per beat (16.8 Hz)
    and 8 (11.2 Hz); 16.8 is closer.
    """
    if target_hz <= 0.0:
        msg = f"modulation rate must be positive, got {target_hz!r}"
        raise ValueError(msg)
    if bpm <= 0.0:
        msg = f"tempo must be positive, got {bpm!r}"
        raise ValueError(msg)
    beat_hz = bpm / 60.0
    best = min(SUBDIVISIONS, key=lambda k: abs(math.log(k * beat_hz / target_hz)))
    return best * beat_hz


def band_gain_bound(design: FilterDesign, sample_rate: int) -> float:
    """
    The most a filter can amplify any signal's peak: its impulse response's L1.

    A filter's output at any sample is a weighted sum of its inputs, so the
    sum of the absolute weights bounds how far a peak can grow. For the
    fourth-order Butterworth band this package uses it comes out about 1.76,
    whatever the sample rate — a band-pass can ring past its input, and the
    bound has to allow for it.

    Two seconds of impulse response is far more than enough: at 200 Hz a
    Butterworth band has decayed below rounding in a few hundredths of one.
    """
    impulse = np.zeros(2 * sample_rate, dtype=np.float64)
    impulse[0] = 1.0
    response = signal.lfilter(
        np.asarray(design.b, dtype=np.float64),
        np.asarray(design.a, dtype=np.float64),
        impulse,
    )
    return float(np.abs(response).sum())


@dataclass(frozen=True, slots=True)
class Modulation:
    """
    What to modulate, how fast, and how hard.

    ``rate`` is a fixed number of pulses per second or a curve that moves.
    ``depth`` is how far the band dips at the bottom of each pulse: 0 is no
    modulation at all, 1 takes the band to silence. ``band`` is the range that
    gets modulated, in hertz. ``offset`` moves the top of the first cycle to
    that many seconds in, which is how a pulse is lined up with the first beat
    of a track that does not start on one.
    """

    rate: float | BeatCurve
    depth: float = 0.5
    band: tuple[float, float] = (200.0, 1000.0)
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth <= 1.0:
            msg = f"modulation depth must be between 0 and 1, got {self.depth!r}"
            raise ValueError(msg)
        low, high = self.band
        if not 0.0 < low < high:
            msg = f"modulation band must run low to high above 0, got {self.band!r}"
            raise ValueError(msg)
        if isinstance(self.rate, BeatCurve):
            if self.rate.span[0] <= 0.0:
                msg = "a modulation curve must stay above 0 Hz throughout"
                raise ValueError(msg)
        elif self.rate <= 0.0:
            msg = f"modulation rate must be positive, got {self.rate!r}"
            raise ValueError(msg)

    @property
    def active(self) -> bool:
        """Whether this does anything. A depth of zero is the null condition."""
        return self.depth > 0.0

    def describe(self) -> str:
        """One line, for the CLI."""
        rate = (
            self.rate.describe()
            if isinstance(self.rate, BeatCurve)
            else f"{self.rate:g} Hz"
        )
        low, high = self.band
        return f"{rate}, depth {self.depth:g}, band {low:g}-{high:g} Hz"

    def cycles(self, t: FloatArray) -> FloatArray:
        """Pulses completed by absolute times ``t``, counted from ``offset``."""
        if isinstance(self.rate, BeatCurve):
            start = self.rate.cycles(np.asarray([self.offset], dtype=np.float64))
            out: FloatArray = self.rate.cycles(t) - start[0]
            return out
        shifted: FloatArray = self.rate * (t - self.offset)
        return shifted

    def dip(self, t: FloatArray) -> FloatArray:
        """
        How far below full level the band sits at times ``t``: 0 up to ``depth``.

        Zero at the top of every cycle, ``depth`` at the bottom.
        """
        shape: FloatArray = 0.5 * (1.0 - np.cos(TAU * self.cycles(t)))
        out: FloatArray = self.depth * shape
        return out

    def design(self, sample_rate: int) -> FilterDesign:
        """
        The band-pass that picks out what gets modulated.

        Order 2 in scipy's band-pass terms, which is fourth order overall:
        steep enough that a bass line an octave below the band is barely
        touched, gentle enough to ring very little.
        """
        return FilterDesign.butterworth(sample_rate, self.band, "band", order=2)
