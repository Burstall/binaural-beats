#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

"""
Layered binaural generator - E / 83.949 Hz brief

Why layers instead of a richer waveform
---------------------------------------
Harmonic n of a pair detuned by b beats at n*b. A sawtooth or a square
carrier therefore produces beats at b, 2b, 3b... simultaneously, which
muddies the percept. To get a fuller sound while keeping ONE clean beat,
every layer is a separate sine pair with the SAME detune b, placed at a
musically related pitch. The harmony is real; the beat stays single.

Layers
  root      335.796 Hz  (E, base x4)      binaural, full level
  octave    671.592 Hz  (E, base x8)      binaural, quiet, slow swell
  fifth     503.694 Hz  (B, root x1.5)    binaural, quieter, slower swell
  sub        83.949 Hz  (identified freq) mono drone, both ears
  air       pink noise, low-passed        mono bed, very quiet

The two upper layers breathe on slow independent LFOs so the texture
drifts instead of sitting still. Noise is generated with persistent
filter state so it stays continuous across blocks.
"""

import argparse
import numpy as np
import wave

from tuning import auto_octaves, describe

DEFAULT_BASE = 83.949   # the supplied brief; override with --base
SR = 44100

BANDS = {"delta": 2.0, "theta": 4.0, "schumann": 7.83, "alpha": 10.0}


class Pink:
    """Paul Kellet's economy pink filter, state persists across blocks."""

    def __init__(self):
        self.s = np.zeros(3)
        self.rng = np.random.default_rng(7)

    def block(self, n):
        w = self.rng.standard_normal(n)
        out = np.empty(n)
        b0, b1, b2 = self.s
        for i in range(n):
            b0 = 0.99765 * b0 + w[i] * 0.0990460
            b1 = 0.96300 * b1 + w[i] * 0.2965164
            b2 = 0.57000 * b2 + w[i] * 1.0526913
            out[i] = b0 + b1 + b2 + w[i] * 0.1848
        self.s = np.array([b0, b1, b2])
        return out * 0.06


def pair(t, freq, beat, level):
    """One binaural layer: same beat, split around freq."""
    return (level * np.sin(2 * np.pi * (freq - beat / 2) * t),
            level * np.sin(2 * np.pi * (freq + beat / 2) * t))


def render(path, beat, seconds, base=DEFAULT_BASE, root=None,
           air=True, block=SR * 10):
    if root is None:
        root = base * 2.0 ** auto_octaves(base)
    n = int(round(SR * seconds))
    fade = min(int(SR * 12), n // 4)
    pink = Pink()
    lp = 0.0  # one-pole low-pass state for the noise bed

    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)

        for start in range(0, n, block):
            stop = min(start + block, n)
            t = np.arange(start, stop, dtype=np.float64) / SR

            # slow breathing envelopes, ~70 s and ~110 s cycles
            sw_oct = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 70.0))
            sw_fif = 0.40 + 0.60 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 110.0 + 1.7))

            l, r = pair(t, root, beat, 0.22)

            lo, ro = pair(t, root * 2, beat, 0.085)
            l += lo * sw_oct
            r += ro * sw_oct

            lf, rf = pair(t, root * 1.5, beat, 0.055)
            l += lf * sw_fif
            r += rf * sw_fif

            drone = 0.085 * np.sin(2 * np.pi * base * t)
            l += drone
            r += drone

            if air:
                x = pink.block(stop - start)
                y = np.empty_like(x)
                a = 0.02                      # ~140 Hz corner, soft bed
                for i in range(len(x)):
                    lp += a * (x[i] - lp)
                    y[i] = lp
                l += y
                r += y

            idx = np.arange(start, stop)
            env = np.ones(stop - start)
            head = idx < fade
            tail = idx >= n - fade
            env[head] = np.sin(idx[head] / fade * np.pi / 2) ** 2
            env[tail] = np.sin((n - 1 - idx[tail]) / fade * np.pi / 2) ** 2
            l *= env
            r *= env

            stereo = np.stack([l, r], axis=1) * 0.82
            np.clip(stereo, -0.98, 0.98, out=stereo)
            w.writeframes((stereo * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--beat", default="theta")
    p.add_argument("--base", type=float, default=DEFAULT_BASE,
                   help="your identified frequency in Hz")
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--no-air", action="store_true", help="drop the noise bed")
    p.add_argument("--out", default="layered.wav")
    a = p.parse_args()

    beat = BANDS.get(str(a.beat).lower()) or float(a.beat)
    root = a.base * 2.0 ** auto_octaves(a.base)
    render(a.out, beat, a.minutes * 60, base=a.base, air=not a.no_air)
    print(describe(a.base))
    print()
    print(f"{a.out}  beat {beat:g} Hz  root {root:.3f} Hz  "
          f"{a.minutes:g} min")
