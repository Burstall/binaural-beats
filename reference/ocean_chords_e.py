#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///

"""
Generative long-form binaural piece - ocean bed + slow chord movement in E
Brief: base 83.949 Hz | note E | violet

Structure
---------
  1. PEDAL      83.949 Hz mono drone, continuous, never changes.
                The identified frequency acts as tonal centre, so the
                chords above read as modal colour over E rather than
                as key changes.

  2. HARMONY    3-4 voices per chord, each voice its own binaural pair
                detuned by the SAME beat. This is the trick that lets the
                harmony move while the beat stays single: a voice at
                ratio r contributes r*root -/+ beat/2, so every voice
                beats at exactly `beat` no matter what chord is playing.
                Just-intonation ratios, so the voices lock instead of
                grinding against each other.

  3. OCEAN      Two decorrelated pink-noise streams shaped into wave
                events. Decorrelated noise CANNOT carry a binaural beat
                (the percept needs correlated input at both ears), so the
                ocean is texture only - the beat lives in the harmony.
                A notch at the carrier keeps the noise from masking it.

Chord movement is a weighted random walk, so an hour never repeats.

Usage
  python3 ocean_chords_e.py --beat 4 --minutes 60 --out session.wav
  python3 ocean_chords_e.py --beat 7.83 --minutes 30 --seed 12 --ocean 0.16
"""

import argparse
import numpy as np
from scipy import signal
import wave

from tuning import auto_octaves, describe

DEFAULT_BASE = 83.949   # the supplied brief; override with --base
SR = 32000

BANDS = {"delta": 2.0, "theta": 4.0, "schumann": 7.83, "alpha": 10.0}

# NOTE ON NAMING: these keys are scale DEGREES relative to whatever root
# is in use, spelled as if the root were E because that is what the
# original brief supplied. With --base set elsewhere the harmony transposes
# correctly (the ratios are all relative), but a chord printed as "C" is
# really "the major triad a minor sixth above your root". In roman numerals
# the progression is i, i7, VI, III, iv, VII - correct for any root.
#
# Just-intonation ratios relative to the root. Voices are chosen to sit
# between roughly 250 and 900 Hz - low enough to feel warm, high enough
# that the binaural percept stays strong.
CHORDS = {
    "Em":  [1.0, 6 / 5, 3 / 2, 9 / 4],          # E  G  B  F#
    "C":   [8 / 5, 2.0, 12 / 5, 3.0],           # C  E  G  B
    "G":   [6 / 5, 3 / 2, 9 / 5, 12 / 5],       # G  B  D  G
    "Am":  [4 / 3, 8 / 5, 2.0, 12 / 5],         # A  C  E  G
    "D":   [9 / 5, 9 / 4, 8 / 3, 18 / 5],       # D  F# A  D
    "Em7": [1.0, 6 / 5, 3 / 2, 9 / 5],          # E  G  B  D
}

MOVES = {
    "Em":  [("C", 3), ("G", 3), ("Am", 2), ("Em7", 2)],
    "C":   [("G", 3), ("Em", 3), ("Am", 2)],
    "G":   [("Em", 3), ("D", 2), ("C", 2), ("Em7", 2)],
    "Am":  [("Em", 3), ("C", 2), ("G", 2)],
    "D":   [("G", 3), ("Em", 2)],
    "Em7": [("C", 3), ("Am", 2), ("G", 2)],
}

XF = 16.0          # crossfade seconds between chords
DUR_MIN = 38.0     # chord length range
DUR_MAX = 62.0


def plan_chords(total, rng):
    """Weighted random walk through the progression. Returns (name, t0, t1)."""
    out, t, cur = [], 0.0, "Em"
    while t < total + DUR_MAX:
        d = rng.uniform(DUR_MIN, DUR_MAX)
        out.append((cur, t, t + d))
        opts, wts = zip(*MOVES[cur])
        cur = rng.choice(opts, p=np.array(wts, float) / sum(wts))
        t += d
    return out


def chord_gain(t, t0, t1, first, last):
    """Equal-power envelope: sin in, cos out, flat between."""
    g = np.ones_like(t)
    a, b = t0 - XF / 2, t0 + XF / 2
    if first:
        g[t < b] = 1.0
    else:
        m = (t >= a) & (t < b)
        g[m] = np.sin((t[m] - a) / XF * np.pi / 2)
        g[t < a] = 0.0
    c, d = t1 - XF / 2, t1 + XF / 2
    if not last:
        m = (t >= c) & (t < d)
        g[m] = np.cos((t[m] - c) / XF * np.pi / 2)
        g[t >= d] = 0.0
    return g


def plan_waves(total, rng):
    """Ocean swell events: (peak_time, rise, decay, amplitude)."""
    out, t = [], 4.0
    while t < total + 20:
        rise = rng.uniform(1.3, 2.8)
        decay = rng.uniform(5.0, 10.5)
        out.append((t, rise, decay, rng.uniform(0.55, 1.0)))
        t += rng.uniform(6.5, 13.0)
    return out


def wave_env(t, waves, sets):
    """Sum of asymmetric swells, modulated by a slow set cycle."""
    env = np.zeros_like(t)
    lo, hi = t[0] - 14, t[-1] + 4
    for pk, rise, decay, amp in waves:
        if pk < lo or pk > hi:
            continue
        d = t - pk
        e = np.zeros_like(t)
        up = (d >= -rise) & (d < 0)
        e[up] = np.sin((d[up] + rise) / rise * np.pi / 2) ** 2
        dn = d >= 0
        e[dn] = np.exp(-d[dn] / (decay / 2.4))
        env += amp * e
    slow = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * t / sets))
    return np.clip(env, 0, 2.2) * slow


class Ocean:
    """Decorrelated pink noise, split dark/bright, notched at the carrier."""

    def __init__(self, notch_hz, q=2.0):
        self.rng = np.random.default_rng(2027)
        # pink-ish: one-pole cascade approximating -3 dB/octave
        self.pink_b, self.pink_a = [0.049922, -0.095993, 0.050612, -0.004408], \
                                   [1.0, -2.494956, 2.017265, -0.522189]
        self.dark_b, self.dark_a = signal.butter(2, 520 / (SR / 2), "low")
        self.brt_b, self.brt_a = signal.butter(2, [900 / (SR / 2),
                                                   6500 / (SR / 2)], "band")
        self.nb, self.na = signal.iirnotch(notch_hz, q, SR)
        n = 2
        self.z = [{"p": np.zeros(3), "d": np.zeros(2),
                   "b": np.zeros(4), "n": np.zeros(2)} for _ in range(n)]

    def block(self, n):
        chans = []
        for st in self.z:
            w = self.rng.standard_normal(n) * 0.35
            p, st["p"] = signal.lfilter(self.pink_b, self.pink_a, w, zi=st["p"])
            dark, st["d"] = signal.lfilter(self.dark_b, self.dark_a, p, zi=st["d"])
            brt, st["b"] = signal.lfilter(self.brt_b, self.brt_a, p, zi=st["b"])
            chans.append((dark, brt, st))
        return chans

    def notch(self, x, st):
        y, st["n"] = signal.lfilter(self.nb, self.na, x, zi=st["n"])
        return y


def render(path, beat, seconds, ocean_level, seed,
           base=DEFAULT_BASE, block_s=10.0):
    root = base * 2.0 ** auto_octaves(base)
    rng = np.random.default_rng(seed)
    chords = plan_chords(seconds, rng)
    waves = plan_waves(seconds, rng)
    sets = rng.uniform(85.0, 125.0)
    oc = Ocean(root)

    n = int(round(SR * seconds))
    block = int(SR * block_s)
    fade = min(int(SR * 22), n // 5)
    voice = 0.070

    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)

        for start in range(0, n, block):
            stop = min(start + block, n)
            m = stop - start
            t = np.arange(start, stop, dtype=np.float64) / SR

            left = np.zeros(m)
            right = np.zeros(m)

            for i, (name, t0, t1) in enumerate(chords):
                if t1 + XF < t[0] or t0 - XF > t[-1]:
                    continue
                g = chord_gain(t, t0, t1, i == 0, i == len(chords) - 1)
                if not g.any():
                    continue
                for k, r in enumerate(CHORDS[name]):
                    f = root * r
                    # each voice breathes on its own slow cycle
                    br = 0.62 + 0.38 * (0.5 + 0.5 * np.sin(
                        2 * np.pi * t / (47.0 + 19.0 * k) + k * 1.9))
                    lv = voice * g * br
                    left += lv * np.sin(2 * np.pi * (f - beat / 2) * t)
                    right += lv * np.sin(2 * np.pi * (f + beat / 2) * t)

            ped = 0.080 * np.sin(2 * np.pi * base * t)
            left += ped
            right += ped

            env = wave_env(t, waves, sets)
            bright = np.clip(env, 0, 1.6) ** 1.6
            for ch, (dark, brt, st) in zip((0, 1), oc.block(m)):
                sig = oc.notch(dark * env + brt * 0.55 * bright, st)
                if ch == 0:
                    left += sig * ocean_level
                else:
                    right += sig * ocean_level

            idx = np.arange(start, stop)
            ev = np.ones(m)
            head = idx < fade
            tail = idx >= n - fade
            ev[head] = np.sin(idx[head] / fade * np.pi / 2) ** 2
            ev[tail] = np.sin((n - 1 - idx[tail]) / fade * np.pi / 2) ** 2

            stereo = np.stack([left * ev, right * ev], axis=1) * 1.25
            np.clip(stereo, -0.98, 0.98, out=stereo)
            w.writeframes((stereo * 32767).astype(np.int16).tobytes())

    return chords, root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--beat", default="theta")
    p.add_argument("--base", type=float, default=DEFAULT_BASE,
                   help="your identified frequency in Hz")
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--ocean", type=float, default=0.42, help="ocean bed level")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="ocean_chords.wav")
    a = p.parse_args()

    beat = BANDS.get(str(a.beat).lower()) or float(a.beat)
    ch, root = render(a.out, beat, a.minutes * 60, a.ocean, a.seed,
                      base=a.base)
    prog = " ".join(c[0] for c in ch if c[1] < a.minutes * 60)
    print(describe(a.base))
    print()
    print(f"{a.out}  beat {beat:g} Hz  root {root:.3f} Hz  "
          f"{a.minutes:g} min")
    print(f"progression: {prog}")
