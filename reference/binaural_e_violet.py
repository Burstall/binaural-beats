#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

"""
Binaural beat generator
Brief: base frequency 83.949 Hz | note E (A=448.2 ref) | colour violet (406 nm)

Design notes
------------
83.949 Hz is too low to serve as the binaural CARRIER: the binaural-beat
percept is generated in the superior olivary complex from interaural phase
differences, and it weakens sharply below roughly 200 Hz. Most headphones
also roll off hard down there.

So this generator does two things at once:

  1. BINAURAL PAIR on the same note two octaves up (335.796 Hz = E),
     split as carrier +/- beat/2 between the ears. This is what produces
     the perceived pulse.
  2. SUB DRONE at the literal 83.949 Hz, identical in both ears, sitting
     underneath as a monaural root so the identified frequency is
     physically present in the mix.

Usage
-----
  python3 binaural_e_violet.py --beat 4 --minutes 20 --out theta.wav
  python3 binaural_e_violet.py --beat 10 --minutes 15 --loop --out alpha_loop.wav
  python3 binaural_e_violet.py --carrier 167.898 --beat 2 --minutes 45 --out sleep.wav
"""

import argparse
import numpy as np
import wave

from tuning import auto_octaves, describe

DEFAULT_BASE = 83.949   # the supplied brief; override with --base
SR = 44100

BEAT_BANDS = {
    "delta": 2.0,      # deep rest / sleep onset
    "theta": 4.0,      # meditative, hypnagogic
    "alpha": 10.0,     # relaxed alert, calm focus
    "schumann": 7.83,  # alpha/theta border
    "beta": 16.0,      # active concentration
}


def render(path, carrier, beat, seconds, drone_level, beat_level, loop,
           base=DEFAULT_BASE, block=SR * 10):
    """Stream the session to disk in blocks so length is not memory bound."""
    if loop:
        # land on a whole number of beat cycles so the file butt-joins
        # onto itself without a click
        seconds = round(seconds * beat) / beat

    n = int(round(SR * seconds))
    fade = 0 if loop else min(int(SR * 8), n // 4)   # 8 s fade in / out

    fl = carrier - beat / 2
    fr = carrier + beat / 2

    # headroom check up front, done on the analytic peak not a scan
    peak = 2 * beat_level + drone_level
    gain = 0.89 / peak if peak > 0.89 else 1.0

    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)

        for start in range(0, n, block):
            stop = min(start + block, n)
            # absolute sample index keeps phase continuous across blocks
            t = np.arange(start, stop, dtype=np.float64) / SR

            left = beat_level * np.sin(2 * np.pi * fl * t)
            right = beat_level * np.sin(2 * np.pi * fr * t)

            if drone_level > 0:
                drone = drone_level * np.sin(2 * np.pi * base * t)
                left += drone
                right += drone

            if fade:
                idx = np.arange(start, stop)
                env = np.ones(stop - start)
                head = idx < fade
                tail = idx >= n - fade
                env[head] = np.sin(idx[head] / fade * np.pi / 2) ** 2
                env[tail] = np.sin((n - 1 - idx[tail]) / fade * np.pi / 2) ** 2
                left *= env
                right *= env

            stereo = np.stack([left, right], axis=1) * gain
            w.writeframes((stereo * 32767).astype(np.int16).tobytes())

    return seconds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--beat", default="theta",
                   help="beat frequency in Hz, or a band name: "
                        + ", ".join(BEAT_BANDS))
    p.add_argument("--base", type=float, default=DEFAULT_BASE,
                   help="your identified frequency in Hz")
    p.add_argument("--carrier", type=float, default=None,
                   help="binaural carrier; default is --base octave-shifted "
                        "into the 250-520 Hz window, same note")
    p.add_argument("--octaves", type=int, default=None,
                   help="force the octave shift instead of auto")
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--drone", type=float, default=0.09,
                   help="level of the base-frequency sub drone, 0 to disable")
    p.add_argument("--level", type=float, default=0.30,
                   help="level of each binaural tone")
    p.add_argument("--loop", action="store_true",
                   help="no fades, snap length for seamless looping")
    p.add_argument("--out", default="binaural.wav")
    a = p.parse_args()

    beat = BEAT_BANDS.get(str(a.beat).lower(), None)
    if beat is None:
        beat = float(a.beat)

    shift = a.octaves if a.octaves is not None else auto_octaves(a.base)
    carrier = a.carrier if a.carrier is not None else a.base * 2.0 ** shift

    secs = render(a.out, carrier, beat, a.minutes * 60,
                  a.drone, a.level, a.loop, base=a.base)

    print(describe(a.base))
    print()
    print(f"{a.out}")
    print(f"  left      {carrier - beat/2:.3f} Hz")
    print(f"  right     {carrier + beat/2:.3f} Hz")
    print(f"  beat      {beat:g} Hz")
    print(f"  sub drone {a.base:g} Hz (both ears)" if a.drone > 0
          else "  no drone")
    print(f"  length    {secs/60:.2f} min")
