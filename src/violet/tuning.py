"""
Frequency arithmetic: notes, cents, tuning references, carriers, colour.

Pure functions over floats, no audio and no numpy. Everything the rest of
the package needs to turn a base frequency into musical and visual
quantities lives here:

* nearest equal-tempered note, and the deviation in cents
* the tuning reference implied by treating a frequency as exactly in tune
* ``auto_octaves`` — the whole-octave shift that lands a base frequency in
  the usable binaural carrier window, preserving the note
* just-intonation ratio tables, so :mod:`violet.harmony` can spell chords as
  exact ratios rather than equal-tempered approximations
* the octave-arithmetic mapping from frequency to wavelength and an
  approximate sRGB hex — a curiosity, not physics

Stage 2.
"""

from __future__ import annotations
