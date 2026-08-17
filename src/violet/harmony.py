"""
Chords as ratio sets, and the walk that moves between them.

Chords are defined as tuples of just-intonation ratios relative to the root,
keyed by roman numeral so the labels stay true for any root. Movement is a
weighted transition graph walked by a seeded generator, which makes a
progression reproducible from its seed and non-repeating over an hour.

Crossfades between chords are equal-power (``sin``/``cos``), so the summed
power stays flat through a transition instead of dipping in the middle.

Stage 4.
"""

from __future__ import annotations
