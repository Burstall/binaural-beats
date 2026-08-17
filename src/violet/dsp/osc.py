"""
Oscillators: sine pairs, drones, phase accumulators.

Fixed-frequency tones are computed from absolute time, ``t = arange(start,
stop) / sr``, which keeps phase continuous across blocks for free.

Time-varying frequency cannot be phased as ``2*pi*f*t``; it needs an
integrated phase, ``phase += 2*pi*cumsum(f)/sr``, with the accumulator
carried across blocks. That is what the phase accumulator here is for
(roadmap item 2, beat automation curves).

Stage 3.
"""

from __future__ import annotations
