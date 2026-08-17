"""
The ``Layer`` protocol and its implementations.

A layer renders a stereo pair for a span of absolute sample indices::

    def render(self, span: Span) -> tuple[NDArray, NDArray]: ...

Adding a sound source means writing one class, not touching the engine.
Layers that carry state across blocks (noise generators, filters, phase
accumulators) declare it explicitly so the engine can reason about whether a
render is repeatable.

Implementations: ``BinauralPair`` (stage 3), ``Pedal`` (stage 3),
``ChordBed`` (stage 4), ``Ocean`` (stage 5), ``IsochronicGate`` (roadmap).
"""

from __future__ import annotations
