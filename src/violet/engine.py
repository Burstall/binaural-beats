"""
The block-streaming renderer.

The engine owns the loop and nothing else: it walks absolute sample indices
in blocks, asks each layer for its stereo contribution, sums them, applies
the master fade and gain, and hands the block to a sink. Length is therefore
not memory bound — a 90-minute render costs the same working set as a
5-second one.

Block boundaries are an implementation detail and must not be audible.
Layers are handed absolute indices for exactly that reason, and the engine's
own envelopes are computed from absolute indices too.

Stage 3.
"""

from __future__ import annotations
