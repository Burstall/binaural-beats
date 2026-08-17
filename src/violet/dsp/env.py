"""
Envelopes: fades, swells, slow LFOs.

All envelopes are functions of absolute time or absolute sample index, so a
fade computed over two blocks is identical to the same fade computed over
one. The asymmetric swell — fast rise, slow exponential decay — is what
gives an ocean wave its shape; a symmetric envelope sounds like a siren
instead.

Stage 3 (fades, LFOs), stage 5 (swells).
"""

from __future__ import annotations
