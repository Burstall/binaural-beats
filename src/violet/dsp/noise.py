"""
Noise generators with persistent state.

Pink and brown noise, produced by filtering white noise and carrying the
filter state forward block to block so the output is one continuous stream
rather than a sequence of independently-started ones.

Each stream owns its own random generator, seeded independently. Sharing one
generator between channels would interleave their draws, which makes the
noise realisation depend on the block size — a correctness bug, since
rendering the same configuration at two block sizes must agree.

Stage 5.
"""

from __future__ import annotations
