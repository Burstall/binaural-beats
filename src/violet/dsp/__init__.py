"""
Signal generation and processing primitives.

Every primitive in this subpackage is either a pure function of absolute
time or an object that carries its state forward explicitly. There is no
hidden global state and no per-block reset, because both produce clicks at
block boundaries.
"""

from __future__ import annotations
