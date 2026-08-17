"""
Typed wrappers over the scipy IIR filters this package uses.

Butterworth low-pass, high-pass and band-pass, plus the notch. Each wrapper
holds its own ``zi`` state and applies ``scipy.signal.lfilter`` block by
block, one state per channel per filter. Dropping that state resets the
filter and clicks.

Stage 5.
"""

from __future__ import annotations
