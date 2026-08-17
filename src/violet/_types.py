"""Array aliases shared across the package."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: A block of audio samples, always float64. Quantisation happens once, at
#: the sink, so nothing upstream has to think about sample formats.
FloatArray = npt.NDArray[np.float64]

#: Absolute sample indices.
IntArray = npt.NDArray[np.int64]

#: What every layer returns: one array per ear, in that order.
Stereo = tuple[FloatArray, FloatArray]

#: Two pi, spelled once.
TAU = 2.0 * np.pi

__all__ = ["TAU", "FloatArray", "IntArray", "Stereo"]
