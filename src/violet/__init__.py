"""
violet — long-form binaural and ambient audio generation.

The package is built around two ideas:

* A ``Layer`` renders a stereo pair for an *absolute* span of sample
  indices, so phase and filter state stay continuous however the render is
  chopped into blocks.
* The base frequency is a parameter. No module in this package holds a
  literal base frequency; it enters as configuration and flows outward into
  the pedal drone, the carrier, the chord root and the ocean's notch.

Evidence for brainwave entrainment is mixed and this package makes no health
claims. See the README.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("violet")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
