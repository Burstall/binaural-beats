"""
Named configurations, as frozen dataclasses and as TOML.

The built-in presets reproduce the three prototypes in ``reference/``. User
presets are TOML files read with ``tomllib`` from the standard library, so
adding one never means editing Python.

Stage 6.
"""

from __future__ import annotations
