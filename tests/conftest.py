"""
Shared test fixtures.

The prototypes in ``reference/`` are the specification this package was
refactored from, so several tests check the package against them directly
rather than against a transcription of what they were understood to do.
They are loaded by path — they are scripts, not an installed package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"

_loaded: dict[str, ModuleType] = {}


def load_reference(name: str) -> ModuleType:
    """
    Import a prototype from ``reference/`` under a private module name.

    The prototypes import each other by bare name (``from tuning import ...``),
    so the reference directory goes on ``sys.path`` first. It is appended, not
    prepended, so it can never shadow a real package.
    """
    if name in _loaded:
        return _loaded[name]

    if str(REFERENCE_DIR) not in sys.path:
        sys.path.append(str(REFERENCE_DIR))

    path = REFERENCE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_reference_{name}", path)
    if spec is None or spec.loader is None:
        msg = f"cannot load reference prototype {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _loaded[name] = module
    return module


@pytest.fixture(scope="session")
def reference_tuning() -> ModuleType:
    """The frozen ``reference/tuning.py`` prototype."""
    return load_reference("tuning")
