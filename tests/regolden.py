"""
Regenerate the golden hashes.

Run this only when you have changed the output *on purpose*::

    uv run python tests/regolden.py

Then read the diff. If a preset you did not touch has moved, something is
wrong, and the hash is telling you so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_presets import (
    GOLDEN,
    LIBRARY,
    NAMES,
    digest,
    render_seconds,
)


def main() -> None:
    """Render every preset and write the hashes."""
    hashes = {name: digest(render_seconds(LIBRARY[name])) for name in NAMES}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    for name, value in hashes.items():
        print(f"{name:<12} {value}")


if __name__ == "__main__":
    main()
