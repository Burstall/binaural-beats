"""
Guards on the shape of the package rather than its behaviour.

The base frequency is a parameter. It is easy to type a number into a module
while chasing a bug and never take it out again, at which point the package
quietly has an opinion about what frequency you should be listening to. This
test fails if that happens.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "violet"

# The tone-test result the project is named for, and the carrier derived from
# it. Both are configuration. Neither belongs in Python source, including in
# a default argument, a docstring or a comment — a documented default is
# still a default.
FORBIDDEN = (
    re.compile(r"83\.949"),
    re.compile(r"335\.796"),
)


def python_sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_the_package_has_sources_to_check() -> None:
    """Guard against this file passing because it looked in the wrong place."""
    assert SRC.is_dir()
    assert len(python_sources()) >= 10


def test_no_module_contains_a_literal_base_frequency() -> None:
    offences: list[str] = []
    for path in python_sources():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN:
                if pattern.search(line):
                    rel = path.relative_to(SRC.parents[1])
                    offences.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offences, "base frequency hard-coded in:\n" + "\n".join(offences)


def test_tuning_is_free_of_audio_dependencies() -> None:
    """
    The frequency maths is standard library only.

    Nothing heavier than the standard library, so it stays fast to test and
    cannot quietly drift into DSP.
    """
    text = (SRC / "tuning.py").read_text(encoding="utf-8")
    for banned in ("import numpy", "import scipy", "import soundfile", "from violet"):
        assert banned not in text, banned
