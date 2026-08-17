"""Stage 1: the skeleton is importable and the entry point runs."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

import violet
from violet.cli import app

MODULES = [
    "violet.tuning",
    "violet.harmony",
    "violet.layers",
    "violet.engine",
    "violet.presets",
    "violet.cli",
    "violet.dsp",
    "violet.dsp.osc",
    "violet.dsp.noise",
    "violet.dsp.filters",
    "violet.dsp.env",
]


@pytest.mark.parametrize("name", MODULES)
def test_architecture_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_version_is_populated() -> None:
    assert violet.__version__ != "0.0.0.dev0"


def test_cli_reports_its_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert violet.__version__ in result.output


def test_cli_help_without_arguments() -> None:
    result = CliRunner().invoke(app, [])
    assert "binaural" in result.output.lower()
