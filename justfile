# Task runner for violet.  Install with:  uv tool install rust-just
#
# Every recipe is a single `uv run ...` command, so nothing here is load
# bearing — the README documents the equivalent commands directly and they
# work without `just` installed.

set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# List the recipes.
default:
    @just --list

# Run the test suite.
test *args:
    uv run pytest {{ args }}

# Run the test suite with a coverage report.
cov:
    uv run pytest --cov --cov-report=term-missing

# Lint (and check formatting).
lint:
    uv run ruff check .
    uv run ruff format --check .

# Fix what can be fixed, then format.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Type check.
typecheck:
    uv run mypy

# Everything CI runs.
check: lint typecheck test

# Render a preset.  Example:  just render ocean "--minutes 30 --out out/o.flac"
# (the CLI lands in stage 6)
render preset="ocean" *args:
    uv run violet render {{ preset }} {{ args }}

# Print the tuning maths for a base frequency.
tune freq="83.949":
    uv run violet tune {{ freq }}
