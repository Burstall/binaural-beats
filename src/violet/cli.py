"""The ``violet`` command line interface."""

from __future__ import annotations

from typing import Annotated

import typer

from violet import __version__

app = typer.Typer(
    name="violet",
    help="Generate long-form binaural and ambient audio.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(*, value: bool) -> None:
    if value:
        typer.echo(f"violet {__version__}")
        raise typer.Exit


VersionFlag = Annotated[
    bool,
    typer.Option(
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version,
        is_eager=True,
    ),
]


@app.callback()
def main(*, version: VersionFlag = False) -> None:
    """Generate long-form binaural and ambient audio."""


if __name__ == "__main__":  # pragma: no cover
    app()
