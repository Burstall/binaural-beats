"""``violet library``: gather, tag and publish music for the phone app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from violet.archive import Pool, UrllibHttp
from violet.library import (
    DEFAULT_POOLS,
    Library,
    Wrangler,
    add_file,
    classify,
    edit_tags,
    fetch,
    publish,
    reanalyse,
)

library_app = typer.Typer(
    name="library",
    help="Gather, tag and publish music for the phone app.",
    no_args_is_help=True,
)


def echo(message: str) -> None:
    """
    ``typer.echo``, but never killed by a title the console cannot spell.

    Track titles come from all over the world and a Windows console, or a
    file it is redirected to, is often cp1252. A character it cannot encode
    becomes ``?`` rather than a crash halfway through a publish.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    typer.echo(message.encode(encoding, "replace").decode(encoding))


RootOption = Annotated[
    Path,
    typer.Option("--root", help="The library folder.", file_okay=False),
]


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg="red", err=True)
    return typer.Exit(code=1)


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


@library_app.command("fetch")
def fetch_command(
    *,
    hours: Annotated[
        float, typer.Option("--hours", help="Total hours to gather.")
    ] = 40.0,
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-q",
            help="Your own Archive search; replaces the built-in pools.",
        ),
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", min=1, max=8, help="Downloads at once.")
    ] = 4,
    root: RootOption = Path("library"),
) -> None:
    """
    Gather Creative Commons music from the Internet Archive.

    Without --query, fills the three built-in pools (steady beats, ambient,
    other calm instrumentals) in proportion to --hours. Interrupt it any time;
    running it again carries on.
    """
    library = Library.open(root)
    if query:
        pools: tuple[Pool, ...] = (Pool("custom", hours, tuple(query)),)
        scale_hours = 1.0
    else:
        pools = DEFAULT_POOLS
        scale_hours = hours / sum(p.hours for p in DEFAULT_POOLS)
    report = fetch(
        library,
        UrllibHttp(),
        pools,
        scale_hours=scale_hours,
        workers=workers,
        log=echo,
    )
    for name, added in report.added.items():
        echo(f"{name:8s} +{added} tracks, {_hours(report.seconds[name])} in the pool")
    echo(f"looked at {report.items} items, turned down {report.rejected} tracks")
    echo(f"library: {len(library.entries)} tracks, {_hours(library.seconds())}")


@library_app.command("add")
def add_command(  # noqa: PLR0913 - every one is part of the credit
    files: Annotated[
        list[Path],
        typer.Argument(
            help="MP3, FLAC, WAV, Ogg or AIFF files.", exists=True, dir_okay=False
        ),
    ],
    *,
    artist: Annotated[str, typer.Option("--artist", help="Who made it.")],
    source_url: Annotated[
        str,
        typer.Option(
            "--source-url", help="Where it came from, e.g. its Bandcamp page."
        ),
    ],
    title: Annotated[
        str | None,
        typer.Option("--title", help="Title (one file only; default: file name)."),
    ] = None,
    album: Annotated[str | None, typer.Option("--album")] = None,
    license_name: Annotated[
        str, typer.Option("--license", help="Licence, as it should be credited.")
    ] = ("Personal use (purchased)"),
    license_url: Annotated[
        str | None, typer.Option("--license-url", help="Default: the source URL.")
    ] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", "-t", help="A tag; repeat for more.")
    ] = None,
    root: RootOption = Path("library"),
) -> None:
    """Add music you have, with its credit."""
    if title is not None and len(files) > 1:
        msg = "--title names one track; leave it out to use each file's name"
        raise _fail(msg)
    library = Library.open(root)
    for path in files:
        try:
            entry = add_file(
                library,
                path,
                title=title,
                artist=artist,
                album=album,
                license_name=license_name,
                license_url=license_url or source_url,
                source_url=source_url,
                tags=tag or (),
            )
        except ValueError as error:
            raise _fail(str(error)) from error
        library.save()
        bpm = f"{entry.bpm:g} bpm" if entry.bpm is not None else "no steady beat"
        echo(f"{entry.id}  {entry.title}  ({entry.duration / 60:.1f} min, {bpm})")


@library_app.command("list")
def list_command(
    *,
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag", "-t", help="Only tracks with this tag; repeat for all of several."
        ),
    ] = None,
    root: RootOption = Path("library"),
) -> None:
    """List the library, with the analyser's tags filled in."""
    library = Library.open(root)
    classify(library)
    wanted = set(tag or ())
    shown = [e for e in library.entries.values() if wanted <= set(e.tags)]
    for entry in sorted(shown, key=lambda e: (e.artist.lower(), e.title.lower())):
        bpm = f"{entry.bpm:6.1f}" if entry.bpm is not None else "     -"
        tags = " ".join(sorted(entry.tags))
        timing = f"{entry.duration / 60:5.1f}m {bpm} {entry.beat_strength:4.2f}"
        echo(f"{entry.id}  {timing}  {entry.artist} - {entry.title}  [{tags}]")
    total = sum(e.duration for e in shown)
    echo(f"{len(shown)} tracks, {_hours(total)}")


@library_app.command(
    "tag",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": False},
)
def tag_command(
    track: Annotated[
        str, typer.Argument(help="A track id, or enough of its start to be unique.")
    ],
    changes: Annotated[list[str], typer.Argument(help="+tag to add, -tag to remove.")],
    *,
    root: RootOption = Path("library"),
) -> None:
    """Add (+tag) and remove (-tag) tags by hand."""
    library = Library.open(root)
    try:
        entry = library.find(track)
    except KeyError as error:
        raise _fail(str(error.args[0])) from error
    add = [c[1:] for c in changes if c.startswith("+")]
    remove = [c[1:] for c in changes if c.startswith("-")]
    stray = [c for c in changes if not c.startswith(("+", "-"))]
    if stray:
        msg = f"say +tag or -tag, not {stray[0]!r}"
        raise _fail(msg)
    try:
        edit_tags(entry, add, remove)
    except ValueError as error:
        raise _fail(str(error)) from error
    library.save()
    queued = " (queued for the next publish)" if entry.in_d1 else ""
    echo(f"{entry.id}  {' '.join(sorted(entry.tags))}{queued}")


@library_app.command("analyse")
def analyse_command(*, root: RootOption = Path("library")) -> None:
    """Measure every track again with the current analyser."""
    library = Library.open(root)
    count = reanalyse(library, log=echo)
    echo(f"measured {count} tracks; publish to carry the new numbers to the app")


@library_app.command("publish")
def publish_command(
    *,
    web: Annotated[
        Path,
        typer.Option(
            "--web",
            help="The web/ folder with wrangler.jsonc.",
            file_okay=False,
            exists=True,
        ),
    ] = Path("web"),
    root: RootOption = Path("library"),
) -> None:
    """Upload new tracks to R2 and bring D1 up to date. Safe to run again."""
    library = Library.open(root)
    try:
        report = publish(library, Wrangler(web), log=echo)
    except RuntimeError as error:
        raise _fail(str(error)) from error
    echo(
        f"uploaded {report.uploaded} files; {report.rows} rows in D1 "
        f"({report.new_rows} new, {report.tag_edits} tag edits applied)",
    )
