"""The ``violet`` command line interface."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from violet import __version__, tuning
from violet.dsp.curves import BeatCurve
from violet.dsp.modulation import Modulation, tempo_locked
from violet.engine import render_to_file
from violet.layers import ChordBed, Modulated, Ocean, Track
from violet.presets import Preset, load_library, resolve_beat
from violet.trial import ARMS, Trial

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from violet.layers import Layer

app = typer.Typer(
    name="violet",
    help="Generate long-form binaural and ambient audio.",
    no_args_is_help=True,
    add_completion=False,
)

trial_app = typer.Typer(
    name="trial",
    help="Run a blinded trial: is the beat doing anything for you?",
    no_args_is_help=True,
)
app.add_typer(trial_app)

echo = typer.echo


def _version(*, value: bool) -> None:
    if value:
        echo(f"violet {__version__}")
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


# ---------------------------------------------------------------------------
# shared options
# ---------------------------------------------------------------------------

PresetsOption = Annotated[
    Path | None,
    typer.Option(
        "--presets",
        help="A TOML file of your own presets, overriding the built-ins by name.",
        exists=True,
        dir_okay=False,
    ),
]


MusicOption = Annotated[
    Path | None,
    typer.Option(
        "--music",
        help="An audio file to play under the preset (fills focus-music's slot).",
        exists=True,
        dir_okay=False,
    ),
]
HzOption = Annotated[
    str | None,
    typer.Option("--hz", help="Modulation rate in Hz, or alpha/beta/..."),
]
BpmOption = Annotated[
    float | None,
    typer.Option("--bpm", help="Lock the modulation to this tempo."),
]
DepthOption = Annotated[
    float | None,
    typer.Option("--depth", help="Modulation depth, 0 (none) to 1 (full)."),
]
OffsetOption = Annotated[
    float | None,
    typer.Option("--offset", help="Seconds to the track's first beat."),
]


def _modulation(
    current: Modulation | None,
    hz: str | None,
    bpm: float | None,
    depth: float | None,
    offset: float | None,
) -> Modulation | None:
    """
    The preset's modulation with the command line's changes, or ``None``.

    ``None`` means nothing was asked for and the preset stands as it is. Asking
    for any of it on a preset without a modulation starts from 16 Hz at a
    depth of 0.4, the focus-music defaults.
    """
    if hz is None and bpm is None and depth is None and offset is None:
        return None
    base = current or Modulation(rate=resolve_beat("beta"), depth=0.4)
    rate: float | BeatCurve = base.rate
    if hz is not None:
        rate = resolve_beat(hz)
    if bpm is not None:
        if isinstance(rate, BeatCurve):
            msg = "--bpm cannot lock a modulation rate that moves"
            raise ValueError(msg)
        rate = tempo_locked(rate, bpm)
    return Modulation(
        rate=rate,
        depth=base.depth if depth is None else depth,
        band=base.band,
        offset=base.offset if offset is None else offset,
    )


def _library(paths: Path | None) -> object:
    return load_library(paths) if paths else load_library()


def _fail(message: str) -> typer.Exit:
    echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


@app.command("presets")
def list_presets(presets: PresetsOption = None) -> None:
    """List the available presets."""
    library = _library(presets)
    width = max(len(preset.name) for preset in library)  # type: ignore[attr-defined]
    for preset in library:  # type: ignore[attr-defined]
        echo(f"  {preset.name:<{width}}  {preset.description}")


@app.command()
def show(
    name: Annotated[str, typer.Argument(help="Preset name.")],
    presets: PresetsOption = None,
) -> None:
    """Show what a preset would render, without rendering it."""
    try:
        preset = _library(presets)[name]  # type: ignore[index]
    except KeyError as error:
        raise _fail(str(error).strip("\"'")) from None
    _describe(preset)


@app.command()
def tune(
    frequency: Annotated[float, typer.Argument(help="A frequency in Hz.")],
    a4: Annotated[
        float, typer.Option("--a4", help="Reference for naming notes.")
    ] = tuning.A4_STANDARD,
) -> None:
    """Show the tuning and colour arithmetic for a frequency."""
    try:
        echo(tuning.describe(frequency, a4))
    except ValueError as error:
        raise _fail(str(error)) from None


@app.command()
def render(  # noqa: PLR0913 - every one of these is a real knob
    name: Annotated[str, typer.Argument(help="Preset name.")] = "ocean",
    *,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output file; .wav or .flac."),
    ] = None,
    minutes: Annotated[
        float | None, typer.Option("--minutes", "-m", help="Length in minutes.")
    ] = None,
    beat: Annotated[
        str | None,
        typer.Option("--beat", "-b", help="Beat in Hz, or delta/theta/alpha/..."),
    ] = None,
    base: Annotated[
        float | None,
        typer.Option("--base", help="Your base frequency in Hz."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for the progression.")
    ] = None,
    loop: Annotated[
        bool | None,
        typer.Option("--loop/--no-loop", help="Close the render into a loop."),
    ] = None,
    block: Annotated[
        float | None,
        typer.Option("--block", help="Block size in seconds. Affects memory only."),
    ] = None,
    subtype: Annotated[
        str | None,
        typer.Option("--subtype", help="Sample format, e.g. PCM_16, PCM_24."),
    ] = None,
    music: MusicOption = None,
    hz: HzOption = None,
    bpm: BpmOption = None,
    depth: DepthOption = None,
    offset: OffsetOption = None,
    presets: PresetsOption = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Render a preset to a WAV or FLAC file."""
    try:
        preset: Preset = _library(presets)[name]  # type: ignore[index]
    except KeyError as error:
        raise _fail(str(error).strip("\"'")) from None

    try:
        preset = preset.with_overrides(
            minutes=minutes,
            beat=None if beat is None else resolve_beat(beat),
            base_hz=base,
            seed=seed,
            loop=loop,
            block_seconds=block,
            modulation=_modulation(preset.modulation, hz, bpm, depth, offset),
            music=music,
        )
        layers = preset.build_layers()
    except (ValueError, FileNotFoundError) as error:
        raise _fail(str(error)) from None

    destination = out or Path(f"{preset.name}.flac")
    _render_and_report(preset, layers, destination, subtype=subtype, quiet=quiet)


@app.command()
def modulate(  # noqa: PLR0913 - every one of these is a real knob
    music: Annotated[
        Path,
        typer.Argument(help="Audio file to modulate.", exists=True, dir_okay=False),
    ],
    *,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output file; .wav or .flac."),
    ] = None,
    hz: HzOption = None,
    bpm: BpmOption = None,
    depth: DepthOption = None,
    offset: OffsetOption = None,
    subtype: Annotated[
        str | None,
        typer.Option("--subtype", help="Sample format, e.g. PCM_16, PCM_24."),
    ] = None,
    presets: PresetsOption = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """
    Modulate one music file, start to finish, into another.

    The focus-music preset run over exactly the length of the file, with no
    fade of its own: the music already starts and ends the way it does.
    """
    try:
        preset: Preset = _library(presets)["focus-music"]  # type: ignore[index]
    except KeyError as error:
        raise _fail(str(error).strip("\"'")) from None

    try:
        preset = preset.with_overrides(
            music=music,
            modulation=_modulation(preset.modulation, hz, bpm, depth, offset),
        )
        track = next(_tracks(preset.build_layers()))
        preset = replace(
            preset.with_overrides(minutes=track.seconds / 60.0), fade_seconds=0.0
        )
        layers = preset.build_layers()
    except (StopIteration, ValueError, FileNotFoundError) as error:
        message = str(error) or "the focus-music preset has no track layer"
        raise _fail(message) from None

    destination = out or music.with_name(f"{music.stem}.focus.flac")
    if destination.resolve() == music.resolve():
        message = "refusing to write over the input; give a different --out"
        raise _fail(message)
    _render_and_report(preset, layers, destination, subtype=subtype, quiet=quiet)


def _tracks(layers: Iterable[Layer]) -> Iterator[Track]:
    """Every track in a mix, including inside a modulated group."""
    for layer in layers:
        if isinstance(layer, Track):
            yield layer
        elif isinstance(layer, Modulated):
            yield from _tracks(layer.layers)


def _render_and_report(
    preset: Preset,
    layers: list[Layer],
    destination: Path,
    *,
    subtype: str | None,
    quiet: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        _describe(preset, layers)
        if preset.modulation is not None and preset.gain is None:
            echo("\nmeasuring the modulated mix's peak ...")

    try:
        config = preset.settled_config(layers)
    except ValueError as error:
        raise _fail(str(error)) from None

    if not quiet:
        echo(f"rendering to {destination} ...")

    started = time.perf_counter()
    try:
        result = render_to_file(layers, config, destination, subtype=subtype)
    except ValueError as error:
        raise _fail(str(error)) from None
    elapsed = time.perf_counter() - started

    if quiet:
        return

    for note in result.notes:
        echo(f"  note      {note}")
    echo(
        f"  peak      {result.peak:.3f} ({result.peak_dbfs:+.1f} dBFS), "
        f"{result.clipped} clipped"
    )
    if result.loop_crossfade_frames:
        echo(f"  loop fold {result.loop_crossfade_frames / result.sample_rate:.0f} s")
    echo(
        f"  wrote     {destination} — {result.seconds / 60:.2f} min in "
        f"{elapsed:.1f} s ({result.seconds / max(elapsed, 1e-9):.0f}x realtime)"
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _describe(preset: Preset, layers: list[Layer] | None = None) -> None:
    echo(f"{preset.name} — {preset.description}\n")
    echo(tuning.describe(preset.base_hz))

    beat = preset.beat
    carrier = preset.carrier_hz
    echo("")
    if isinstance(beat, BeatCurve):
        echo(f"beat curve          {beat.describe()}")
        low, high = beat.span
        echo(
            f"  ears sweep        "
            f"{carrier - high / 2:.3f}-{carrier - low / 2:.3f} / "
            f"{carrier + low / 2:.3f}-{carrier + high / 2:.3f} Hz"
        )
    elif beat > 0:
        echo(f"beat                {beat:g} Hz")
        left, right = carrier - beat / 2, carrier + beat / 2
        echo(f"  ears at           {left:.3f} / {right:.3f} Hz")
    else:
        echo(f"beat                {beat:g} Hz")
        echo("  ears identical    no interaural difference, so no beat at all")
    echo(f"length              {preset.minutes:g} min at {preset.sample_rate} Hz")
    echo(f"seed                {preset.seed}")
    if preset.loop:
        echo(f"loop                yes, {preset.loop_crossfade:g} s fold if needed")
    if preset.modulation is not None:
        echo(f"modulation          {preset.modulation.describe()}")

    echo("\nlayers")
    try:
        built = preset.build_layers() if layers is None else layers
    except (ValueError, FileNotFoundError) as error:
        echo(f"  ({error})")
        return
    for layer in built:
        echo(f"  {_layer_line(layer, preset)}")


def _layer_line(layer: object, preset: Preset) -> str:
    match layer:
        case Modulated():
            inner = "\n".join(
                "    " + _layer_line(one, preset).replace("\n", "\n  ")
                for one in layer.layers
            )
            return f"modulated {layer.modulation.describe()}\n{inner}"
        case Track():
            again = ", repeating" if layer.repeat else ""
            return (
                f"track     level {layer.level:.3f}, {layer.path.name} "
                f"({layer.seconds / 60.0:.2f} min at {layer.sample_rate} Hz{again})"
            )
        case ChordBed():
            numerals = " ".join(
                event.chord.numeral
                for event in layer.events
                if event.start < preset.seconds
            )
            names = " ".join(layer.events[0].chord.note_labels(layer.root))
            return (
                f"chords    level {layer.level:.3f}, opens on "
                f"{layer.events[0].chord.numeral} ({names})\n"
                f"            {numerals}"
            )
        case Ocean():
            return (
                f"ocean     level {layer.level:.3f}, {len(layer.swells)} waves, "
                f"notched at {layer.notch_hz:.1f} Hz"
            )
        case _:
            described = getattr(layer, "carrier", None) or getattr(layer, "freq", None)
            kind = type(layer).__name__.lower()
            where = f" at {described:.3f} Hz" if described else ""
            return f"{kind:<9} level {layer.level:.3f}{where}"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------

TrialsDir = Annotated[
    Path,
    typer.Option("--dir", help="Where trials live."),
]


def _open_trial(directory: Path, name: str) -> Trial:
    try:
        return Trial.load(directory / name)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise _fail(str(error)) from None


@trial_app.command("start")
def trial_start(  # noqa: PLR0913 - the render knobs, same as `render`
    name: Annotated[str, typer.Argument(help="A name for this trial.")],
    *,
    preset_name: Annotated[
        str, typer.Option("--preset", help="Preset to test.")
    ] = "ocean",
    minutes: Annotated[float | None, typer.Option("--minutes", "-m")] = None,
    beat: Annotated[str | None, typer.Option("--beat", "-b")] = None,
    base: Annotated[float | None, typer.Option("--base")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    scale: Annotated[
        str, typer.Option("--scale", help="Rating scale, as low-high.")
    ] = "1-5",
    music: MusicOption = None,
    hz: HzOption = None,
    bpm: BpmOption = None,
    depth: DepthOption = None,
    offset: OffsetOption = None,
    presets: PresetsOption = None,
    directory: TrialsDir = Path("trials"),
) -> None:
    """Render both arms of a new trial and seal which is which."""
    try:
        preset: Preset = _library(presets)[preset_name]  # type: ignore[index]
    except KeyError as error:
        raise _fail(str(error).strip("\"'")) from None

    try:
        low, high = (int(part) for part in scale.split("-", 1))
    except ValueError:
        message = f"--scale must look like 1-5, got {scale!r}"
        raise _fail(message) from None

    try:
        preset = preset.with_overrides(
            minutes=minutes,
            beat=None if beat is None else resolve_beat(beat),
            base_hz=base,
            seed=seed,
            modulation=_modulation(preset.modulation, hz, bpm, depth, offset),
            music=music,
        )
        echo(f"rendering two arms of {preset.minutes:g} min each. This takes a while.")
        trial = Trial.create(preset, directory / name, name, scale=(low, high))
    except (ValueError, FileNotFoundError) as error:
        raise _fail(str(error)) from None

    echo(f"\ntrial {trial.name} — {trial.preset}, {trial.treatment_label} vs none\n")
    for arm in ARMS:
        echo(f"  {trial.arm_path(arm)}")
    echo(
        f"\nOne of those has {trial.treatment_noun} and the other does not. They "
        f"are\notherwise the same render: same seed, same chords, same waves, same\n"
        f"noise, same music.\n"
        f"\nUncompressed on purpose — FLAC would give the answer away by file size.\n"
        f"\nRun `violet trial next {trial.name}` to be told which to play. Rate each\n"
        f"session {trial.scale[0]} to {trial.scale[1]} with `violet trial log`.\n"
    )


@trial_app.command("next")
def trial_next(
    name: Annotated[str, typer.Argument(help="Trial name.")],
    *,
    directory: TrialsDir = Path("trials"),
) -> None:
    """Say which arm to play next, keeping the two balanced."""
    trial = _open_trial(directory, name)
    arm = trial.next_arm()
    echo(f"play {arm}: {trial.arm_path(arm)}")
    echo(f'afterwards: violet trial log {name} {arm} --state N --note "..."')


@trial_app.command("log")
def trial_log(
    name: Annotated[str, typer.Argument(help="Trial name.")],
    arm: Annotated[str, typer.Argument(help="Which arm you listened to.")],
    *,
    state: Annotated[int, typer.Option("--state", "-s", help="Your rating.")],
    note: Annotated[str, typer.Option("--note", "-n")] = "",
    directory: TrialsDir = Path("trials"),
) -> None:
    """Record one listening session."""
    trial = _open_trial(directory, name)
    try:
        session = trial.log(arm.upper(), state, note)
    except ValueError as error:
        raise _fail(str(error)) from None

    counts = trial.counts()
    echo(f"logged {session.arm} = {session.state}")
    echo("sessions so far: " + ", ".join(f"{a} {counts[a]}" for a in ARMS))
    if session.after_reveal:
        echo("(after unblinding, so it will not count towards the comparison)")


@trial_app.command("status")
def trial_status(
    name: Annotated[str, typer.Argument(help="Trial name.")],
    *,
    directory: TrialsDir = Path("trials"),
) -> None:
    """Show progress without giving anything away."""
    trial = _open_trial(directory, name)
    counts = trial.counts()
    echo(f"trial {trial.name} — {trial.preset}, {trial.treatment_label} vs none")
    echo(f"  created   {trial.created}")
    echo("  sessions  " + ", ".join(f"{a} {counts[a]}" for a in ARMS))
    echo(f"  revealed  {trial.revealed_at or 'no'}")

    for session in trial.sessions:
        marker = " (post-reveal)" if session.after_reveal else ""
        note = f"  {session.note}" if session.note else ""
        echo(f"    {session.at}  {session.arm} = {session.state}{note}{marker}")

    if not trial.revealed:
        echo(
            "\nRatings per arm are shown; which arm carries the beat is not, and "
            "will not be until you reveal."
        )


@trial_app.command("reveal")
def trial_reveal(
    name: Annotated[str, typer.Argument(help="Trial name.")],
    *,
    i_am_done: Annotated[
        bool, typer.Option("--i-am-done", help="Yes, end the blind for good.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Reveal below the session floor anyway.")
    ] = False,
    directory: TrialsDir = Path("trials"),
) -> None:
    """Unblind, and report what the ratings do and do not support."""
    trial = _open_trial(directory, name)
    try:
        result = trial.reveal(acknowledged=i_am_done, force=force)
    except ValueError as error:
        raise _fail(str(error)) from None

    echo(f"trial {trial.name} — unblinded\n")
    echo(f"  {result.beat_arm} carried {trial.treatment_noun}")
    echo(f"    ({trial.treatment_label})")
    echo(f"  {result.null_arm} had none\n")
    echo(
        f"  with beat     n={result.n_beat}  mean {result.mean_beat:.2f}  "
        f"{list(result.beat_states)}"
    )
    echo(
        f"  without       n={result.n_null}  mean {result.mean_null:.2f}  "
        f"{list(result.null_states)}"
    )
    echo(f"  difference    {result.difference:+.2f}")
    if result.excluded:
        echo(f"  excluded      {result.excluded} session(s) logged after unblinding")

    echo(f"\n{result.verdict}\n")
    echo(
        f"For scale: detecting a large effect needs about "
        f"{result.sessions_for_large_effect} sessions per arm, a medium one about "
        f"{result.sessions_for_medium_effect}. And a p-value only means what it "
        f"says if you fixed the number of sessions before you started, rather "
        f"than stopping when the numbers looked interesting."
    )


@trial_app.command("list")
def trial_list(*, directory: TrialsDir = Path("trials")) -> None:
    """List the trials in a directory."""
    if not directory.is_dir():
        echo(f"no trials in {directory}")
        return
    found = sorted(
        path for path in directory.iterdir() if (path / "trial.json").is_file()
    )
    if not found:
        echo(f"no trials in {directory}")
        return
    for path in found:
        trial = Trial.load(path)
        counts = trial.counts()
        state = "revealed" if trial.revealed else "blind"
        echo(
            f"  {trial.name:<16} {trial.preset:<12} {trial.treatment_label:<24} "
            + ", ".join(f"{a} {counts[a]}" for a in ARMS)
            + f"  [{state}]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
