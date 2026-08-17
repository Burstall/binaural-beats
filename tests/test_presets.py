"""
Stage 6: presets, the CLI, and the golden files.

The golden test is the one that matters most here. Everything else checks that
a preset means what it says; the goldens check that it still means it tomorrow.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from violet.cli import app
from violet.engine import ArraySink, render
from violet.layers import Air, BinauralPair, ChordBed, Layer, Ocean, Pedal
from violet.presets import (
    BANDS,
    BUILTIN_PRESETS,
    Preset,
    load_library,
    resolve_beat,
)

GOLDEN = Path(__file__).parent / "golden" / "presets.json"
GOLDEN_SECONDS = 5.0

LIBRARY = load_library()
NAMES = LIBRARY.names


def render_seconds(preset: Preset, seconds: float = GOLDEN_SECONDS) -> np.ndarray:
    """A short render of a preset, with its own settings otherwise intact."""
    short = preset.with_overrides(minutes=seconds / 60.0, block_seconds=1.0)
    sink = ArraySink()
    render(short.build_layers(), short.render_config(), sink)
    return sink.result


def digest(audio: np.ndarray) -> str:
    """Hash the quantised samples — what a listener would actually receive."""
    quantised = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    return hashlib.sha256(quantised.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------


def test_the_builtin_presets_load() -> None:
    assert len(LIBRARY) >= 6
    assert {"tones", "layered", "ocean"} <= set(NAMES)


def test_the_preset_file_is_shipped_inside_the_package() -> None:
    """Not read from the source tree — an installed wheel has to work too."""
    assert BUILTIN_PRESETS.is_file()
    assert BUILTIN_PRESETS.parent.name == "data"


@pytest.mark.parametrize("name", NAMES)
def test_every_preset_describes_a_renderable_piece(name: str) -> None:
    preset = LIBRARY[name]
    assert preset.description
    assert preset.base_hz > 0
    assert preset.layers
    assert 250.0 <= preset.carrier_hz <= 520.0
    layers = preset.build_layers()
    assert len(layers) == len(preset.layers)


@pytest.mark.parametrize("name", NAMES)
def test_every_preset_stays_under_the_ceiling(name: str) -> None:
    """Constraint 7, on every preset that ships."""
    preset = LIBRARY[name].with_overrides(minutes=8.0 / 60.0, block_seconds=1.0)
    sink = ArraySink()
    result = render(preset.build_layers(), preset.render_config(), sink)
    assert result.peak < 0.95, result.peak
    assert result.clipped == 0
    assert np.max(np.abs(sink.result)) < 0.95


@pytest.mark.parametrize("name", NAMES)
def test_every_preset_is_block_size_invariant(name: str) -> None:
    """Constraint 2, on every preset that ships: one second against ten."""
    preset = LIBRARY[name].with_overrides(minutes=5.0 / 60.0)
    reference = ArraySink()
    render(
        preset.with_overrides(block_seconds=10.0).build_layers(),
        preset.with_overrides(block_seconds=10.0).render_config(),
        reference,
    )
    other = ArraySink()
    render(
        preset.with_overrides(block_seconds=1.0).build_layers(),
        preset.with_overrides(block_seconds=1.0).render_config(),
        other,
    )
    assert np.max(np.abs(other.result - reference.result)) < 1e-6


@pytest.mark.parametrize("name", NAMES)
def test_every_preset_is_deterministic(name: str) -> None:
    preset = LIBRARY[name].with_overrides(minutes=4.0 / 60.0, block_seconds=1.0)
    first = render_seconds(preset, 4.0)
    second = render_seconds(preset, 4.0)
    assert np.array_equal(first, second)


def test_the_prototype_presets_match_their_scripts() -> None:
    """Levels and rates, against the prototypes they came from."""
    tones = LIBRARY["tones"]
    assert tones.sample_rate == 44100
    assert [spec.level for spec in tones.layers] == [0.30, 0.09]

    layered = LIBRARY["layered"]
    assert layered.gain == 0.82
    assert [spec.level for spec in layered.layers] == [0.22, 0.085, 0.055, 0.085, 0.06]
    assert [getattr(spec, "ratio", None) for spec in layered.layers[:3]] == [
        1.0,
        2.0,
        1.5,
    ]

    ocean = LIBRARY["ocean"]
    assert ocean.sample_rate == 32000
    assert ocean.gain == 1.25
    assert ocean.fade_seconds == 22.0
    assert ocean.fade_denominator == 5
    assert [spec.level for spec in ocean.layers] == [0.070, 0.080, 0.42]


def test_layer_kinds_build_the_layers_they_name() -> None:
    kinds = {
        "tones": (BinauralPair, Pedal),
        "layered": (BinauralPair, BinauralPair, BinauralPair, Pedal, Air),
        "ocean": (ChordBed, Pedal, Ocean),
    }
    for name, expected in kinds.items():
        built = LIBRARY[name].build_layers()
        assert tuple(type(layer) for layer in built) == expected


def test_the_layered_preset_swells_its_upper_voices() -> None:
    pairs = [
        layer
        for layer in LIBRARY["layered"].build_layers()
        if isinstance(layer, BinauralPair)
    ]
    assert pairs[0].swell is None
    assert pairs[1].swell is not None
    assert pairs[1].swell.period == 70.0
    assert pairs[2].swell is not None
    assert pairs[2].swell.period == 110.0
    assert pairs[2].swell.phase == 1.7


def test_the_air_bed_matches_the_prototypes_one_pole() -> None:
    """
    The cutoff is written in Hz, and lands on the prototype's coefficient.

    The prototype hard-coded ``a = 0.02``, which is a sample-rate-dependent
    number pretending to be a filter setting. The preset gives a frequency
    instead — 141.7974 Hz, which at 44.1 kHz *is* 0.02, and which still means
    the same thing if the rate changes.
    """
    air = next(
        layer for layer in LIBRARY["layered"].build_layers() if isinstance(layer, Air)
    )
    assert air.cutoff_hz == pytest.approx(141.7974, abs=1e-4)
    coefficient = 1.0 - np.exp(-2.0 * np.pi * air.cutoff_hz / 44100)
    assert coefficient == pytest.approx(0.02, abs=1e-8)


# ---------------------------------------------------------------------------
# overrides, bands, and user files
# ---------------------------------------------------------------------------


def test_named_bands_resolve() -> None:
    assert resolve_beat("theta") == 4.0
    assert resolve_beat("THETA") == 4.0
    assert resolve_beat("7.83") == 7.83
    assert resolve_beat(2.0) == 2.0
    assert set(BANDS) == {"delta", "theta", "schumann", "alpha", "beta"}


def test_an_unknown_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown beat"):
        resolve_beat("gamma-ish")


def test_overrides_apply_and_none_is_ignored() -> None:
    preset = LIBRARY["ocean"]
    changed = preset.with_overrides(minutes=3.0, beat=None, seed=99)
    assert changed.minutes == 3.0
    assert changed.beat == preset.beat
    assert changed.seed == 99
    assert preset.with_overrides() is preset


def test_a_different_base_transposes_everything() -> None:
    """The base frequency is a parameter, end to end."""
    preset = LIBRARY["ocean"].with_overrides(base_hz=136.10)
    assert preset.carrier_hz == pytest.approx(272.2)

    layers = preset.build_layers()
    bed = next(layer for layer in layers if isinstance(layer, ChordBed))
    pedal = next(layer for layer in layers if isinstance(layer, Pedal))
    ocean = next(layer for layer in layers if isinstance(layer, Ocean))

    assert bed.root == pytest.approx(272.2)
    assert pedal.freq == pytest.approx(136.10)
    assert ocean.notch_hz == pytest.approx(272.2)
    assert bed.events[0].chord.note_labels(bed.root)[0].startswith("C#")


def test_a_zero_beat_tone_preset_has_identical_channels() -> None:
    """With no detune there is no interaural difference at all."""
    preset = LIBRARY["tones"].with_overrides(beat=0.0, minutes=4.0 / 60.0)
    audio = render_seconds(preset, 4.0)
    assert np.array_equal(audio[:, 0], audio[:, 1])


def test_the_null_condition_changes_only_the_beat() -> None:
    """
    Roadmap item 1, available now: beat=0 is a real placebo.

    Same seed, same progression, same waves, same noise — the ocean bed is
    bit-identical between the two renders — and the only difference is that one
    has no detune between the ears and therefore no beat. The ocean stays
    decorrelated in both, because that is what it is; decorrelated noise
    carries no beat either way, which is precisely why it is safe to leave in.
    """
    beating = LIBRARY["ocean"].with_overrides(beat=4.0, minutes=4.0 / 60.0)
    silent = LIBRARY["ocean"].with_overrides(beat=0.0, minutes=4.0 / 60.0)

    assert [event.chord.numeral for event in _bed(beating).events] == [
        event.chord.numeral for event in _bed(silent).events
    ]

    one, two = _ocean_only(beating), _ocean_only(silent)
    assert np.array_equal(one, two)
    assert not np.array_equal(render_seconds(beating, 4.0), render_seconds(silent, 4.0))


def _bed(preset: Preset) -> ChordBed:
    return next(layer for layer in preset.build_layers() if isinstance(layer, ChordBed))


def _ocean_only(preset: Preset) -> np.ndarray:
    short = preset.with_overrides(minutes=4.0 / 60.0, block_seconds=1.0)
    bed = [layer for layer in short.build_layers() if isinstance(layer, Ocean)]
    sink = ArraySink()
    render(bed, short.render_config(), sink)
    return sink.result


def test_a_user_file_extends_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "mine.toml"
    path.write_text(
        """
[mine]
description = "A frequency of my own."
base_hz = 220.0
beat = "alpha"
minutes = 1.0

[[mine.layer]]
kind = "pair"
level = 0.25

[tones]
description = "Replaced."
base_hz = 100.0
beat = 3.0
minutes = 2.0

[[tones.layer]]
kind = "pair"
level = 0.2
""",
        encoding="utf-8",
    )
    library = load_library(path)
    assert "mine" in library
    assert library["mine"].beat == 10.0
    assert library["mine"].carrier_hz == pytest.approx(440.0)
    assert library["tones"].description == "Replaced."
    assert library["tones"].base_hz == 100.0


def test_a_bad_layer_kind_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[x]\ndescription = "d"\nbase_hz = 100.0\nbeat = 4.0\n'
        '[[x.layer]]\nkind = "trumpet"\nlevel = 0.1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not one of"):
        load_library(path)


def test_a_bad_layer_field_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[x]\ndescription = "d"\nbase_hz = 100.0\nbeat = 4.0\n'
        '[[x.layer]]\nkind = "pair"\nloudness = 0.1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bad pair layer"):
        load_library(path)


def test_a_preset_with_no_layers_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[x]\ndescription = "d"\nbase_hz = 100.0\nbeat = 4.0\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"no \[\[x\.layer\]\] tables"):
        load_library(path)


def test_a_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="no preset file"):
        load_library("nowhere.toml")


def test_an_unknown_preset_lists_what_there_is() -> None:
    with pytest.raises(KeyError, match="available: "):
        LIBRARY["nope"]


def test_the_preset_file_parses_as_plain_toml() -> None:
    """No custom syntax: it is a TOML file and nothing more."""
    with BUILTIN_PRESETS.open("rb") as handle:
        document = tomllib.load(handle)
    assert set(document) == set(NAMES)


# ---------------------------------------------------------------------------
# golden files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_golden(name: str) -> None:
    """
    Five seconds of every preset, hashed.

    A failure here is not necessarily a bug. It means the output changed, and
    the question is whether you meant it to. If you did, re-run
    ``uv run python -m tests.regolden`` — or just delete the entry and let the
    next run tell you the new hash — and commit the change as its own act.
    """
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert name in expected, f"no golden hash for {name!r}"
    assert digest(render_seconds(LIBRARY[name])) == expected[name], name


def test_the_golden_file_covers_every_preset() -> None:
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(expected) == set(NAMES)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def runner() -> CliRunner:
    return CliRunner()


def test_cli_lists_presets() -> None:
    result = runner().invoke(app, ["presets"])
    assert result.exit_code == 0
    for name in NAMES:
        assert name in result.output


def test_cli_shows_a_preset() -> None:
    result = runner().invoke(app, ["show", "ocean"])
    assert result.exit_code == 0
    assert "335.796" in result.output
    assert "notched at" in result.output
    assert "E4 G4 B4 F#5" in result.output


def test_cli_reports_an_unknown_preset() -> None:
    result = runner().invoke(app, ["show", "trumpet"])
    assert result.exit_code == 1
    assert "unknown preset" in result.output


def test_cli_tunes_a_frequency() -> None:
    result = runner().invoke(app, ["tune", "83.949"])
    assert result.exit_code == 0
    assert "32.1 cents sharp" in result.output
    assert "406.0 nm" in result.output


def test_cli_rejects_a_frequency_of_zero() -> None:
    result = runner().invoke(app, ["tune", "0"])
    assert result.exit_code == 1
    assert "finite and positive" in result.output


def test_cli_renders_a_file(tmp_path: Path) -> None:
    out = tmp_path / "session.flac"
    result = runner().invoke(
        app,
        ["render", "tones", "--minutes", "0.05", "--beat", "alpha", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    audio, rate = sf.read(out, dtype="float64", always_2d=True)
    assert rate == 44100
    assert audio.shape == (round(44100 * 3.0), 2)
    assert np.max(np.abs(audio)) < 0.95
    assert "peak" in result.output


def test_cli_renders_a_loop_and_reports_the_snapping(tmp_path: Path) -> None:
    out = tmp_path / "loop.wav"
    result = runner().invoke(
        app,
        [
            "render",
            "tones-loop",
            "--minutes",
            "0.5",
            "--out",
            str(out),
            "--subtype",
            "PCM_24",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "to close the loop" in result.output
    assert sf.info(out).subtype == "PCM_24"


def test_cli_honours_a_user_preset_file(tmp_path: Path) -> None:
    path = tmp_path / "mine.toml"
    path.write_text(
        '[mine]\ndescription = "Mine."\nbase_hz = 220.0\nbeat = 6.0\nminutes = 0.05\n'
        '[[mine.layer]]\nkind = "pair"\nlevel = 0.2\n',
        encoding="utf-8",
    )
    out = tmp_path / "mine.wav"
    result = runner().invoke(
        app, ["render", "mine", "--presets", str(path), "--out", str(out), "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert result.output == ""


def test_cli_overrides_the_base_frequency(tmp_path: Path) -> None:
    result = runner().invoke(
        app,
        [
            "render",
            "tones",
            "--base",
            "136.10",
            "--minutes",
            "0.02",
            "--out",
            str(tmp_path / "om.wav"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "C#3" in result.output
    assert "272.200 Hz" in result.output


def test_cli_rejects_an_unknown_band(tmp_path: Path) -> None:
    result = runner().invoke(
        app,
        [
            "render",
            "tones",
            "--beat",
            "gamma-ish",
            "--minutes",
            "0.02",
            "--out",
            str(tmp_path / "x.wav"),
        ],
    )
    assert result.exit_code == 1
    assert "unknown beat" in result.output


def test_all_layers_of_all_presets_are_typed_as_layers() -> None:
    for name in NAMES:
        built: list[Layer] = LIBRARY[name].build_layers()
        for layer in built:
            assert isinstance(layer, Layer)
            assert layer.peak >= 0.0
