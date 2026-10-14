"""
Stage 9: amplitude modulation, recorded music, and the placebo for both.

The headline test is the null condition. A depth of zero has to give back the
unmodulated mix *sample for sample* — not nearly, not after quantisation —
because that is what makes a blinded trial of the modulation exact, and it is
the property easiest to lose to a refactor that "simplifies" the arithmetic
into ``rest + gain * band``.

The rest pin down what the modulation is: the rate it pulses at, how deep, in
which band, the same in both ears, and that it stays inside its own peak
bound.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from typer.testing import CliRunner

import music
from violet.cli import app
from violet.dsp.curves import BeatCurve
from violet.dsp.modulation import (
    SUBDIVISIONS,
    Modulation,
    band_gain_bound,
    tempo_locked,
)
from violet.engine import ArraySink, NullSink, RenderConfig, render
from violet.layers import Air, Layer, Modulated, Pedal, Span, StatefulLayer, Track
from violet.presets import TrackSpec, load_library, resolve_modulation
from violet.trial import Trial

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from violet._types import FloatArray

SR = 32000


def run(
    layers: Sequence[Layer], seconds: float, block: float = 1.0, sr: int = SR
) -> FloatArray:
    """Render at unity gain with no fade, so what comes out is the layers."""
    sink = ArraySink()
    config = RenderConfig(
        sample_rate=sr,
        duration=seconds,
        block_seconds=block,
        gain=1.0,
        fade_seconds=0.0,
    )
    render(layers, config, sink)
    return sink.result


def envelope(x: FloatArray, sr: int = SR, trim: float = 0.5) -> FloatArray:
    """
    Amplitude envelope of a single tone, from its analytic signal.

    The ends are trimmed: the band filter starts from rest and the Hilbert
    transform wraps, and neither has anything to do with the modulation.
    """
    cut = int(trim * sr)
    out: FloatArray = np.abs(signal.hilbert(x))[cut:-cut]
    return out


def tone(freq: float, level: float = 0.5) -> Pedal:
    """A mono sine, which is what a pedal is."""
    return Pedal(freq=freq, level=level)


def modulated(
    inner: Sequence[Layer], rate: float = 16.0, depth: float = 0.5, sr: int = SR
) -> Modulated:
    return Modulated(
        layers=tuple(inner),
        modulation=Modulation(rate=rate, depth=depth),
        sample_rate=sr,
    )


# ---------------------------------------------------------------------------
# the modulation itself
# ---------------------------------------------------------------------------


def test_tempo_locking_hits_the_grid_exactly_where_it_can() -> None:
    assert tempo_locked(16.0, 120.0) == pytest.approx(16.0)
    assert tempo_locked(8.0, 120.0) == pytest.approx(8.0)
    assert tempo_locked(32.0, 120.0) == pytest.approx(32.0)


def test_tempo_locking_picks_the_nearest_subdivision_by_ratio() -> None:
    """At 84 bpm, 16 Hz sits between 11.2 (8 per beat) and 16.8 (12 per beat)."""
    assert tempo_locked(16.0, 84.0) == pytest.approx(16.8)
    rate = tempo_locked(16.0, 93.0)
    per_beat = rate / (93.0 / 60.0)
    assert round(per_beat) in SUBDIVISIONS
    assert per_beat == pytest.approx(round(per_beat))


@pytest.mark.parametrize(("hz", "bpm"), [(0.0, 120.0), (16.0, 0.0), (-1.0, 90.0)])
def test_tempo_locking_rejects_nonsense(hz: float, bpm: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        tempo_locked(hz, bpm)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rate": 16.0, "depth": 1.5}, "depth"),
        ({"rate": 16.0, "depth": -0.1}, "depth"),
        ({"rate": 0.0}, "rate must be positive"),
        ({"rate": 16.0, "band": (1000.0, 200.0)}, "band"),
        ({"rate": 16.0, "band": (0.0, 200.0)}, "band"),
        ({"rate": BeatCurve(points=((0.0, 10.0), (60.0, 0.0)))}, "above 0 Hz"),
    ],
)
def test_a_bad_modulation_is_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Modulation(**kwargs)  # type: ignore[arg-type]


def test_the_dip_is_zero_on_the_cycle_and_full_depth_between() -> None:
    """Full level on every pulse boundary, so locked to a tempo, on the beat."""
    mod = Modulation(rate=16.0, depth=0.4, offset=0.25)
    tops = 0.25 + np.arange(10) / 16.0
    bottoms = tops + 1.0 / 32.0
    assert np.allclose(mod.dip(tops), 0.0, atol=1e-12)
    assert np.allclose(mod.dip(bottoms), 0.4)
    t = np.linspace(0.0, 3.0, 10_001)
    assert mod.dip(t).min() >= -1e-12
    assert mod.dip(t).max() <= 0.4 + 1e-12


def test_a_moving_rate_integrates_like_a_beat_curve() -> None:
    curve = BeatCurve(points=((0.0, 16.0), (60.0, 8.0)), shape="exponential")
    mod = Modulation(rate=curve, depth=0.5, offset=10.0)
    t = np.linspace(0.0, 90.0, 9001)
    expected = curve.cycles(t) - curve.cycles(np.array([10.0]))[0]
    assert np.allclose(mod.cycles(t), expected)
    assert "exponential" in mod.describe()


def test_the_band_bound_is_the_filters_l1_gain() -> None:
    """About 1.76 for the fourth-order band, whatever the sample rate."""
    for sr in (32000, 44100, 48000):
        bound = band_gain_bound(Modulation(rate=16.0).design(sr), sr)
        assert 1.7 < bound < 1.8


# ---------------------------------------------------------------------------
# the null condition
# ---------------------------------------------------------------------------


def test_a_depth_of_zero_is_the_input_sample_for_sample() -> None:
    """The placebo is not an approximation of the original. It is the original."""
    inner: list[Layer] = [tone(500.0), Air(level=0.03, sample_rate=SR)]
    plain = run(inner, 4.0)
    for layer in inner:
        if isinstance(layer, StatefulLayer):
            layer.reset()
    nulled = run([modulated(inner, depth=0.0)], 4.0)
    assert np.array_equal(plain, nulled)


def test_a_depth_of_zero_has_the_inputs_peak_bound() -> None:
    inner = [tone(500.0, 0.3), tone(80.0, 0.2)]
    assert modulated(inner, depth=0.0).peak == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# what it does to sound
# ---------------------------------------------------------------------------


def test_an_in_band_tone_pulses_at_the_rate_and_to_the_depth() -> None:
    out = run([modulated([tone(500.0)], rate=16.0, depth=0.5)], 6.0)
    env = envelope(out[:, 0])

    freqs = np.fft.rfftfreq(env.size, 1.0 / SR)
    spectrum = np.abs(np.fft.rfft(env - env.mean()))
    assert freqs[np.argmax(spectrum)] == pytest.approx(16.0, abs=0.25)

    # Full level 0.5 at the top of each pulse, half that at the bottom.
    assert env.max() == pytest.approx(0.5, rel=0.03)
    assert env.min() == pytest.approx(0.25, rel=0.06)


@pytest.mark.parametrize("freq", [50.0, 5000.0])
def test_a_tone_outside_the_band_barely_moves(freq: float) -> None:
    """Bass stays still, so the kick does not pump; the top stays still too."""
    out = run([modulated([tone(freq)], rate=16.0, depth=0.5)], 6.0)
    env = envelope(out[:, 0])
    ripple = (env.max() - env.min()) / env.max()
    assert ripple < 0.06, ripple


def test_the_pulse_is_the_same_in_both_ears() -> None:
    """Loudness, not interaural phase: a mono input stays mono."""
    out = run([modulated([tone(500.0), Air(level=0.03, sample_rate=SR)])], 3.0)
    assert np.array_equal(out[:, 0], out[:, 1])


def test_the_peak_bound_holds_against_a_loud_in_band_mix() -> None:
    inner: list[Layer] = [
        tone(300.0, 0.3),
        tone(450.0, 0.3),
        tone(700.0, 0.3),
        Air(level=0.02, sample_rate=SR),
    ]
    layer = modulated(inner, depth=1.0)
    out = run([layer], 10.0)
    assert np.abs(out).max() <= layer.peak


def test_a_modulated_mix_is_block_size_invariant() -> None:
    """The band filters carry their state; block edges must not be heard."""
    inner_a: list[Layer] = [tone(500.0), Air(level=0.03, sample_rate=SR)]
    inner_b: list[Layer] = [tone(500.0), Air(level=0.03, sample_rate=SR)]
    one = run([modulated(inner_a)], 5.0, block=1.0)
    ten = run([modulated(inner_b)], 5.0, block=10.0)
    assert np.max(np.abs(one - ten)) < 1e-6


def test_a_reset_gives_the_same_render_twice() -> None:
    layer = modulated([tone(500.0), Air(level=0.03, sample_rate=SR)])
    first = run([layer], 3.0)
    second = run([layer], 3.0)
    assert np.array_equal(first, second)


def test_a_modulated_group_must_be_rendered_in_order() -> None:
    layer = modulated([tone(500.0)])
    layer.render(Span(0, 100, SR))
    with pytest.raises(ValueError, match="in order"):
        layer.render(Span(500, 600, SR))


def test_a_modulated_group_refuses_another_sample_rate() -> None:
    layer = modulated([tone(500.0)])
    with pytest.raises(ValueError, match="filters are designed for one rate"):
        layer.render(Span(0, 100, 44100))


def test_an_empty_modulated_group_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        modulated([])


# ---------------------------------------------------------------------------
# recorded music
# ---------------------------------------------------------------------------


def write(path: Path, data: FloatArray, sr: int = SR) -> Path:
    sf.write(str(path), data, sr, "FLOAT")
    return path


def test_a_track_reads_back_exactly_what_the_file_holds(tmp_path: Path) -> None:
    data = np.random.default_rng(3).uniform(-0.5, 0.5, (SR * 3, 2))
    track = Track(path=write(tmp_path / "t.wav", data), level=1.0)
    out = run([track], 3.0, block=0.7)
    assert np.allclose(out, data.astype(np.float32), atol=0.0)


def test_a_track_can_be_read_in_any_order(tmp_path: Path) -> None:
    """Stateless, because a file can be seeked. Unlike the ocean."""
    data = np.random.default_rng(4).uniform(-0.5, 0.5, (SR, 2))
    track = Track(path=write(tmp_path / "t.wav", data))
    late = track.render(Span(20000, 30000, SR))
    early = track.render(Span(0, 10000, SR))
    assert np.allclose(late[0], data[20000:30000, 0].astype(np.float32))
    assert np.allclose(early[1], data[:10000, 1].astype(np.float32))


def test_a_track_repeats_or_falls_silent_past_its_end(tmp_path: Path) -> None:
    data = np.random.default_rng(5).uniform(-0.5, 0.5, (SR, 2))
    path = write(tmp_path / "t.wav", data)
    looped = run([Track(path=path, repeat=True)], 2.5, block=0.6)
    once = run([Track(path=path, repeat=False)], 2.5, block=0.6)
    stored = data.astype(np.float32)
    assert np.allclose(looped[SR : 2 * SR], stored)
    assert np.allclose(looped[2 * SR :], stored[: SR // 2])
    assert np.allclose(once[:SR], stored)
    assert not once[SR:].any()


def write_ogg(path: Path, data: FloatArray, sr: int = SR) -> Path:
    """
    Ogg Vorbis, written a chunk at a time.

    In chunks because libsndfile's Vorbis encoder overflows the stack on
    Windows when handed a whole file's worth in one call.
    """
    with sf.SoundFile(str(path), "w", sr, data.shape[1], format="OGG") as handle:
        for start in range(0, len(data), 4096):
            handle.write(data[start : start + 4096])
    return path


def test_an_ogg_track_reads_the_same_at_any_block_size(tmp_path: Path) -> None:
    """
    Vorbis cannot be seeked to the sample, so it is read straight through.

    Seeking it per block, as an earlier version did, gave samples up to 0.47
    apart between 0.37 s and 10 s blocks.
    """
    path = write_ogg(tmp_path / "t.ogg", music.synthesise(SR)[: SR * 4])
    straight = sf.read(str(path), dtype="float64", always_2d=True)[0]
    small = run([Track(path=path)], 4.0, block=0.37)
    large = run([Track(path=path)], 4.0, block=10.0)
    assert np.array_equal(small, large)
    assert np.array_equal(small, straight)


def test_an_ogg_track_repeats_exactly(tmp_path: Path) -> None:
    path = write_ogg(tmp_path / "t.ogg", music.synthesise(SR)[:SR])
    straight = sf.read(str(path), dtype="float64", always_2d=True)[0]
    looped = run([Track(path=path, repeat=True)], 2.5, block=0.6)
    assert np.array_equal(looped[SR : 2 * SR], straight)
    assert np.array_equal(looped[2 * SR :], straight[: SR // 2])


def test_an_ogg_track_refuses_to_jump(tmp_path: Path) -> None:
    track = Track(path=write_ogg(tmp_path / "t.ogg", music.synthesise(SR)[:SR]))
    track.render(Span(0, 1000, SR))
    with pytest.raises(ValueError, match="read in order"):
        track.render(Span(5000, 6000, SR))
    early = track.render(Span(0, 1000, SR))
    assert early[0].any()


def test_a_mono_track_goes_to_both_ears(tmp_path: Path) -> None:
    data = np.random.default_rng(6).uniform(-0.5, 0.5, SR)
    out = run([Track(path=write(tmp_path / "m.wav", data), level=0.5)], 1.0)
    assert np.array_equal(out[:, 0], out[:, 1])
    assert np.allclose(out[:, 0], 0.5 * data.astype(np.float32))


def test_a_tracks_peak_is_measured_not_guessed(tmp_path: Path) -> None:
    data = np.zeros((SR, 2))
    data[1234, 1] = -0.625
    track = Track(path=write(tmp_path / "p.wav", data), level=0.8)
    assert track.file_peak == pytest.approx(0.625)
    assert track.peak == pytest.approx(0.5)


def test_a_track_at_another_rate_says_so(tmp_path: Path) -> None:
    track = Track(path=write(tmp_path / "r.wav", np.zeros((100, 2)), 48000))
    with pytest.raises(ValueError, match="render at the track's own rate"):
        track.render(Span(0, 50, SR))


def test_bad_tracks_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Track(path=tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="3 channels"):
        Track(path=write(tmp_path / "c.wav", np.zeros((10, 3))))
    with pytest.raises(ValueError, match="negative"):
        Track(path=write(tmp_path / "l.wav", np.zeros((10, 2))), level=-1.0)


def test_the_air_bed_stays_inside_its_bound() -> None:
    """It once claimed 0.35 at full level and reached ten. Never again."""
    for seed in (1, 7, 2027):
        layer = Air(level=1.0, sample_rate=SR, seed=seed)
        out = run([layer], 120.0, block=10.0)
        assert np.abs(out).max() <= layer.peak


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def test_modulation_tables_resolve() -> None:
    assert resolve_modulation({"hz": "beta"}).rate == 16.0
    locked = resolve_modulation({"hz": 16, "bpm": 84, "depth": 0.3})
    assert locked.rate == pytest.approx(16.8)
    assert locked.depth == 0.3
    moving = resolve_modulation({"hz": [16, 10], "at_minutes": [0, 30]})
    assert isinstance(moving.rate, BeatCurve)
    banded = resolve_modulation({"hz": 10, "band_hz": [300, 800], "offset_seconds": 1})
    assert banded.band == (300.0, 800.0)
    assert banded.offset == 1.0
    same = Modulation(rate=12.0)
    assert resolve_modulation(same) is same


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"depth": 0.5}, "needs hz"),
        ({"hz": 16, "wobble": 1}, "unknown keys"),
        ({"hz": [16, 10]}, "one rate"),
        ({"hz": [16, 10], "at_minutes": [0, 30], "bpm": 90}, "tempo"),
    ],
)
def test_bad_modulation_tables_are_reported(
    table: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_modulation(table)


def test_a_modulated_loop_is_refused() -> None:
    preset = load_library()["deep-focus"]
    with pytest.raises(ValueError, match="cannot be closed into a seamless loop"):
        preset.with_overrides(loop=True)


def test_the_placebo_switches_off_the_beat_and_the_depth_and_nothing_else() -> None:
    library = load_library()
    focus = library["deep-focus"].placebo()
    assert focus.modulation is not None
    assert focus.modulation.depth == 0.0
    assert focus.modulation.rate == 16.0
    assert not focus.tests_something
    assert library["descent"].placebo().beat == 0.0
    assert library["descent"].tests_something
    assert library["deep-focus"].treatment().startswith("modulation 16 Hz")
    assert focus.treatment() == "nothing"


def test_the_placebo_renders_the_unmodulated_mix() -> None:
    preset = load_library()["deep-focus"].with_overrides(minutes=4.0 / 60.0)
    without = replace(preset, modulation=None)
    placebo = preset.placebo()
    a, b = ArraySink(), ArraySink()
    render(without.build_layers(), without.render_config(), a)
    render(placebo.build_layers(), placebo.render_config(), b)
    assert np.array_equal(a.result, b.result)


def test_an_empty_music_slot_says_what_it_needs() -> None:
    preset = load_library()["focus-music"]
    with pytest.raises(ValueError, match="--music"):
        preset.build_layers()
    with pytest.raises(FileNotFoundError, match="--music"):
        TrackSpec().sample_rate  # noqa: B018 - the property is the test


def test_music_fills_the_slot_and_sets_the_rate() -> None:
    path = music.track_path(48000)
    preset = load_library()["focus-music"].with_overrides(music=path)
    assert preset.sample_rate == 48000
    tracks = [spec for spec in preset.layers if isinstance(spec, TrackSpec)]
    assert len(tracks) == 1
    assert tracks[0].path == path
    assert tracks[0].repeat


def test_music_joins_a_preset_that_has_no_slot() -> None:
    preset = load_library()["ocean"].with_overrides(music=music.track_path())
    assert isinstance(preset.layers[0], TrackSpec)
    assert len(preset.layers) == 4
    assert preset.sample_rate == music.SAMPLE_RATE


def test_a_track_in_a_preset_file_is_relative_to_that_file(tmp_path: Path) -> None:
    folder = tmp_path / "mine"
    folder.mkdir()
    write(folder / "song.wav", np.zeros((SR, 2)), 22050)
    (folder / "p.toml").write_text(
        """
[mine]
description = "mine"
base_hz = 100.0
beat = 0.0
modulation = { hz = 12, depth = 0.25 }

[[mine.layer]]
kind = "track"
path = "song.wav"
""",
        encoding="utf-8",
    )
    preset = load_library(folder / "p.toml")["mine"]
    assert preset.sample_rate == 22050
    (layer,) = preset.build_layers()
    assert isinstance(layer, Modulated)
    (track,) = layer.layers
    assert isinstance(track, Track)
    assert track.path == folder / "song.wav"


def test_a_preset_file_with_a_moved_track_still_loads(tmp_path: Path) -> None:
    (tmp_path / "p.toml").write_text(
        """
[gone]
description = "gone"
base_hz = 100.0
beat = 0.0

[[gone.layer]]
kind = "track"
path = "nowhere.wav"
""",
        encoding="utf-8",
    )
    preset = load_library(tmp_path / "p.toml")["gone"]
    with pytest.raises(FileNotFoundError):
        preset.build_layers()


def test_a_bad_modulation_in_a_preset_file_names_the_preset(tmp_path: Path) -> None:
    (tmp_path / "p.toml").write_text(
        """
[shaky]
description = "shaky"
base_hz = 100.0
modulation = { depth = 0.5 }

[[shaky.layer]]
kind = "pedal"
level = 0.1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shaky: a modulation table needs hz"):
        load_library(tmp_path / "p.toml")


def test_a_modulated_presets_gain_is_measured() -> None:
    """Close to the ceiling, where the analytic bound would sit far below it."""
    preset = load_library()["focus-music"].with_overrides(
        music=music.track_path(), minutes=10.0 / 60.0
    )
    layers = preset.build_layers()
    config = preset.settled_config(layers)
    assert config.gain is not None
    result = render(layers, config, NullSink())
    assert result.peak == pytest.approx(min(preset.ceiling, result.peak / config.gain))
    assert result.peak <= preset.ceiling + 1e-12
    analytic = preset.ceiling / layers[0].peak
    assert config.gain > analytic


def test_an_explicit_gain_is_left_alone() -> None:
    preset = replace(load_library()["deep-focus"], gain=0.5)
    assert preset.settled_config(preset.build_layers()).gain == 0.5


def test_an_unmodulated_preset_keeps_the_analytic_gain() -> None:
    preset = load_library()["tones"]
    assert preset.settled_config(preset.build_layers()).gain is None


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------


def test_a_modulation_trial_differs_only_in_the_modulation(tmp_path: Path) -> None:
    # No fade, so one arm can be compared with the file itself.
    preset = replace(
        load_library()["focus-music"].with_overrides(
            music=music.track_path(), minutes=6.0 / 60.0
        ),
        fade_seconds=0.0,
    )
    trial = Trial.create(preset, tmp_path / "t", "t")
    assert trial.modulation is not None
    assert "modulation 16 Hz" in trial.treatment_label
    assert "beat" not in trial.treatment_label

    a, _ = sf.read(str(trial.arm_path("A")))
    b, _ = sf.read(str(trial.arm_path("B")))
    assert a.shape == b.shape
    assert trial.arm_path("A").stat().st_size == trial.arm_path("B").stat().st_size

    # One arm is the track itself at the shared gain, give or take one
    # quantisation step; the other is not.
    gain = preset.settled_config(preset.build_layers()).gain
    assert gain is not None
    source, _ = sf.read(str(music.track_path()))
    expected = source[: a.shape[0]] * gain
    errors = [np.abs(arm - expected).max() for arm in (a, b)]
    assert min(errors) <= 1.5 / 32768
    assert max(errors) > 0.01

    reloaded = Trial.load(trial.directory)
    assert reloaded.modulation == trial.modulation


def test_a_trial_with_nothing_to_test_is_refused(tmp_path: Path) -> None:
    preset = load_library()["deep-focus"].placebo()
    with pytest.raises(ValueError, match="neither"):
        Trial.create(preset, tmp_path / "t", "t")


def test_a_trial_from_before_modulation_still_loads(tmp_path: Path) -> None:
    folder = tmp_path / "old"
    folder.mkdir()
    state = {
        "name": "old",
        "preset": "ocean",
        "beat": 4.0,
        "base_hz": 100.0,
        "seed": 5,
        "minutes": 1.0,
        "sample_rate": 32000,
        "scale": [1, 5],
        "created": "2026-08-17T12:00:00+00:00",
        "revealed_at": None,
    }
    (folder / "trial.json").write_text(json.dumps(state), encoding="utf-8")
    trial = Trial.load(folder)
    assert trial.modulation is None
    assert trial.treatment_label == "beat 4 Hz"


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_cli_modulates_a_file_to_its_own_length_and_rate(tmp_path: Path) -> None:
    source = music.track_path(48000)
    out = tmp_path / "focus.flac"
    result = CliRunner().invoke(
        app, ["modulate", str(source), "--bpm", "84", "--depth", "0.3", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "16.8 Hz, depth 0.3" in result.output
    info = sf.info(str(out))
    assert info.samplerate == 48000
    assert info.frames == sf.info(str(source)).frames


def test_cli_modulating_at_zero_depth_hands_back_the_track(tmp_path: Path) -> None:
    source = music.track_path()
    out = tmp_path / "same.wav"
    result = CliRunner().invoke(
        app, ["modulate", str(source), "--depth", "0", "-o", str(out), "-q"]
    )
    assert result.exit_code == 0, result.output
    original, _ = sf.read(str(source), dtype="int16")
    copied, _ = sf.read(str(out), dtype="int16")
    assert np.array_equal(original, copied)


def test_cli_modulate_will_not_overwrite_its_input(tmp_path: Path) -> None:
    source = tmp_path / "x.wav"
    source.write_bytes(music.track_path().read_bytes())
    result = CliRunner().invoke(app, ["modulate", str(source), "-o", str(source)])
    assert result.exit_code == 1
    assert "refusing to write over the input" in result.output


def test_cli_renders_focus_music_with_a_track(tmp_path: Path) -> None:
    out = tmp_path / "f.flac"
    result = CliRunner().invoke(
        app,
        [
            "render",
            "focus-music",
            "--music",
            str(music.track_path()),
            "--minutes",
            "0.5",
            "--bpm",
            "120",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "measuring the modulated mix's peak" in result.output
    assert "repeating" in result.output
    assert "0 clipped" in result.output
    assert sf.info(str(out)).duration == pytest.approx(30.0)


def test_cli_without_music_says_how_to_give_it(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["render", "focus-music", "-o", str(tmp_path / "x.flac")]
    )
    assert result.exit_code == 1
    assert "--music" in result.output


def test_cli_show_explains_an_empty_slot() -> None:
    result = CliRunner().invoke(app, ["show", "focus-music"])
    assert result.exit_code == 0
    assert "modulation          16 Hz, depth 0.4" in result.output
    assert "--music" in result.output


def test_cli_can_modulate_any_preset(tmp_path: Path) -> None:
    out = tmp_path / "o.flac"
    result = CliRunner().invoke(
        app,
        ["render", "tones", "--minutes", "0.1", "--depth", "0.2", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "modulated 16 Hz, depth 0.2" in result.output


def test_cli_refuses_to_lock_a_moving_rate(tmp_path: Path) -> None:
    presets = tmp_path / "p.toml"
    presets.write_text(
        """
[glide]
description = "glide"
base_hz = 100.0
beat = 0.0
modulation = { hz = [16, 10], at_minutes = [0, 1] }

[[glide.layer]]
kind = "pedal"
level = 0.1
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["render", "glide", "--presets", str(presets), "--bpm", "90"]
    )
    assert result.exit_code == 1
    assert "cannot lock a modulation rate that moves" in result.output


def test_cli_starts_a_curve_trial_without_tripping_on_the_label(
    tmp_path: Path,
) -> None:
    """It used to format the curve with :g after rendering both arms, and crash."""
    result = CliRunner().invoke(
        app,
        [
            "trial",
            "start",
            "d",
            "--preset",
            "descent",
            "--minutes",
            "0.1",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "exponential" in result.output


def test_cli_starts_a_music_trial(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "trial",
            "start",
            "m",
            "--preset",
            "focus-music",
            "--music",
            str(music.track_path()),
            "--minutes",
            "0.1",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "modulation 16 Hz" in result.output
    status = CliRunner().invoke(app, ["trial", "status", "m", "--dir", str(tmp_path)])
    assert "modulation 16 Hz" in status.output
