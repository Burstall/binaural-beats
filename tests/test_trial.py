"""
Stage 7: the blinded trial.

Most of these test the blind rather than the audio. A trial that leaks which
arm is which is worse than no trial, because it produces a number you will
believe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from violet.cli import app
from violet.engine import render_to_file
from violet.presets import load_library
from violet.trial import (
    ARMS,
    MINIMUM_SESSIONS,
    Session,
    Trial,
    compare,
    sessions_for_effect,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Short enough that a test can afford two of them, long enough to be a render.
BRIEF = 0.02


def make_trial(directory: Path, preset: str = "tones", **overrides: float) -> Trial:
    base = load_library()[preset].with_overrides(minutes=BRIEF, **overrides)  # type: ignore[arg-type]
    return Trial.create(base, directory / "t", "t")


def fill(trial: Trial, beat_states: list[int], null_states: list[int]) -> str:
    """Log ratings, returning which arm we treated as the beat arm."""
    beat_arm = ARMS[0]
    for state in beat_states:
        trial.log(beat_arm, state)
    for state in null_states:
        trial.log(ARMS[1], state)
    return beat_arm


# ---------------------------------------------------------------------------
# the blind
# ---------------------------------------------------------------------------


def test_both_arms_are_rendered(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    for arm in ARMS:
        assert trial.arm_path(arm).is_file()
    assert trial.state_path.is_file()
    assert trial.sealed_path.is_file()


def test_the_two_arms_are_byte_for_byte_the_same_length(tmp_path: Path) -> None:
    """
    The leak this design exists to avoid.

    FLAC would give the answer away by file size: with no detune the two
    channels of a tonal preset are identical, its mid/side stage reduces the
    difference channel to almost nothing, and the null arm comes out half the
    size. Uncompressed, both files are exactly as long as each other.
    """
    trial = make_trial(tmp_path)
    sizes = {arm: trial.arm_path(arm).stat().st_size for arm in ARMS}
    assert sizes[ARMS[0]] == sizes[ARMS[1]]
    assert all(path.suffix == ".wav" for path in map(trial.arm_path, ARMS))


def test_flac_would_have_leaked_it(tmp_path: Path) -> None:
    """
    The control for the test above, so the reason is on the record.

    Rendering the same two arms compressed and comparing the sizes: the null
    arm is dramatically smaller. This is why trials are uncompressed.
    """
    preset = load_library()["tones"].with_overrides(minutes=BRIEF)
    sizes = []
    for beat in (preset.beat, 0.0):
        arm = preset.with_overrides(beat=beat)
        path = tmp_path / f"{beat:g}.flac"
        render_to_file(arm.build_layers(), arm.render_config(), path)
        sizes.append(path.stat().st_size)
    assert sizes[1] < sizes[0] * 0.75


def test_exactly_one_arm_carries_the_beat(tmp_path: Path) -> None:
    """
    One arm has identical channels and the other does not.

    Read from the audio rather than from the sealed file, which is the one
    thing a test is allowed to do and a listener is not.
    """
    trial = make_trial(tmp_path)
    identical = []
    for arm in ARMS:
        audio, _ = sf.read(trial.arm_path(arm), dtype="float64", always_2d=True)
        identical.append(bool(np.array_equal(audio[:, 0], audio[:, 1])))
    assert sum(identical) == 1


def test_the_sealed_file_does_not_say_which(tmp_path: Path) -> None:
    """Grep-proof, which is the whole job it has."""
    trial = make_trial(tmp_path)
    sealed = trial.sealed_path.read_text(encoding="utf-8")
    payload = next(
        line for line in sealed.splitlines() if line and not line.startswith("#")
    )
    assert "beat_arm" not in payload
    assert '"A"' not in payload
    assert '"B"' not in payload
    assert "obfuscation, not encryption" in sealed


def test_the_visible_state_does_not_say_which(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    state = trial.state_path.read_text(encoding="utf-8")
    assert "beat_arm" not in state
    assert "A" not in state.replace("base_hz", "").replace("scale", "")


def test_the_assignment_is_not_derivable_from_the_seed(tmp_path: Path) -> None:
    """
    Same preset, same seed, and the arm still moves.

    If the assignment came from the render seed you could recover it by
    re-running the walk, and every visible byte in the directory would tell you
    the answer. It comes from `secrets` instead.
    """
    seen = set()
    for attempt in range(24):
        trial = make_trial(tmp_path / str(attempt), seed=5)
        audio, _ = sf.read(trial.arm_path("A"), dtype="float64", always_2d=True)
        seen.add(bool(np.array_equal(audio[:, 0], audio[:, 1])))
        if len(seen) == 2:
            return
    pytest.fail("the beat landed on the same arm 24 times running")


def test_a_preset_with_no_beat_cannot_be_trialled(tmp_path: Path) -> None:
    """There would be nothing to compare against."""
    preset = load_library()["tones"].with_overrides(minutes=BRIEF, beat=0.0)
    with pytest.raises(ValueError, match="already the null condition"):
        Trial.create(preset, tmp_path / "t", "t")


def test_a_trial_will_not_overwrite_another(tmp_path: Path) -> None:
    make_trial(tmp_path)
    preset = load_library()["tones"].with_overrides(minutes=BRIEF)
    with pytest.raises(ValueError, match="not empty"):
        Trial.create(preset, tmp_path / "t", "t")


def test_a_backwards_scale_is_rejected(tmp_path: Path) -> None:
    preset = load_library()["tones"].with_overrides(minutes=BRIEF)
    with pytest.raises(ValueError, match="low to high"):
        Trial.create(preset, tmp_path / "t", "t", scale=(5, 1))


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_a_trial_round_trips_through_its_directory(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    trial.log("A", 4, "slept")
    reloaded = Trial.load(trial.directory)

    assert reloaded.name == trial.name
    assert reloaded.beat == trial.beat
    assert reloaded.scale == trial.scale
    assert reloaded.sessions[0].state == 4
    assert reloaded.sessions[0].note == "slept"


def test_loading_a_directory_with_no_trial_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no trial at"):
        Trial.load(tmp_path / "nothing")


@pytest.mark.parametrize(
    ("arm", "state", "message"),
    [("C", 3, "arm must be one of"), ("A", 0, "between 1 and 5"), ("A", 9, "between")],
)
def test_a_bad_rating_is_refused(
    tmp_path: Path, arm: str, state: int, message: str
) -> None:
    trial = make_trial(tmp_path)
    with pytest.raises(ValueError, match=message):
        trial.log(arm, state)


def test_counts_track_both_arms(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    assert trial.counts() == {"A": 0, "B": 0}
    trial.log("A", 3)
    trial.log("A", 4)
    trial.log("B", 2)
    assert trial.counts() == {"A": 2, "B": 1}


def test_the_next_arm_keeps_the_two_balanced(tmp_path: Path) -> None:
    """
    Chosen for you, because choosing by mood measures the mood.

    With one arm behind, the answer is forced; level, it is a coin toss.
    """
    trial = make_trial(tmp_path)
    trial.log("A", 3)
    assert trial.next_arm() == "B"
    trial.log("B", 3)
    assert trial.next_arm() in ARMS

    for _ in range(4):
        trial.log("B", 3)
    assert trial.next_arm() == "A"


# ---------------------------------------------------------------------------
# unblinding
# ---------------------------------------------------------------------------


def test_revealing_needs_saying_so_out_loud(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    with pytest.raises(ValueError, match="ends the blind"):
        trial.reveal()
    assert not Trial.load(trial.directory).revealed


def test_revealing_early_is_refused(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    trial.log("A", 4)
    with pytest.raises(ValueError, match=f"{MINIMUM_SESSIONS} per arm is the floor"):
        trial.reveal(acknowledged=True)
    assert not Trial.load(trial.directory).revealed


def test_revealing_early_is_possible_when_abandoning(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    fill(trial, [4, 5], [2, 1])
    result = trial.reveal(acknowledged=True, force=True)
    assert result.beat_arm in ARMS
    assert result.null_arm != result.beat_arm
    assert Trial.load(trial.directory).revealed


def test_the_reveal_matches_the_audio(tmp_path: Path) -> None:
    """The arm it names is the one whose channels differ."""
    trial = make_trial(tmp_path)
    fill(trial, [4], [2])
    result = trial.reveal(acknowledged=True, force=True)

    audio, _ = sf.read(trial.arm_path(result.beat_arm), dtype="float64", always_2d=True)
    assert not np.array_equal(audio[:, 0], audio[:, 1])
    null, _ = sf.read(trial.arm_path(result.null_arm), dtype="float64", always_2d=True)
    assert np.array_equal(null[:, 0], null[:, 1])


def test_sessions_after_unblinding_are_marked_and_excluded(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    fill(trial, [4, 4], [2, 2])
    trial.reveal(acknowledged=True, force=True)

    revealed = Trial.load(trial.directory)
    session = revealed.log("A", 5, "knew which one")
    assert session.after_reveal is True
    assert revealed.counts()["A"] == 2

    result = revealed.reveal(acknowledged=True, force=True)
    assert result.excluded == 1
    assert result.n_beat + result.n_null == 4


def test_a_sealed_file_with_no_payload_is_reported(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    fill(trial, [4], [2])
    trial.sealed_path.write_text("# nothing but a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no sealed payload"):
        trial.reveal(acknowledged=True, force=True)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def sessions(beat: list[int], null: list[int]) -> list[Session]:
    return [Session("A", state, "", "now") for state in beat] + [
        Session("B", state, "", "now") for state in null
    ]


def test_a_clear_difference_is_reported_as_one() -> None:
    result = compare(sessions([5, 5, 4, 5, 4, 5, 5], [1, 2, 1, 1, 2, 1, 2]), "A")
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.cliffs_delta == pytest.approx(1.0)
    assert result.difference > 2.5
    assert "more than chance" in result.verdict


def test_no_difference_is_reported_as_none() -> None:
    result = compare(sessions([3, 4, 2, 3, 4, 3], [3, 2, 4, 3, 3, 4]), "A")
    assert result.p_value is not None
    assert result.p_value > 0.05
    assert "No difference beyond" in result.verdict


def test_a_small_significant_result_is_told_it_will_shrink() -> None:
    """
    The caveat has to match the direction, or it reads as a contradiction.

    Saying "could not detect a large effect" immediately after detecting one is
    nonsense. The real point for a small significant sample is the winner's
    curse: it cleared the threshold because the effect it happened to see was
    on the large side, so it will shrink with more data.
    """
    result = compare(sessions([5, 5, 5], [1, 1, 1]), "A")
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert "expect it to shrink" in result.verdict
    assert "could not reliably" not in result.verdict
    assert str(result.sessions_for_large_effect) in result.verdict


def test_a_small_null_result_is_told_it_proves_nothing() -> None:
    """The other direction: no difference found is not no difference."""
    result = compare(sessions([3, 4, 3], [3, 4, 4]), "A")
    assert result.p_value is not None
    assert result.p_value > 0.05
    assert "could not reliably have detected" in result.verdict
    assert "not evidence there isn't one" in result.verdict


def test_a_large_trial_is_not_lectured() -> None:
    result = compare(sessions([4] * 30 + [5] * 10, [3] * 30 + [2] * 10), "A")
    assert "could not reliably" not in result.verdict
    assert "expect it to shrink" not in result.verdict


def test_identical_ratings_have_nothing_to_test() -> None:
    result = compare(sessions([3, 3, 3], [3, 3, 3]), "A")
    assert result.p_value is None
    assert "nothing to test" in result.verdict


def test_one_empty_arm_has_nothing_to_compare() -> None:
    result = compare(sessions([3, 4], []), "A")
    assert result.p_value is None
    assert result.cliffs_delta is None
    assert "Nothing to compare" in result.verdict
    assert np.isnan(result.mean_null)


def test_cliffs_delta_runs_from_minus_one_to_one() -> None:
    assert compare(sessions([5, 5], [1, 1]), "A").cliffs_delta == pytest.approx(1.0)
    assert compare(sessions([1, 1], [5, 5]), "A").cliffs_delta == pytest.approx(-1.0)
    assert compare(sessions([1, 5], [1, 5]), "A").cliffs_delta == pytest.approx(0.0)


def test_the_beat_arm_is_reported_as_given() -> None:
    result = compare(sessions([5, 4], [1, 2]), "B")
    assert result.beat_arm == "B"
    assert result.null_arm == "A"
    # The states swap with it: "A" is now the null arm.
    assert result.null_states == (5, 4)


def test_the_sessions_needed_are_sobering() -> None:
    assert sessions_for_effect(0.8) == 27
    assert sessions_for_effect(0.5) == 68
    assert sessions_for_effect(0.2) > 400
    with pytest.raises(ValueError, match="must be positive"):
        sessions_for_effect(0.0)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def run(*args: str) -> object:
    return CliRunner().invoke(app, list(args))


def test_the_cli_runs_a_whole_trial(tmp_path: Path) -> None:
    directory = str(tmp_path)
    started = run(
        "trial",
        "start",
        "sleep-test",
        "--preset",
        "tones",
        "--minutes",
        str(BRIEF),
        "--dir",
        directory,
    )
    assert started.exit_code == 0, started.output  # type: ignore[attr-defined]
    assert "one of those has the beat" in started.output.lower()  # type: ignore[attr-defined]
    assert "file size" in started.output  # type: ignore[attr-defined]

    nudge = run("trial", "next", "sleep-test", "--dir", directory)
    assert nudge.exit_code == 0  # type: ignore[attr-defined]
    assert "play " in nudge.output  # type: ignore[attr-defined]

    for arm, state in (("A", 4), ("B", 2), ("a", 5)):
        logged = run(
            "trial",
            "log",
            "sleep-test",
            arm,
            "--state",
            str(state),
            "--note",
            "ok",
            "--dir",
            directory,
        )
        assert logged.exit_code == 0, logged.output  # type: ignore[attr-defined]

    status = run("trial", "status", "sleep-test", "--dir", directory)
    assert status.exit_code == 0  # type: ignore[attr-defined]
    assert "A 2, B 1" in status.output  # type: ignore[attr-defined]
    assert "will not be until you reveal" in status.output  # type: ignore[attr-defined]

    listed = run("trial", "list", "--dir", directory)
    assert "sleep-test" in listed.output  # type: ignore[attr-defined]
    assert "[blind]" in listed.output  # type: ignore[attr-defined]

    early = run("trial", "reveal", "sleep-test", "--i-am-done", "--dir", directory)
    assert early.exit_code == 1  # type: ignore[attr-defined]
    assert "floor" in early.output  # type: ignore[attr-defined]

    revealed = run(
        "trial", "reveal", "sleep-test", "--i-am-done", "--force", "--dir", directory
    )
    assert revealed.exit_code == 0, revealed.output  # type: ignore[attr-defined]
    assert "unblinded" in revealed.output  # type: ignore[attr-defined]
    assert "carried the" in revealed.output  # type: ignore[attr-defined]
    assert "sessions per arm" in revealed.output  # type: ignore[attr-defined]

    after = run("trial", "list", "--dir", directory)
    assert "[revealed]" in after.output  # type: ignore[attr-defined]


def test_the_cli_status_gives_nothing_away(tmp_path: Path) -> None:
    directory = str(tmp_path)
    run(
        "trial",
        "start",
        "t",
        "--preset",
        "tones",
        "--minutes",
        str(BRIEF),
        "--dir",
        directory,
    )
    run("trial", "log", "t", "A", "--state", "5", "--dir", directory)
    status = run("trial", "status", "t", "--dir", directory)
    assert "carried" not in status.output  # type: ignore[attr-defined]


def test_the_cli_reports_a_missing_trial(tmp_path: Path) -> None:
    result = run("trial", "status", "nope", "--dir", str(tmp_path))
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "no trial at" in result.output  # type: ignore[attr-defined]


def test_the_cli_reports_an_empty_trial_directory(tmp_path: Path) -> None:
    result = run("trial", "list", "--dir", str(tmp_path / "nowhere"))
    assert result.exit_code == 0  # type: ignore[attr-defined]
    assert "no trials in" in result.output  # type: ignore[attr-defined]


def test_the_cli_rejects_a_bad_scale(tmp_path: Path) -> None:
    result = run(
        "trial",
        "start",
        "t",
        "--preset",
        "tones",
        "--minutes",
        str(BRIEF),
        "--scale",
        "wide",
        "--dir",
        str(tmp_path),
    )
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "--scale must look like" in result.output  # type: ignore[attr-defined]


def test_the_cli_rejects_an_unknown_preset_for_a_trial(tmp_path: Path) -> None:
    result = run("trial", "start", "t", "--preset", "trumpet", "--dir", str(tmp_path))
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "unknown preset" in result.output  # type: ignore[attr-defined]


def test_the_cli_refuses_to_start_a_trial_with_no_beat(tmp_path: Path) -> None:
    result = run(
        "trial",
        "start",
        "t",
        "--preset",
        "tones",
        "--beat",
        "0",
        "--minutes",
        str(BRIEF),
        "--dir",
        str(tmp_path),
    )
    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "null condition" in result.output  # type: ignore[attr-defined]
