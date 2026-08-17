"""
Blinded self-experiment: whether the beat actually does anything for you.

The null condition is one parameter set to zero. Render two files from the same
configuration, the same seed, the same progression and the same noise, one with
``beat = 4.0`` and one with ``beat = 0.0``. With no detune there is no
interaural difference and therefore no binaural beat at all, while everything
else about the two files is identical — the ocean beds are bit-identical, and
there is a test that says so. That is as close to a placebo as audio gets, and
it costs nothing to make.

What this module is for is the bookkeeping that makes the blind hold: assigning
the arms in a way you cannot derive, keeping the mapping out of sight, telling
you which one to play so your own mood does not choose for you, and being
honest at the end about whether the numbers support anything.

Four things the design deliberately gets right
----------------------------------------------
**The assignment does not come from the seed.** It comes from
:mod:`secrets`. If it were drawn from the render seed you could recover it by
re-running the walk, and everything visible in the trial directory would tell
you the answer.

**The renders are WAV, not FLAC.** A compressed format leaks the arm through
its own file size: with no detune the two channels of a tonal preset are
identical, FLAC's mid/side stage reduces the difference channel to nothing, and
the null file comes out *half the size*. Measured on the shipped presets, the
ratio is 0.50 for ``tones``, 0.71 for ``layered`` and 0.91 for ``ocean``. One
glance at a directory listing and the trial is over. Uncompressed files are
byte-for-byte the same length, so this costs disk and buys the blind.

**Which arm to play next is chosen for you.** Left to pick freely, you will
choose by mood, and mood is exactly the thing being measured.
:meth:`Trial.next_arm` keeps the two arms balanced.

**Unblinding is awkward on purpose.** :meth:`Trial.reveal` wants an explicit
acknowledgement and refuses below a floor of sessions unless overridden, and it
records that it happened, so ratings logged afterwards are marked as no longer
blind.

On what the statistics can and cannot tell you
----------------------------------------------
Very little, at realistic session counts. Detecting even a *large* effect
(Cohen's *d* of 0.8) with a rank test at 80% power and the conventional 5%
threshold takes 27 sessions per arm. A medium effect takes 68.
Ten sessions will not settle anything, and :func:`compare` says so rather than
handing you a p-value to over-read.

It also cannot protect you from looking at the numbers, deciding to stop
because they look interesting, and then quoting the p-value as if the stopping
rule had been fixed in advance. Decide how many sessions you are going to do
before you start.
"""

from __future__ import annotations

import base64
import json
import math
import secrets
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from scipy import stats

from violet.dsp.curves import BeatCurve
from violet.engine import render_to_file

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from violet.presets import Preset

__all__ = [
    "ARMS",
    "MINIMUM_SESSIONS",
    "Comparison",
    "Session",
    "Trial",
    "compare",
    "sessions_for_effect",
]

#: The two arms. Which one carries the beat is decided by :mod:`secrets`.
ARMS: tuple[str, str] = ("A", "B")

#: Below this many sessions per arm, :meth:`Trial.reveal` refuses without an
#: explicit override. Not a threshold at which the result becomes meaningful —
#: see the module docstring — just a floor below which it is certainly not.
MINIMUM_SESSIONS = 6

_STATE = "trial.json"
_SEALED = "sealed.txt"
_LOG = "log.jsonl"

_SEAL_HEADER = """\
# This file holds which arm carries the beat, encoded so that you cannot read
# it by accident — while grepping the directory, or glancing at it in an editor.
#
# It is obfuscation, not encryption, and it is not pretending otherwise. You own
# this machine, and the decoder is a dozen lines away in violet/trial.py. What
# it protects against is your own eyes, not an attacker. If you want to know,
# run `violet trial reveal`. That is the honest way, and it records that you did.
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


#: A rate as it sits in the state file: a number, or a curve's points.
BeatJson = float | dict[str, object]


def _beat_to_json(beat: float | BeatCurve) -> BeatJson:
    """A rate as JSON. Curves keep their points, so a trial stays reproducible."""
    if isinstance(beat, BeatCurve):
        return {
            "shape": beat.shape,
            "points": [list(point) for point in beat.points],
        }
    return beat


def _beat_from_json(value: BeatJson) -> float | BeatCurve:
    """The inverse."""
    if isinstance(value, dict):
        raw = value["points"]
        assert isinstance(raw, list)  # noqa: S101 - our own state file
        points = tuple((float(at), float(hz)) for at, hz in raw)
        return BeatCurve(points=points, shape=str(value["shape"]))  # type: ignore[arg-type]
    return float(value)


# ---------------------------------------------------------------------------
# sealing
# ---------------------------------------------------------------------------


def _seal(payload: dict[str, Any], path: Path) -> None:
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    path.write_text(f"{_SEAL_HEADER}{blob}\n", encoding="utf-8")


def _unseal(path: Path) -> dict[str, Any]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not lines:
        msg = f"{path} has no sealed payload; the trial cannot be unblinded"
        raise ValueError(msg)
    decoded: dict[str, Any] = json.loads(base64.b64decode(lines[0]))
    return decoded


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Session:
    """One listening session and what you made of it."""

    arm: str
    state: int
    note: str
    at: str

    #: True when logged after the trial was unblinded. Such sessions are
    #: excluded from the comparison, because they are no longer blind.
    after_reveal: bool = False


@dataclass(frozen=True, slots=True)
class Trial:
    """
    A blinded trial: two rendered arms, a sealed mapping, and a log.

    Everything on this object is safe to show. The mapping lives in the sealed
    file and is only read by :meth:`reveal`.
    """

    directory: Path
    name: str
    preset: str

    #: The rate under test: a number, or a curve that moves. Either way the
    #: other arm is flat zero, which is the whole design.
    beat: float | BeatCurve
    base_hz: float
    seed: int
    minutes: float
    sample_rate: int
    scale: tuple[int, int] = (1, 5)
    created: str = ""
    revealed_at: str | None = None

    # -- paths --------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        """Where the visible configuration lives."""
        return self.directory / _STATE

    @property
    def sealed_path(self) -> Path:
        """Where the mapping lives. Not read except by :meth:`reveal`."""
        return self.directory / _SEALED

    @property
    def log_path(self) -> Path:
        """Where the ratings are appended."""
        return self.directory / _LOG

    def arm_path(self, arm: str) -> Path:
        """The rendered file for one arm."""
        return self.directory / f"{arm}.wav"

    @property
    def revealed(self) -> bool:
        """Whether this trial has been unblinded."""
        return self.revealed_at is not None

    @property
    def beat_label(self) -> str:
        """How to name the rate under test, curve or not."""
        if isinstance(self.beat, BeatCurve):
            return self.beat.describe()
        return f"{self.beat:g} Hz"

    # -- creation and loading ----------------------------------------------

    @classmethod
    def create(
        cls,
        preset: Preset,
        directory: Path,
        name: str,
        scale: tuple[int, int] = (1, 5),
    ) -> Trial:
        """
        Render both arms and seal the mapping.

        The beat goes to one arm and zero to the other, chosen by
        :mod:`secrets` so that nothing in the visible state implies which. The
        renders are otherwise the same configuration and the same seed, so the
        progression, the wave events and the noise are identical between them.
        """
        fastest = (
            preset.beat.span[1] if isinstance(preset.beat, BeatCurve) else preset.beat
        )
        if fastest <= 0.0:
            msg = (
                f"a trial needs a preset with a beat to test; {preset.name!r} "
                f"never rises above 0 Hz, which is already the null condition"
            )
            raise ValueError(msg)
        if directory.exists() and any(directory.iterdir()):
            msg = f"{directory} already exists and is not empty"
            raise ValueError(msg)
        low, high = scale
        if low >= high:
            msg = f"rating scale must run low to high, got {scale!r}"
            raise ValueError(msg)

        directory.mkdir(parents=True, exist_ok=True)
        trial = cls(
            directory=directory,
            name=name,
            preset=preset.name,
            beat=preset.beat,
            base_hz=preset.base_hz,
            seed=preset.seed,
            minutes=preset.minutes,
            sample_rate=preset.sample_rate,
            scale=scale,
            created=_now(),
        )

        beat_arm = secrets.choice(ARMS)
        for arm in ARMS:
            beat = preset.beat if arm == beat_arm else 0.0
            arm_preset = preset.with_overrides(beat=beat)
            # PCM_16 WAV: uncompressed, so both files are the same length and
            # the format cannot give the arm away. See the module docstring.
            render_to_file(
                arm_preset.build_layers(),
                arm_preset.render_config(),
                trial.arm_path(arm),
                subtype="PCM_16",
            )

        # The arm is all this needs to hold. What the beat *is* already sits in
        # the visible state; which arm has it is the only secret.
        _seal({"beat_arm": beat_arm}, trial.sealed_path)
        trial.save()
        return trial

    @classmethod
    def load(cls, directory: Path) -> Trial:
        """Read a trial from its directory."""
        path = directory / _STATE
        if not path.is_file():
            msg = f"no trial at {directory} (expected {_STATE})"
            raise FileNotFoundError(msg)
        fields = json.loads(path.read_text(encoding="utf-8"))
        fields["scale"] = tuple(fields["scale"])
        fields["beat"] = _beat_from_json(fields["beat"])
        return cls(directory=directory, **fields)

    def save(self) -> None:
        """Write the visible configuration back out."""
        fields = {
            "name": self.name,
            "preset": self.preset,
            "beat": _beat_to_json(self.beat),
            "base_hz": self.base_hz,
            "seed": self.seed,
            "minutes": self.minutes,
            "sample_rate": self.sample_rate,
            "scale": list(self.scale),
            "created": self.created,
            "revealed_at": self.revealed_at,
        }
        self.state_path.write_text(
            json.dumps(fields, indent=2) + "\n", encoding="utf-8"
        )

    # -- logging ------------------------------------------------------------

    @property
    def sessions(self) -> tuple[Session, ...]:
        """Every session logged so far, in order."""
        if not self.log_path.is_file():
            return ()
        return tuple(
            Session(**json.loads(line))
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def log(self, arm: str, state: int, note: str = "") -> Session:
        """Append one rating."""
        if arm not in ARMS:
            msg = f"arm must be one of {ARMS}, got {arm!r}"
            raise ValueError(msg)
        low, high = self.scale
        if not low <= state <= high:
            msg = f"state must be between {low} and {high}, got {state!r}"
            raise ValueError(msg)

        session = Session(
            arm=arm, state=state, note=note, at=_now(), after_reveal=self.revealed
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(session)) + "\n")
        return session

    def counts(self) -> dict[str, int]:
        """Blind sessions logged per arm. Safe to show — it names no beat."""
        counted = dict.fromkeys(ARMS, 0)
        for session in self.sessions:
            if not session.after_reveal:
                counted[session.arm] += 1
        return counted

    def next_arm(self) -> str:
        """
        Which arm to play next, keeping the two balanced.

        Chosen rather than left to you, because choosing by mood is choosing by
        the thing being measured. Ties are broken by :mod:`secrets`, so a run
        of alternating sessions cannot be predicted either.
        """
        counted = self.counts()
        fewest = min(counted.values())
        candidates = [arm for arm, count in counted.items() if count == fewest]
        return secrets.choice(candidates)

    # -- unblinding ---------------------------------------------------------

    def reveal(self, *, acknowledged: bool = False, force: bool = False) -> Comparison:
        """
        Unblind, and report what the ratings do and do not support.

        Records that it happened, so anything logged afterwards is marked as no
        longer blind and left out of the comparison.
        """
        if not acknowledged:
            msg = (
                "revealing ends the blind for good. Pass acknowledged=True "
                "(`--i-am-done` on the command line) if that is what you want."
            )
            raise ValueError(msg)

        counted = self.counts()
        if not force and min(counted.values()) < MINIMUM_SESSIONS:
            msg = (
                f"only {counted[ARMS[0]]} and {counted[ARMS[1]]} sessions logged; "
                f"{MINIMUM_SESSIONS} per arm is the floor. Keep going, or pass "
                f"force=True (`--force`) if you are abandoning the trial."
            )
            raise ValueError(msg)

        sealed = _unseal(self.sealed_path)
        beat_arm = str(sealed["beat_arm"])

        if not self.revealed:
            revealed = replace(self, revealed_at=_now())
            revealed.save()

        return compare(self.sessions, beat_arm=beat_arm, scale=self.scale)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def sessions_for_effect(d: float) -> int:
    """
    Sessions per arm needed to detect an effect of size ``d``, roughly.

    The usual two-sample approximation for 80% power at a two-sided 5%
    threshold, ``n = 16 / d**2``, inflated by 5% for the efficiency a rank test
    gives up against a t-test on normal data. Rounded up, and not to be taken
    to more than one significant figure.
    """
    if d <= 0.0:
        msg = f"effect size must be positive, got {d!r}"
        raise ValueError(msg)
    return math.ceil(16.0 / (d * d) * 1.05)


@dataclass(frozen=True, slots=True)
class Comparison:
    """What the ratings support, and what they do not."""

    beat_arm: str
    null_arm: str
    beat_states: tuple[int, ...]
    null_states: tuple[int, ...]

    p_value: float | None
    cliffs_delta: float | None
    verdict: str
    excluded: int = 0

    @property
    def n_beat(self) -> int:
        """Blind sessions on the arm carrying the beat."""
        return len(self.beat_states)

    @property
    def n_null(self) -> int:
        """Blind sessions on the silent arm."""
        return len(self.null_states)

    @property
    def mean_beat(self) -> float:
        """Mean rating with the beat, or nan with no sessions."""
        return _mean(self.beat_states)

    @property
    def mean_null(self) -> float:
        """Mean rating without it."""
        return _mean(self.null_states)

    @property
    def difference(self) -> float:
        """Mean with the beat minus mean without."""
        return self.mean_beat - self.mean_null

    @property
    def sessions_for_large_effect(self) -> int:
        """How many sessions per arm a large effect would need."""
        return sessions_for_effect(0.8)

    @property
    def sessions_for_medium_effect(self) -> int:
        """How many a medium one would need."""
        return sessions_for_effect(0.5)


def _mean(values: Sequence[int]) -> float:
    return float("nan") if not values else sum(values) / len(values)


def _cliffs_delta(a: Sequence[int], b: Sequence[int]) -> float | None:
    """
    How often a beats b, minus how often b beats a, over all pairs.

    Reported instead of a standardised mean difference because these are
    ordinal ratings on a five-point scale, where the distance between 3 and 4
    is not something anyone can defend as equal to the distance between 4 and 5.
    """
    if not a or not b:
        return None
    above = sum(1 for x in a for y in b if x > y)
    below = sum(1 for x in a for y in b if x < y)
    return (above - below) / (len(a) * len(b))


def compare(
    sessions: Sequence[Session],
    beat_arm: str,
    scale: tuple[int, int] = (1, 5),
) -> Comparison:
    """
    Compare the two arms, and be honest about the answer.

    A two-sided Mann-Whitney U on the ratings — a rank test, because the
    ratings are ordinal and the samples are small — plus Cliff's delta as the
    effect size. Sessions logged after unblinding are excluded.
    """
    del scale
    null_arm = next(arm for arm in ARMS if arm != beat_arm)
    blind = [session for session in sessions if not session.after_reveal]
    excluded = len(sessions) - len(blind)

    beat_states = tuple(s.state for s in blind if s.arm == beat_arm)
    null_states = tuple(s.state for s in blind if s.arm == null_arm)

    p_value: float | None = None
    if beat_states and null_states and len(set(beat_states + null_states)) > 1:
        p_value = float(
            stats.mannwhitneyu(beat_states, null_states, alternative="two-sided").pvalue
        )

    delta = _cliffs_delta(beat_states, null_states)
    verdict = _verdict(len(beat_states), len(null_states), p_value, delta)

    return Comparison(
        beat_arm=beat_arm,
        null_arm=null_arm,
        beat_states=beat_states,
        null_states=null_states,
        p_value=p_value,
        cliffs_delta=delta,
        verdict=verdict,
        excluded=excluded,
    )


def _verdict(
    n_beat: int, n_null: int, p_value: float | None, delta: float | None
) -> str:
    """A sentence that does not oversell what a handful of sessions can show."""
    needed = sessions_for_effect(0.8)
    if not n_beat or not n_null:
        return (
            "Nothing to compare: one arm has no sessions. A trial needs ratings "
            "on both."
        )
    if p_value is None:
        return (
            "Every rating is identical, so there is nothing to test. Either the "
            "scale is too coarse for what you are noticing, or nothing is "
            "happening."
        )

    smallest = min(n_beat, n_null)
    strength = "" if delta is None else f" Cliff's delta is {delta:+.2f}."
    small = smallest < needed

    # The caveat is a different one in each direction, and saying the wrong one
    # reads as a contradiction. A significant result from a small sample is not
    # "undetectable" — it is real in these sessions, and likely to shrink when
    # you collect more, because a small study only reaches significance when
    # the effect it happened to see was on the large side.
    if p_value < 0.05:  # noqa: PLR2004 - the conventional threshold, as such
        caveat = (
            f" That is a real difference across these {n_beat + n_null} sessions, "
            f"but at {smallest} per arm an effect only clears the threshold when "
            f"it lands on the large side, so expect it to shrink as you collect "
            f"more. About {needed} per arm before it is worth much on its own — "
            f"a reason to keep going, not an answer."
            if small
            else ""
        )
        return (
            f"The ratings differ by more than chance alone would comfortably "
            f"explain (p = {p_value:.3f}).{strength}{caveat}"
        )

    caveat = (
        f" Which settles nothing: at {smallest} sessions per arm this trial "
        f"could not reliably have detected even a large effect, and would need "
        f"about {needed} per arm to. Absence of a difference here is not "
        f"evidence there isn't one."
        if small
        else ""
    )
    return (
        f"No difference beyond what chance would produce (p = {p_value:.3f})."
        f"{strength}{caveat}"
    )
