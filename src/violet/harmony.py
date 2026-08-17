"""
Chords as exact ratio sets, and the walk that moves between them.

Naming
------
Chords are keyed by roman numeral — ``i``, ``i7``, ``VI``, ``III``, ``iv``,
``VII`` — because that is what they are. The prototype spelled them ``Em``,
``C``, ``G``, ``Am``, ``D``, ``Em7``, which was true of the root it happened
to start from and false of every other root. A numeral is correct for any
root; a letter name is only correct for one. When you want letter names,
:meth:`Chord.note_labels` computes them from the root you are actually using.

Voicing
-------
Each voice is a degree of the five-limit just table in
:mod:`violet.tuning`, plus a whole number of octaves. Held as
:class:`~fractions.Fraction`, so a fifth is exactly 3/2 and stacking a third
on a third gives exactly a fifth.

The voicings span from the root itself up to 18/5 of it — a root in the
carrier window puts them between about 336 Hz and 1210 Hz. Low enough to stay
warm, high enough that the binaural percept holds. The prototype's comment
claimed a ceiling of 900 Hz, which two of the six chords exceed; the range is
a fifth wider at the top than that.

Just intonation matters more here than anywhere else in the package. These
are four sustained tones held for forty seconds at a time; equal-tempered
thirds beat against each other audibly over that long, while exact ratios
lock.

Movement
--------
A weighted transition graph, walked by a seeded generator. Each chord lasts
between 38 and 62 seconds and the walk never terminates on its own, so an
hour never repeats and the same seed always gives the same hour.

The order of the options in :data:`TRANSITIONS` is part of the seed contract.
Reordering the targets of a chord changes which one a given random draw
selects, and therefore changes every progression that passes through it.

Crossfades
----------
Equal power, not equal amplitude. The outgoing chord leaves on a cosine and
the incoming chord arrives on a sine over the *same* window, so
``g_out**2 + g_in**2`` is exactly 1 throughout. Fading one out linearly while
fading the other in linearly would dip by 3 dB in the middle of every
transition, which on a slow ambient piece is heard as a sag rather than as a
change of chord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from violet.tuning import JUST_SEMITONES, implied_reference, note_of

if TYPE_CHECKING:
    from collections.abc import Mapping
    from fractions import Fraction

    from violet._types import FloatArray

__all__ = [
    "CHORDS",
    "TRANSITIONS",
    "Chord",
    "ChordEvent",
    "WalkConfig",
    "crossfade_gain",
    "plan_progression",
    "validate_transitions",
]

#: A voice: a semitone of the just table, and how many octaves above the root.
Degree = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Chord:
    """A chord as a set of exact ratios above the root."""

    numeral: str
    degrees: tuple[Degree, ...]

    @property
    def ratios(self) -> tuple[Fraction, ...]:
        """Exact frequency ratios of the voices, relative to the root."""
        return tuple(
            JUST_SEMITONES[semitone] * 2**octave for semitone, octave in self.degrees
        )

    @property
    def voices(self) -> int:
        """How many tones this chord sounds."""
        return len(self.degrees)

    def frequencies(self, root_hz: float) -> tuple[float, ...]:
        """Voice frequencies for a given root."""
        return tuple(root_hz * float(ratio) for ratio in self.ratios)

    def note_labels(self, root_hz: float, a4: float | None = None) -> tuple[str, ...]:
        """
        Letter names of the voices, named against ``a4``.

        ``a4`` defaults to the reference implied by the root itself, which is
        what you want: it names the voices against the piece's own tuning, so
        the deviation you see is the just interval's own and not the root's
        offset from concert pitch added on top.

        These names are approximations by nature. A just minor seventh sits
        17.6 cents above the equal-tempered one, and no letter name carries
        that.
        """
        reference = implied_reference(root_hz) if a4 is None else a4
        return tuple(
            note_of(freq, reference).label for freq in self.frequencies(root_hz)
        )


def _chord(numeral: str, *degrees: Degree) -> Chord:
    return Chord(numeral=numeral, degrees=degrees)


#: The six chords of the prototype's progression, as scale degrees.
#:
#: Read as (semitone, octave). The comments give the letter names for a root
#: of E, which is where these voicings came from, and are a note about their
#: history rather than a claim about any particular render.
CHORDS: dict[str, Chord] = {
    # i        root, minor third, fifth, ninth            E  G  B  F#
    "i": _chord("i", (0, 0), (3, 0), (7, 0), (2, 1)),
    # i7       root, minor third, fifth, minor seventh    E  G  B  D
    "i7": _chord("i7", (0, 0), (3, 0), (7, 0), (10, 0)),
    # VI       major triad on the minor sixth, plus its seventh   C E G B
    "VI": _chord("VI", (8, 0), (0, 1), (3, 1), (7, 1)),
    # III      major triad on the minor third, plus its octave    G B D G
    "III": _chord("III", (3, 0), (7, 0), (10, 0), (3, 1)),
    # iv       minor seventh on the fourth                        A C E G
    "iv": _chord("iv", (5, 0), (8, 0), (0, 1), (3, 1)),
    # VII      major triad on the minor seventh, plus its octave   D F# A D
    "VII": _chord("VII", (10, 0), (2, 1), (5, 1), (10, 1)),
}

#: Where each chord can go, and how strongly it wants to. Order is part of
#: the seed contract — see the module docstring.
TRANSITIONS: dict[str, tuple[tuple[str, int], ...]] = {
    "i": (("VI", 3), ("III", 3), ("iv", 2), ("i7", 2)),
    "VI": (("III", 3), ("i", 3), ("iv", 2)),
    "III": (("i", 3), ("VII", 2), ("VI", 2), ("i7", 2)),
    "iv": (("i", 3), ("VI", 2), ("III", 2)),
    "VII": (("III", 3), ("i", 2)),
    "i7": (("VI", 3), ("iv", 2), ("III", 2)),
}


@dataclass(frozen=True, slots=True)
class WalkConfig:
    """How the walk moves, as opposed to where it can go."""

    #: Crossfade length in seconds, centred on the join between two chords.
    crossfade: float = 16.0

    #: How long a chord is held, before the crossfades either side of it.
    min_seconds: float = 38.0
    max_seconds: float = 62.0

    #: Where the walk starts. The tonic, unless you want otherwise.
    start: str = "i"

    #: The graph to walk. Replaceable, so a preset can supply its own.
    transitions: Mapping[str, tuple[tuple[str, int], ...]] = field(
        default_factory=lambda: TRANSITIONS
    )
    chords: Mapping[str, Chord] = field(default_factory=lambda: CHORDS)

    def __post_init__(self) -> None:
        if self.crossfade <= 0.0:
            msg = f"crossfade must be positive, got {self.crossfade!r}"
            raise ValueError(msg)
        if not 0.0 < self.min_seconds <= self.max_seconds:
            msg = (
                f"chord length must satisfy 0 < min <= max, got "
                f"min={self.min_seconds!r}, max={self.max_seconds!r}"
            )
            raise ValueError(msg)
        if self.min_seconds < self.crossfade:
            msg = (
                f"a chord held for {self.min_seconds:g} s cannot carry a "
                f"{self.crossfade:g} s crossfade at each end; it would still be "
                f"arriving when it starts to leave"
            )
            raise ValueError(msg)
        validate_transitions(self.transitions, self.chords, self.start)


@dataclass(frozen=True, slots=True)
class ChordEvent:
    """One chord, sounding from ``start`` to ``end``."""

    chord: Chord
    start: float
    end: float

    #: The first chord of a piece does not fade in — it is simply there — and
    #: the last does not fade out, because there is nothing to fade to.
    fade_in: bool = True
    fade_out: bool = True

    def support(self, crossfade: float) -> tuple[float, float]:
        """
        The window outside which this event contributes exactly nothing.

        Unbounded on whichever side does not fade: an event that never fades in
        was already sounding, and one that never fades out never stops. The
        bound is exact, not conservative, which is what lets a renderer skip
        events without the result depending on where the blocks fall.
        """
        lo = self.start - crossfade / 2.0 if self.fade_in else -math.inf
        hi = self.end + crossfade / 2.0 if self.fade_out else math.inf
        return lo, hi


def validate_transitions(
    transitions: Mapping[str, tuple[tuple[str, int], ...]],
    chords: Mapping[str, Chord],
    start: str,
) -> None:
    """
    Check that a transition graph can actually be walked.

    Every chord needs somewhere to go, every target has to exist, and weights
    have to be positive — a dead end would end the piece early and a missing
    target would end it with a ``KeyError`` forty minutes in.
    """
    if start not in chords:
        msg = f"start chord {start!r} is not in the chord table"
        raise ValueError(msg)

    for name, moves in transitions.items():
        if name not in chords:
            msg = f"transition table has {name!r}, which is not a known chord"
            raise ValueError(msg)
        if not moves:
            msg = f"chord {name!r} is a dead end: it has nowhere to go"
            raise ValueError(msg)
        for target, weight in moves:
            if target not in chords:
                msg = f"chord {name!r} moves to {target!r}, which does not exist"
                raise ValueError(msg)
            if weight <= 0:
                msg = f"weight for {name!r} -> {target!r} must be positive"
                raise ValueError(msg)

    missing = set(chords) - set(transitions)
    if missing:
        msg = f"chords with no transitions defined: {sorted(missing)}"
        raise ValueError(msg)


def plan_progression(
    total_seconds: float,
    rng: np.random.Generator,
    config: WalkConfig | None = None,
) -> tuple[ChordEvent, ...]:
    """
    Walk the graph for at least ``total_seconds``, returning the chord events.

    The walk overshoots by up to one chord length, so the last chord of a
    render is a chord in the middle of a progression rather than a chord that
    happens to stop. Nothing after the end is rendered; it exists so the final
    crossfade has something to fade towards.

    Two draws per chord, in this order: the length, then the next chord. That
    ordering is the seed contract.
    """
    walk = config or WalkConfig()
    events: list[ChordEvent] = []
    clock = 0.0
    current = walk.start

    while clock < total_seconds + walk.max_seconds:
        held = float(rng.uniform(walk.min_seconds, walk.max_seconds))
        events.append(
            ChordEvent(
                chord=walk.chords[current],
                start=clock,
                end=clock + held,
                fade_in=bool(events),
                fade_out=True,
            )
        )
        targets, weights = zip(*walk.transitions[current], strict=True)
        probabilities = np.array(weights, dtype=float)
        probabilities /= probabilities.sum()
        current = str(rng.choice(targets, p=probabilities))
        clock += held

    last = events[-1]
    events[-1] = ChordEvent(
        chord=last.chord,
        start=last.start,
        end=last.end,
        fade_in=last.fade_in,
        fade_out=False,
    )
    return tuple(events)


def crossfade_gain(
    t: FloatArray,
    event: ChordEvent,
    crossfade: float,
) -> FloatArray:
    """
    Gain of one chord event at absolute times ``t``.

    One inside a chord, a quarter-cosine on the way out, a quarter-sine on the
    way in, and exactly zero outside. The fade windows are centred on the
    joins, so an event's fade-out shares its window with the next event's
    fade-in and the two are complementary in power.
    """
    gain = np.ones_like(t)
    half = crossfade / 2.0

    if event.fade_in:
        rise_from = event.start - half
        rise_to = event.start + half
        rising = (t >= rise_from) & (t < rise_to)
        gain[rising] = np.sin((t[rising] - rise_from) / crossfade * np.pi / 2.0)
        gain[t < rise_from] = 0.0

    if event.fade_out:
        fall_from = event.end - half
        fall_to = event.end + half
        falling = (t >= fall_from) & (t < fall_to)
        gain[falling] = np.cos((t[falling] - fall_from) / crossfade * np.pi / 2.0)
        gain[t >= fall_to] = 0.0

    return gain
