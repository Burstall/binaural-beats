"""
Parity with the frozen prototypes.

The three scripts in ``reference/`` are the specification. These tests render
the same configuration through both the prototype and the package and compare
the samples, so a refactor that changes the sound has to be a deliberate act
rather than an accident.

Comparison is in int16, because that is what the prototypes write. The
prototypes quantise with ``(x * 32767).astype(int16)``, which truncates;
libsndfile scales by 32768 and rounds to nearest. Those two disagree by up to
about one and a half counts on a full-scale sample, so the tolerance is two
counts — six parts in a hundred thousand, and far tighter than any DSP error
would be.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from violet import tuning
from violet.engine import RenderConfig, loop_length, render_to_file
from violet.harmony import plan_progression
from violet.layers import BinauralPair, ChordBed, Layer, Pedal

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

SR = 44100
ORIGIN_HZ = 83.949

# Levels and gains that live inside the prototypes as local variables, so
# there is nothing to import them from. They move into presets at stage 6.
OCEAN_SR = 32000
OCEAN_VOICE_LEVEL = 0.070
OCEAN_PEDAL_LEVEL = 0.080
OCEAN_MASTER_GAIN = 1.25
OCEAN_FADE_SECONDS = 22.0
OCEAN_FADE_DENOMINATOR = 5

#: One int16 count is 1/32767 of full scale; two is the quantisation
#: disagreement between the prototype's truncation and libsndfile's rounding.
TOLERANCE_COUNTS = 2


def read_int16(path: Path, sample_rate: int = SR) -> np.ndarray:
    audio, rate = sf.read(path, dtype="int16", always_2d=True)
    assert rate == sample_rate
    return np.asarray(audio, dtype=np.int64)


def prototype_one_layers(beat: float, level: float, drone: float) -> list[Layer]:
    return [
        BinauralPair(carrier=tuning.carrier_for(ORIGIN_HZ), beat=beat, level=level),
        Pedal(freq=ORIGIN_HZ, level=drone),
    ]


@pytest.mark.parametrize(
    ("beat", "level", "drone", "seconds"),
    [
        # The script's own defaults.
        (4.0, 0.30, 0.09, 3.0),
        # A named band, a louder mix, and the drone switched off.
        (7.83, 0.35, 0.10, 2.5),
        (10.0, 0.30, 0.0, 2.0),
        # Long enough that the 8-second fade cap stops biting.
        (2.0, 0.30, 0.09, 40.0),
    ],
)
def test_prototype_one_parity(
    tmp_path: Path,
    reference_binaural: ModuleType,
    beat: float,
    level: float,
    drone: float,
    seconds: float,
) -> None:
    carrier = tuning.carrier_for(ORIGIN_HZ)

    reference_path = tmp_path / "reference.wav"
    reference_binaural.render(
        str(reference_path),
        carrier,
        beat,
        seconds,
        drone,
        level,
        False,  # noqa: FBT003 - the prototype's positional `loop` argument
        base=ORIGIN_HZ,
    )

    ours_path = tmp_path / "violet.wav"
    render_to_file(
        prototype_one_layers(beat, level, drone),
        RenderConfig(SR, duration=seconds, block_seconds=10.0, ceiling=0.89),
        ours_path,
        subtype="PCM_16",
    )

    theirs = read_int16(reference_path)
    ours = read_int16(ours_path)
    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) <= TOLERANCE_COUNTS


def test_prototype_one_loop_mode_parity(
    tmp_path: Path, reference_binaural: ModuleType
) -> None:
    """Loop mode: no fades, and the length snapped to whole beat cycles."""
    beat, level, drone = 4.0, 0.30, 0.09
    asked = 3.1
    carrier = tuning.carrier_for(ORIGIN_HZ)

    reference_path = tmp_path / "reference.wav"
    snapped = reference_binaural.render(
        str(reference_path),
        carrier,
        beat,
        asked,
        drone,
        level,
        True,  # noqa: FBT003 - the prototype's positional `loop` argument
        base=ORIGIN_HZ,
    )
    assert snapped == pytest.approx(loop_length(asked, beat))

    ours_path = tmp_path / "violet.wav"
    render_to_file(
        prototype_one_layers(beat, level, drone),
        RenderConfig(
            SR,
            duration=loop_length(asked, beat),
            block_seconds=10.0,
            fade_seconds=0.0,
            ceiling=0.89,
        ),
        ours_path,
        subtype="PCM_16",
    )

    theirs = read_int16(reference_path)
    ours = read_int16(ours_path)
    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) <= TOLERANCE_COUNTS


def test_headroom_rule_differs_from_the_prototype_at_high_levels(
    tmp_path: Path, reference_binaural: ModuleType
) -> None:
    """
    A deliberate divergence, pinned so it cannot happen by accident.

    The prototype bounds the peak at ``2 * level + drone``, which assumes both
    binaural tones land in the same ear. They do not — that is what makes them
    binaural. Each channel carries one tone plus the drone, so the true bound
    is ``level + drone``, and the prototype attenuates mixes that never needed
    it. At the shipped levels the two rules agree, both leaving the gain at
    1.0; they only part company on a loud mix.
    """
    beat, level, drone, seconds = 4.0, 0.5, 0.2, 2.0
    carrier = tuning.carrier_for(ORIGIN_HZ)

    reference_path = tmp_path / "reference.wav"
    reference_binaural.render(
        str(reference_path),
        carrier,
        beat,
        seconds,
        drone,
        level,
        False,  # noqa: FBT003 - the prototype's positional `loop` argument
        base=ORIGIN_HZ,
    )

    ours_path = tmp_path / "violet.wav"
    result = render_to_file(
        prototype_one_layers(beat, level, drone),
        RenderConfig(SR, duration=seconds, block_seconds=10.0, ceiling=0.89),
        ours_path,
        subtype="PCM_16",
    )

    theirs = read_int16(reference_path).astype(np.float64)
    ours = read_int16(ours_path).astype(np.float64)

    prototype_gain = 0.89 / (2 * level + drone)
    assert result.gain == 1.0
    assert prototype_gain < 1.0

    # Same signal, different level: scaling one to the other reconciles them.
    # The tolerance is wider than elsewhere because this comparison passes
    # through two independent quantisations — theirs at the attenuated level,
    # ours at full — and then rescales one of them, which compounds both.
    assert np.max(np.abs(ours * prototype_gain - theirs)) <= 4.0

    # And ours is still safe without the unnecessary attenuation.
    assert result.peak < 0.95
    assert result.clipped == 0


# ---------------------------------------------------------------------------
# prototype 3, tonal side (stage 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("beat", "seed", "seconds"),
    [
        (4.0, 5, 20.0),
        (7.83, 12, 30.0),
        # Long enough to cross a chord change and its 16-second crossfade.
        (2.0, 5, 60.0),
    ],
)
def test_prototype_three_tonal_parity(
    tmp_path: Path,
    reference_ocean: ModuleType,
    beat: float,
    seed: int,
    seconds: float,
) -> None:
    """
    Pedal plus chords, with the ocean turned all the way down.

    ``--ocean 0`` multiplies the noise bed by zero, which leaves the
    prototype's tonal side alone and gives stage 4 an exact target. The
    ocean's own generator draws from a separate seed, so silencing it does not
    disturb the progression.
    """
    reference_path = tmp_path / "reference.wav"
    _chords, root = reference_ocean.render(
        str(reference_path),
        beat,
        seconds,
        0.0,  # ocean level
        seed,
        base=ORIGIN_HZ,
    )
    assert root == pytest.approx(tuning.carrier_for(ORIGIN_HZ))

    events = plan_progression(seconds, np.random.default_rng(seed))
    layers: list[Layer] = [
        ChordBed(events=events, root=root, beat=beat, level=OCEAN_VOICE_LEVEL),
        Pedal(freq=ORIGIN_HZ, level=OCEAN_PEDAL_LEVEL),
    ]

    ours_path = tmp_path / "violet.wav"
    render_to_file(
        layers,
        RenderConfig(
            OCEAN_SR,
            duration=seconds,
            block_seconds=10.0,
            gain=OCEAN_MASTER_GAIN,
            fade_seconds=OCEAN_FADE_SECONDS,
            fade_max_denominator=OCEAN_FADE_DENOMINATOR,
        ),
        ours_path,
        subtype="PCM_16",
    )

    theirs = read_int16(reference_path, OCEAN_SR)
    ours = read_int16(ours_path, OCEAN_SR)
    assert ours.shape == theirs.shape
    assert np.max(np.abs(ours - theirs)) <= TOLERANCE_COUNTS
