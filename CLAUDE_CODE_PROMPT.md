# Claude Code bootstrap prompt

Copy everything below the line into Claude Code, in an empty directory.

Before you do: put the three prototype scripts (`binaural_e_violet.py`,
`binaural_layered.py`, `ocean_chords_e.py`) into a `reference/` subdirectory
so Claude Code can read them. They are the spec — the new code should
reproduce their output, not reinvent it.

---

I want to build a Python package for generating long-form binaural and
ambient audio, starting from three working prototype scripts in
`reference/`. Read all three carefully before writing any code — they
contain hard-won DSP decisions that must survive the refactor.

## Project

Name: `violet`. CLI entry point: `violet`. Use `uv` for everything.

The origin: a tone-test result of 83.949 Hz, note E, colour violet.
83.949 Hz is ~32 cents sharp of E2 at A440, implying a reference of
A = 448.2 Hz. Shifted up 43 octaves it lands at ~738 THz / 406 nm, which
is violet — hence the name. This mapping is the project's starting
constant, not a claim about physics.

## Non-negotiable technical constraints

These are the things the prototypes got right. Preserve them, and write
tests that fail if they regress.

1. **Absolute-time phase indexing.** Blocks are rendered from an absolute
   sample index, never a per-block reset. `t = arange(start, stop) / sr`.
   Any per-block `arange(0, n)` is a bug that produces a click at every
   block boundary.

2. **Block-size invariance.** Rendering the same config at block sizes of
   1 s and 10 s must produce near-identical output (allow tiny float
   drift; assert max abs difference < 1e-6 on the float signal before
   quantisation). Make this a test.

3. **One beat per mix, regardless of harmony.** Every tonal voice is a
   sine pair detuned by the *same* amount. A voice at ratio `r` off the
   root emits `r*root - beat/2` left and `r*root + beat/2` right. Never
   apply harmonic-rich waveforms to a binaural pair — harmonic *n* beats
   at *n × beat*, which smears the percept. Test: FFT both channels of a
   rendered chord, pair up the peaks, assert every pair differs by
   exactly the beat frequency.

4. **Filter state persists across blocks.** All IIR filtering uses
   `scipy.signal.lfilter` with `zi` carried forward per channel per
   filter. Noise must be continuous across block boundaries.

5. **The carrier notch.** The ocean layer has an IIR notch at the carrier
   frequency so broadband noise doesn't mask the beat.

6. **No ffmpeg dependency.** Use `soundfile` (libsndfile) to stream WAV
   and FLAC directly, writing block by block. The prototypes shelled out
   to ffmpeg; that goes away.

7. **Peak safety.** Assert rendered peak stays under 0.95 with no clipped
   samples, on every preset, in tests.

## Base frequency is a parameter, never a constant

The prototypes started life with 83.949 Hz hard-coded and were retrofitted
with `--base`; `reference/tuning.py` is the result and should be the model
for the real `tuning.py`. In the package, no module may contain a literal
base frequency. It enters as config and flows through: pedal drone,
carrier selection, chord root, and the ocean's notch frequency all derive
from it.

`auto_octaves()` picks the carrier by shifting the base in whole octaves
(preserving the note) into a 250-520 Hz window. Test it with awkward
inputs: 20 Hz, 2000 Hz, exactly 250 Hz, exactly 520 Hz.

Note that the chord dictionary keys are scale degrees spelled as if the
root were E. In the package, either use roman numerals (i, i7, VI, III,
iv, VII) or compute real note names from the root via `tuning.py`. Do not
carry the E-spelled labels forward as if they were absolute.

## Architecture

Aim for a small composable core, not a pile of scripts.

```
violet/
  tuning.py     frequency <-> note <-> cents <-> wavelength/colour;
                just-intonation ratio tables; tuning-reference inference
  harmony.py    chord definitions as ratio sets, weighted transition
                graph, seeded random walk, equal-power crossfade envelopes
  dsp/
    osc.py      sine pairs, pedal drones, phase accumulators
    noise.py    pink/brown generators with persistent state
    filters.py  butterworth, notch — thin typed wrappers over scipy
    env.py      fades, asymmetric swell envelopes, slow LFOs
  layers.py     Layer protocol: render(t) -> (left, right).
                Implementations: BinauralPair, ChordBed, Pedal, Ocean,
                IsochronicGate. Stateful layers declare it explicitly.
  engine.py     block-streaming renderer: takes layers + duration +
                sink, drives them, applies master fade and gain
  presets.py    named configs reproducing the three prototypes exactly
  cli.py        typer CLI
```

The `Layer` protocol is the key abstraction. Adding a new sound source
should mean writing one class, not touching the engine.

Config as frozen dataclasses. Presets loadable from TOML via `tomllib`
(stdlib) so users can add their own without editing Python.

## uv setup

```
uv init --package
uv add numpy scipy soundfile typer
uv add --dev pytest pytest-cov ruff mypy
```

Pin `requires-python = ">=3.12"`. Commit `uv.lock` and `.python-version`.
All commands documented in the README as `uv run ...` — never bare
`python`. Add a `Makefile` or `justfile` with `test`, `lint`, `typecheck`,
`render` targets.

## Testing

pytest, and please make these real property tests rather than smoke tests:

- beat frequency recovered by FFT, per voice, per preset
- block-size invariance (see constraint 2)
- determinism: same seed → identical bytes; different seed → different
  progression
- just-intonation ratios: assert `3/2` is 701.955 cents, not 700
- crossfade is equal-power: `g_a**2 + g_b**2 ≈ 1` through the transition
- noise continuity: no discontinuity spike at block boundaries (assert
  max sample-to-sample delta near a boundary is within distribution)
- peak/clipping safety on every preset
- a golden-file test: 5 seconds of each preset hashed, so refactors that
  change output are caught deliberately

Keep renders in tests short (2–5 s) and use small block sizes.

## CI

GitHub Actions: `astral-sh/setup-uv`, cache enabled, matrix on 3.12/3.13,
run ruff + mypy + pytest. Ruff configured strictly (line length 88,
isort, pydocstyle relaxed). Pre-commit hook running ruff.

## README

Document the DSP reasoning, not just the commands — why the beat lives in
the harmony, why noise is decorrelated, why just intonation. Include the
tuning maths for 83.949 Hz.

**State plainly and early that evidence for brainwave entrainment is
mixed, and make no health claims anywhere in the repo.** Frequency-to-
colour mapping is presented as an octave-arithmetic curiosity, not
physics. Add the epilepsy caution. MIT licence.

## Build order

Do not build everything at once. Stop after each stage and let me review.

1. `uv init`, package skeleton, CI, ruff/mypy passing on an empty package
2. `tuning.py` + tests — pure functions, no audio, fast to verify
3. `dsp/` + `engine.py` + a single `BinauralPair` layer, reproducing
   prototype 1. Block-invariance and beat-recovery tests passing.
4. `harmony.py` + `ChordBed`, reproducing prototype 3's tonal side
5. `Ocean` layer + notch, full prototype 3 parity
6. CLI + presets + README

---

# Roadmap — later milestones, do not build yet

## 1. Blinded self-experiment mode

The most valuable feature, and nearly free to implement. The null
condition is one parameter set to zero: render two files with identical
harmony, identical ocean, identical seed — one with `beat=4.0`, one with
`beat=0.0`. With zero detune there is no binaural beat at all, but the
files are otherwise indistinguishable to a listener.

`violet trial start` renders both, names them randomly A and B, and
writes the mapping to a sealed file the CLI won't display. You listen,
then `violet trial log A --state 4 --note "fell asleep quickly"`. After
N sessions, `violet trial reveal` unblinds and reports whether your
ratings differed by more than chance.

This turns the whole project from something you hope works into something
you can actually find out about. Design the CLI so unblinding early is
awkward.

## 2. Beat automation curves

Constant beat frequency is a limitation. A 45-minute descent — alpha at
10 Hz for 8 minutes, gliding to theta, settling at delta for the last
20 — is a far better sleep tool than any fixed rate.

This needs care: with time-varying frequency you cannot compute phase as
`2π f t`. You must integrate — `phase = 2π · cumsum(f) / sr` — with the
accumulator carried across blocks. Getting this wrong produces audible
frequency jumps at every boundary. Support piecewise-linear and
exponential curves, defined in the preset TOML.

## 3. Real-time engine

Swap the file sink for `sounddevice` and generate indefinitely. Because
the engine is already block-streaming and stateful, this is mostly a new
sink plus a callback. Then live parameter tweaking becomes possible —
adjust beat rate or ocean level and hear it move.

## 4. Spatial movement, and why it fights the beat

Slowly orbiting the chord voices around the head is tempting. Be warned
it's in direct conflict with the binaural beat: both use interaural phase
and time difference as their carrier, so spatialisation actively degrades
the pulse. Worth building as an explicit either/or mode with the tradeoff
documented, not as a default. An honest "these two cannot coexist" is a
better outcome than a muddy compromise.

## 5. Loudness normalisation

`pyloudnorm` to land every preset at the same LUFS target. Two-pass, or a
first-pass analysis render at reduced sample rate. Removes the manual
gain-constant fiddling in the prototypes.

## 6. Progression DSL

A tiny text format for hand-written progressions —
`Em:45 C:50 G:40 Am:55` — parsed into the same chord objects the random
walk produces. Then the random walk becomes one generator among several,
and hand-composed pieces become possible.

## 7. The violet companion

A generated HTML page whose background sits at the mapped colour,
brightness driven by the same wave envelope, exported alongside the
audio. Worth noting honestly in the code comments: 406 nm is outside the
sRGB gamut, so no screen can actually display it — you get the nearest
in-gamut approximation, and that limitation is more interesting than
pretending otherwise.

## 8. Isochronic and monaural modes

Amplitude-gated single tones, which survive speakers and lossy streaming
where binaural beats do not. A new `Layer` implementation, plus a
`--mode` flag. Useful for anywhere headphones aren't practical.
