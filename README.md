# violet

Long-form binaural and ambient audio, generated from a single base frequency.
Streams to WAV or FLAC block by block, so an hour costs the same memory as five
seconds.

```sh
uv run violet render ocean --minutes 45 --out sleep.flac
uv run violet render ocean-loop --out loop.flac       # plays on repeat, no seam
uv run violet tune 83.949
```

## Read this first

**Evidence for brainwave entrainment is mixed.** Some studies find modest
effects on relaxation and self-reported anxiety; others find nothing beyond
what any steady ambient drone produces. The band names this project uses —
delta, theta, alpha — come from EEG and describe *measured brain activity*. The
assumption that a 4 Hz audio beat *drives* 4 Hz brain activity is the unproven
link, and nothing here should be read as asserting it.

**No health claims are made anywhere in this repository.** What is reliable:
these are pleasant things to sit inside, and a thirty-minute reason to lie still
with your eyes shut has value regardless of mechanism.

**If you have epilepsy or another seizure disorder, check with a doctor before
using rhythmic audio or visual stimulation.**

The project's own answer to the evidence question is on the roadmap: render two
files with identical harmony, identical noise and identical seed, one with
`beat = 4.0` and one with `beat = 0.0`. With no detune there is no binaural beat
at all, and the files are otherwise indistinguishable. That is a nearly perfect
placebo control, and it is already possible today —
`--beat 0` produces exactly it, verified by a test.

## Install

Everything runs through [uv](https://docs.astral.sh/uv/). No other setup, and
no ffmpeg.

```sh
uv sync                       # create the environment from uv.lock
uv run violet --help
```

Commands in this README are always `uv run ...`. There is no step where you
activate a virtualenv or call `python` directly.

## The starting point

A tone-test result: **83.949 Hz**, note **E**, colour **violet**. Everything
the package generates is derived from that one number, and it is a *parameter* —
pass `--base` and the whole piece transposes.

| Quantity | Value | How |
|---|---|---|
| Base frequency | 83.949 Hz | given |
| Nearest equal-tempered note | E2 = 82.407 Hz at A440 | `69 + 12·log₂(f/440)` → MIDI 40 |
| Deviation | **32.1 cents sharp** | `1200·log₂(83.949/82.407)` |
| Implied tuning reference | **A = 448.23 Hz** | `440 · 83.949/82.407` |
| Binaural carrier | **335.796 Hz** | base × 4 — up two octaves, same note |
| Shifted 43 octaves | 738.42 THz | `83.949 × 2⁴³` |
| As wavelength | **406.0 nm** | `c/f`, which is violet |
| Nearest sRGB | `#8100CC` | an approximation; see below |

```sh
uv run violet tune 83.949
uv run violet tune 136.10 --a4 432
```

### Why 32 cents sharp is not an error

83.949 Hz is not out of tune. It is *in* tune, at a reference of A = 448.23 Hz
rather than A = 440. Concert pitch is a convention that has moved around by
more than that within living memory. The package reports the implied reference
rather than the deviation as a defect, and when it names chord voices it names
them against the reference the base frequency itself implies — which is why the
chords come out spelled E, G, B, F# instead of something 32 cents wrong.

### The colour, honestly

Doubling an audio frequency 43 times lands it in the band we can see. That is
octave arithmetic and nothing else. There is no physical process connecting a
sound at 84 Hz to light at 406 nm, no mechanism is claimed, and the mapping is
a naming convention that happens to be pleasing.

Two further caveats, both real:

- **406 nm is outside the sRGB gamut.** No screen can display it. What
  `#8100CC` gives you is the nearest in-gamut approximation, and that
  limitation is more interesting than pretending otherwise.
- **The visible band is slightly narrower than an octave** (380–750 nm is a
  ratio of 1.974). So roughly one frequency in eighty has an octave series that
  steps clean over it and has no colour at all. `violet tune` says so rather
  than inventing one.

## The DSP, and why

Six decisions carry the whole design. Each one is enforced by a test, because
each one is easy to break by accident and hard to notice by ear.

### 1. The beat lives in the harmony, and there is only ever one

A binaural beat is not a sound. It is a percept, built in the superior olivary
complex from the *interaural phase difference* between two tones — one in each
ear, a few hertz apart. Nothing in the air beats; your brainstem does the
subtraction.

That has an immediate consequence for making the sound fuller. The obvious move
is a richer waveform, and it is wrong: **harmonic *n* of a pair detuned by *b*
beats at *n·b***. A sawtooth carrier produces beats at *b*, *2b*, *3b*
simultaneously, and the percept smears into mush.

So every tonal voice is a plain sine pair sharing the *same* detune. A voice at
ratio *r* off the root sounds `r·root − beat/2` in one ear and
`r·root + beat/2` in the other. Four-voice chords, an octave, a fifth, a whole
progression — every voice beats at exactly the same rate, and the harmony can
go wherever it likes without touching the pulse.

The test: render a chord, FFT both channels, pair the peaks low-to-high, assert
every pair differs by the beat frequency to within 0.02 Hz. It runs for all six
chords at three beat rates, and again mid-crossfade with eight tones per ear.

### 2. The carrier is not the base frequency

The percept weakens sharply below roughly 200 Hz, and most headphones roll off
hard down there too. 83.949 Hz is far too low to carry it.

So `auto_octaves()` shifts the base by *whole octaves* — preserving the note —
until it lands in a 250–520 Hz window: low enough to stay warm, high enough for
a strong percept, not fatiguing over forty minutes. 83.949 Hz goes up two
octaves to 335.796 Hz. 220 Hz goes up one to 440. 256 Hz is already in range
and stays put.

The base frequency is still physically present, as a mono pedal drone at the
literal pitch underneath everything.

### 3. Just intonation, not equal temperament

An equal-tempered major third is 400 cents. The just 5/4 is **386.31**. On four
tones sustained for forty seconds at a stretch, that 13.7-cent error is audible
as roughness — the partials beat against each other a few times a second.
Exact small-integer ratios lock instead: their partials coincide rather than
collide.

Ratios are held as exact `Fraction`s, so 3/2 is 3/2 and a minor third stacked
on a major third is *exactly* a fifth. There is a test asserting that 3/2 is
701.955 cents and not 700, because the difference is the entire point.

The six chords are spelled as roman numerals — `i`, `i7`, `VI`, `III`, `iv`,
`VII` — because a numeral is true for any root while a letter name is true for
exactly one. Letter names are computed from the root you actually used.

### 4. The ocean is decorrelated, and that is why it stays out of the way

The noise bed is two *independent* streams of pink noise, one per ear.
Uncorrelated noise **cannot carry a binaural beat**: the percept needs a
consistent phase relationship between the ears, and there isn't one. So the
ocean is texture, the harmony is the beat, and the two do not compete.

Two more things make the bed work:

- **A notch at the carrier.** Broadband noise sitting over 335.8 Hz masks the
  beat, so there is a narrow dip carved exactly there — 32 dB deep, and
  inaudible as a change in the noise's colour.
- **Asymmetric swells.** A wave arrives in a second or two and takes eight to
  drain away. An envelope that rises and falls at the same rate does not sound
  like water; it sounds like a siren. Sharp in, slow out, plus a very slow set
  cycle, because real surf comes in sets.

### 5. Absolute-time phase indexing

Blocks are rendered from an absolute sample index — `t = arange(start, stop) / sr`
— never from a per-block reset. A layer is handed a `Span` of *which* samples to
produce rather than *how many*, so `arange(0, n)` is not a mistake that can be
made: the layer does not know `n`.

Everything stateful carries its state forward explicitly: `lfilter` with `zi`
per channel per filter, and one random generator per noise stream. The
consequence is **block-size invariance**, and it is exact rather than
approximate — rendering any preset at 1-second and 10-second blocks gives
bit-identical output. Tests assert a difference below 1e-6 and in practice get
0.0.

This is not fussiness. A per-block reset is a click at every boundary, 360 of
them in an hour, and a shared random generator makes the noise depend on the
block size — which means the renderer cannot reproduce its own output.

### 6. Headroom, measured rather than hoped for

Layers declare an analytic upper bound on what they can contribute. The engine
sums those and scales the mix so the total lands on a ceiling, attenuating only
and never boosting. Noise has no true maximum, so its bound is measured across
seeds with margin on top. Every preset is tested for a peak under 0.95 with
zero clipped samples.

## Seamless loops

An ordinary render fades in and out. Played on repeat that is not a click, it is
a *hole* — three quarters of a minute where the piece drains away to nothing and
climbs back. Loop mode removes it.

```sh
uv run violet render ocean-loop --out loop.flac
uv run violet render tones --loop --minutes 10 --out tone-loop.flac
```

Two mechanisms, chosen per layer, because they are not interchangeable:

**Fixed frequencies are made exactly periodic.** A tone that does not complete
whole cycles over the loop arrives back at the join part-way through a cycle,
and that step is a click. Rounding each frequency to the nearest whole number of
cycles removes it — a five-minute loop moves the carrier by 0.009 cents, which
nothing hears. The beat is rounded to an *even* multiple so that both ears land
on whole cycles, not just the carrier. Every adjustment is reported:

```
note      chord voices snapped to whole cycles to close the loop
note      pedal 83.949 -> 83.95 Hz to close the loop
```

**The chord progression is planned to close.** Chord lengths are rescaled by a
few percent to tile the loop exactly, the final chord is drawn from the moves
that can reach the tonic, and the bed is evaluated modulo the loop so the chords
either side of the join are heard on both sides of it.

**Noise is folded.** It cannot be made periodic in a streaming renderer, so the
render continues past the end and the overhang is blended back over the opening
with equal-power gains. The first sample of the file *is* the continuation of
the last. This costs a second pass over the noise layers — about 30× realtime
instead of 47× — and memory bounded by the fold length, not the render length.

Why not just crossfade everything? Because crossfading a tone against a
phase-shifted copy of itself comb-filters: the two copies fight, and the voice
swells or cancels depending on the phase it happened to land on. Equal-power
gains are right for uncorrelated material and wrong for correlated material.
Each layer says which it is.

The result is measured two ways, because a loop can fail at two scales. The
step across the join is no larger than the steps in its own neighbourhood — no
click. And the short-term *level* across the join sits inside the range of level
changes happening inside the file — no lurch. A single-sample test cannot see
the second one: a file that stops mid-wave and restarts on flat water has two
perfectly ordinary samples either side of the join and still sounds wrong.

## Presets

```sh
uv run violet presets
uv run violet show ocean
```

| Preset | What it is |
|---|---|
| `tones` | One binaural pair over a mono sub drone. The minimal version. |
| `layered` | Root, octave and fifth as separate pairs, plus a noise floor. |
| `ocean` | Ocean bed, slow modal chords, continuous pedal. The long one. |
| `sleep` | The ocean at a delta rate, longer, with the surf closer. |
| `ocean-loop` | Five minutes of ocean that plays on repeat with no seam. |
| `tones-loop` | Ten minutes of tones, mathematically seamless. |

The first three reproduce the three prototype scripts in `reference/`
*sample for sample* — there are tests that render both and compare in int16
within two quantisation counts.

Presets are TOML, not Python, and the built-ins are read exactly the way yours
are. Copy any of them, change what you like, and pass the file:

```toml
[mine]
description = "My frequency, my chords."
base_hz = 136.10
beat = "alpha"
sample_rate = 32000
minutes = 40.0
seed = 12
gain = 1.25

[[mine.layer]]
kind = "chords"
level = 0.07

[[mine.layer]]
kind = "pedal"
level = 0.08

[[mine.layer]]
kind = "ocean"
level = 0.5
```

```sh
uv run violet render mine --presets mine.toml --out om.flac
```

Layer kinds are `pair`, `pedal`, `chords`, `ocean` and `air`. Ratios are
relative to the *carrier*, so a preset transposes correctly to any base; the
pedal is the exception and sounds the base itself. Reusing a built-in name
replaces it.

### Overrides

```sh
uv run violet render ocean --beat schumann --minutes 60 --seed 12
uv run violet render ocean --base 136.10 --out om.flac
uv run violet render ocean --beat 0 --out placebo.flac      # the null condition
```

Named bands: `delta` 2 Hz, `theta` 4, `schumann` 7.83, `alpha` 10, `beta` 16 —
or give a number.

Note that a seed changes nothing for the first half-minute of a render: every
progression starts on the tonic and holds it for at least 38 seconds, so two
seeds are bit-identical until the first crossfade.

## Playback

**Headphones, not speakers.** In a room the two tones mix in the air before they
reach you and the binaural percept does not occur at all — what you get instead
is a physical 4 Hz tremolo, which is a different thing. There's a test that
demonstrates exactly this by summing the channels to mono and recovering the
beat as amplitude modulation.

**Prefer local playback to streaming.** Lossy codecs use joint-stereo techniques
that can collapse the interaural phase information the effect depends on. The
ocean preset is more vulnerable than the plain tones, because broadband noise
gives the encoder more to spend bits on.

## Development

```sh
uv run pytest                  # 350-odd tests, about 40 seconds
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Or with [just](https://github.com/casey/just) (`uv tool install rust-just`):

```sh
just check                     # lint, typecheck, test
just render ocean "--minutes 5"
```

### Layout

```
violet/
  tuning.py     frequency <-> note <-> cents <-> wavelength; just ratios
  harmony.py    chords as ratio sets, weighted walk, equal-power crossfades
  dsp/
    osc.py      sine pairs
    noise.py    pink generators with persistent state
    filters.py  typed wrappers over scipy IIR, one state per channel
    env.py      fades, swells, slow LFOs
  layers.py     the Layer protocol: BinauralPair, Pedal, ChordBed, Ocean, Air
  engine.py     block-streaming renderer, sinks, loop closing
  presets.py    frozen config, TOML loading
  data/         presets.toml — where the base frequency actually lives
  cli.py        typer CLI
```

The `Layer` protocol is the load-bearing abstraction. Adding a sound source
means writing one class with a `render(span)` method; the engine never changes.
Layers that carry state declare it and implement `reset()`. Layers that can be
made periodic implement `snapped_to_loop()`, and the ones that can't simply
don't.

### Tests worth knowing about

Beyond the constraint tests already described: determinism (same seed → same
bytes), equal-power crossfades to 1e-12, noise continuity at block boundaries
measured against the distribution of interior steps, and a golden-file test that
hashes five seconds of every preset. A golden failure means the output changed —
whether that was intended is the question it asks. `uv run python
tests/regolden.py` updates the hashes when the answer is yes.

`tests/test_architecture.py` fails if any module in `src/violet` ever grows a
literal base frequency, in code, a default argument, a docstring or a comment.
A documented default is still a default. That is why 83.949 appears in this
README and in `data/presets.toml` and nowhere in the Python.

## Roadmap

1. **Blinded self-experiment mode.** `violet trial start` renders the beat and
   no-beat versions, names them randomly, and seals the mapping. You log your
   ratings; `violet trial reveal` unblinds and reports whether they differed by
   more than chance. Turns the whole project from something you hope works into
   something you can find out about. The mechanism already exists — `--beat 0` —
   so this is bookkeeping and a deliberately awkward path to unblinding early.
2. **Beat automation curves.** A 45-minute descent from alpha through theta to
   delta is a far better sleep tool than any fixed rate. Needs integrated phase
   (`phase += 2π·cumsum(f)/sr`) with the accumulator carried across blocks;
   `2πft` produces a jump at every boundary.
3. **Real-time engine.** Swap the file sink for `sounddevice`. The engine is
   already block-streaming and stateful, so this is a new sink plus a callback.
4. **Spatial movement, and why it fights the beat.** Orbiting the voices around
   the head uses interaural phase and time difference — exactly what the beat
   uses. They degrade each other. Worth building as an explicit either/or with
   the tradeoff documented, not as a default.
5. **Loudness normalisation.** `pyloudnorm` to land every preset on the same
   LUFS target, replacing the hand-tuned gain constants.
6. **Progression DSL.** `Em:45 C:50 G:40 Am:55` parsed into the same chord
   objects the walk produces, so the walk becomes one generator among several.
7. **The violet companion.** A generated page at the mapped colour, brightness
   driven by the same wave envelope. 406 nm is outside sRGB, so it will be an
   approximation, and that is worth saying on the page itself.
8. **Isochronic and monaural modes.** Amplitude-gated single tones, which
   survive speakers and lossy streaming where binaural beats do not.

## Licence

MIT. See [LICENSE](LICENSE).

The three prototype scripts this package was refactored from are kept unchanged
in `reference/`. They are the specification, and the parity tests compare
against them directly rather than against anyone's memory of what they did.
