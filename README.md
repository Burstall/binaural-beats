# violet — seed bundle

Three working prototypes for generating long-form binaural and ambient
audio, plus a bootstrap prompt for turning them into a proper package.

Everything here runs with no setup beyond `uv`. Each script carries PEP 723
inline dependency metadata, so `uv run` resolves numpy/scipy on first use
and caches them.

```sh
uv run reference/ocean_chords_e.py --minutes 30 --out session.wav
```

## The starting point

A tone-test result: **83.949 Hz**, note **E**, colour **violet**. The
maths, since it's the constant everything else is built on:

| Quantity | Value | Note |
|---|---|---|
| Base frequency | 83.949 Hz | |
| Nearest note (A440) | E2 = 82.407 Hz | 32.1 cents flat of the base |
| Implied tuning reference | A = 448.2 Hz | why it reads sharp against A440 |
| Binaural carrier used | 335.796 Hz | base × 4, same note, usable register |
| Shifted 43 octaves | 738.4 THz | |
| As wavelength | 406.0 nm | violet, just inside the visible range |

The colour mapping is octave arithmetic, not physics — doubling a
frequency 43 times happens to land in the visible band. It's a pleasing
coincidence, not a mechanism. Treated here as a naming convention.

## The three prototypes

All three default to the 83.949 Hz brief and all three accept `--base` to
use any other frequency. See "Using a different frequency" below.

### 1. `reference/binaural_e_violet.py` — plain tones

The minimal viable version. A single binaural pair plus a mono sub drone.

```sh
uv run reference/binaural_e_violet.py --beat theta --minutes 20 --out theta.wav
uv run reference/binaural_e_violet.py --beat 7.83 --minutes 45 --loop --out sch.wav
```

Named bands: `delta` 2 Hz, `theta` 4 Hz, `schumann` 7.83 Hz, `alpha` 10 Hz,
`beta` 16 Hz. `--loop` drops the fades and snaps the length to a whole
number of beat cycles for seamless repeat.

**Why the carrier isn't 83.949 Hz.** The binaural percept is formed from
interaural phase difference and weakens sharply below roughly 200 Hz; most
headphones also roll off hard down there. So the binaural pair sits two
octaves up at 335.796 Hz — the same note — and 83.949 Hz is present as a
mono drone underneath.

### 2. `reference/binaural_layered.py` — harmonically fuller

Adds an octave and a fifth as separately detuned pairs, plus a
low-passed pink noise bed. Upper layers breathe on slow independent
cycles (70 s and 110 s).

```sh
uv run reference/binaural_layered.py --beat theta --minutes 15 --out layered.wav
```

**Why layers rather than a richer waveform.** Harmonic *n* of a pair
detuned by *b* beats at *n × b*. A sawtooth carrier therefore produces
beats at *b*, *2b*, *3b* simultaneously and the percept smears. Every
layer is instead its own sine pair sharing the same detune, so the
harmony thickens while the beat stays single.

### 3. `reference/ocean_chords_e.py` — generative long-form

The full piece: ocean bed, slow modal chord movement, continuous pedal.

```sh
uv run reference/ocean_chords_e.py --minutes 60 --seed 12 --out sleep.wav
uv run reference/ocean_chords_e.py --beat schumann --minutes 45 --out ideas.wav
uv run reference/ocean_chords_e.py --beat delta --minutes 60 --ocean 0.6 --out deep.wav
```

`--ocean` runs from about 0.2 (distant surf) to 0.7 (close-breaking).
Past that it swamps the chords. Every `--seed` gives a different
progression and a different set of waves.

Three design decisions worth preserving:

- **83.949 Hz never moves.** It's a pedal tone under every chord, which
  makes C, Am and G read as modal colour over E rather than as key
  changes — the same device as a tanpura drone.
- **Just intonation, not equal temperament.** Voices sit at exact ratios
  (6/5, 3/2, 8/5) off the root so they lock instead of grinding.
  Equal-tempered thirds have audible roughness on sustained drones.
- **The ocean is notched at the carrier.** Broadband noise over 335.8 Hz
  would mask the beat, so there's a narrow dip carved exactly there. You
  won't hear the notch; the pulse stays legible under the surf.

The ocean is also **decorrelated** between channels, which means it
cannot carry a binaural beat — the percept needs correlated input at both
ears. So the ocean is texture only and the beat lives entirely in the
harmony. That division is deliberate.

## Using a different frequency

The 83.949 Hz brief is a **default, not a hard-coded constant**. Every
script takes `--base`, and `reference/tuning.py` does the maths for any
frequency you give it:

```sh
uv run reference/tuning.py 136.10
uv run reference/tuning.py 136.10 --a4 432     # name against another reference
```

```
base frequency      136.1000 Hz
nearest note        C#3 = 138.591 Hz (A4 = 440)
deviation           31.4 cents flat
implied A4 ref      432.09 Hz
suggested carrier   272.200 Hz (+1 octaves, same note)
colour octave       +42 octaves = 598.57 THz
wavelength          500.8 nm
approx screen hex   #00FF88  (out of sRGB gamut - approximation only)
```

Then render with it — all three scripts accept `--base`, and each prints
the tuning summary before it renders:

```sh
uv run reference/binaural_e_violet.py --base 136.10 --beat alpha --minutes 20 --out om.wav
uv run reference/binaural_layered.py  --base 220    --minutes 15 --out a3.wav
uv run reference/ocean_chords_e.py    --base 136.10 --minutes 45 --out om_ocean.wav
```

### How the carrier is chosen

You almost never want the base frequency itself as the binaural carrier —
it's usually too low. `auto_octaves()` shifts it by whole octaves (so the
note is unchanged) until it lands in a 250–520 Hz window: low enough to
stay warm, high enough for a strong binaural percept, and not fatiguing
over a long session. 83.949 Hz shifts up 2 octaves to 335.796 Hz; 220 Hz
shifts up 1 to 440 Hz; 256 Hz is already in range and stays put.

Override with `--octaves N`, or set the carrier directly with `--carrier`
on script 1.

### Two caveats when changing the base

- **Chord names become relative.** The progression keys in
  `ocean_chords_e.py` are scale degrees spelled as if the root were E,
  because that's what the brief supplied. The harmony transposes correctly
  — the ratios are all relative to the root — but a chord printed as `C`
  actually means "the major triad a minor sixth above your root". In roman
  numerals the progression is i, i7, VI, III, iv, VII, which is accurate
  for any root.
- **The colour changes.** The 43-octave shift that lands 83.949 Hz on
  violet gives a different wavelength for a different base — 136.10 Hz
  comes out green, 440 Hz orange. `tuning.py` finds the right octave count
  automatically, and gives the nearest in-gamut hex with the honest
  warning that spectral colours can't actually be displayed.

## Playback

**Headphones, not speakers.** In a room the two tones mix in the air
before reaching you and the binaural effect simply does not occur.

Prefer local playback over streaming. Lossy codecs use joint-stereo
techniques that can collapse the interaural phase information the effect
depends on — and the ocean piece is more vulnerable than the plain tones,
because broadband noise gives the encoder more to spend bits on.

## Next step

`CLAUDE_CODE_PROMPT.md` is a bootstrap prompt for turning these into a
proper `uv` package with tests and CI. It treats these three scripts as
the spec and includes a staged build order plus a roadmap.

The roadmap item I'd do first is the blinded trial mode. Setting
`beat = 0.0` removes the binaural beat entirely while leaving harmony,
ocean and seed identical — an almost perfect placebo control, available
essentially for free. It lets you find out whether the beat does anything
for you, rather than assuming either way.

## Honest note on the evidence

Research on brainwave entrainment is mixed. Some studies find modest
effects on relaxation and anxiety; others find nothing beyond what any
steady ambient drone produces. The band names (delta, theta, alpha) come
from EEG and describe measured brain activity — the assumption that a
4 Hz audio beat *drives* 4 Hz brain activity is the unproven link.

No health claims are made anywhere in this repository. What is reliable:
these are pleasant things to sit inside, and a 30-minute reason to lie
still with your eyes shut has value regardless of mechanism.

If you have epilepsy, check with a doctor before using rhythmic audio or
visual stimulation.

## Licence

MIT. See `LICENSE`.
