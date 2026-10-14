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

The project's own answer to the evidence question is built in: see
[Finding out for yourself](#finding-out-for-yourself).

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
| `descent` | 45 minutes gliding from alpha through theta down to delta. |
| `meditate` | 20 minutes from alpha to theta, over the ocean. |
| `focus-music` | Your own music, modulated at 16 Hz. Needs `--music FILE`. |
| `deep-focus` | Slow chords over a pedal, no beat, modulated at 16 Hz. |
| `creativity` | The same bed, lighter: 10 Hz at a gentle depth. |

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

Layer kinds are `pair`, `pedal`, `chords`, `ocean`, `air` and `track`. Ratios are
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

## A beat that moves

A fixed rate is a compromise: fast enough to feel like something is happening,
slow enough to sleep to, and therefore neither. The `descent` preset holds alpha
for eight minutes, glides down through theta over the next seventeen, and spends
the last twenty at delta.

```sh
uv run violet render descent --out tonight.flac
```

```toml
beat = { shape = "exponential", at_minutes = [0, 8, 25, 45], hz = [10, 10, 2, 2] }
```

Times are in minutes and the key says so. `linear` moves by a constant number of
hertz per second; `exponential` moves by a constant *ratio*, which is how pitch
is heard — halving from 8 Hz to 4 sounds like the same size of step as halving
from 4 to 2, where a straight line loiters at the top and then plunges.

### The trap

A tone whose frequency changes is **not** `sin(2π·f(t)·t)`. That expression says
"the phase now is the frequency now times the whole elapsed time", which is only
true if the frequency has always been that value. Phase is the *integral* of
frequency.

What makes this dangerous is how it fails. Not with a click — for a smooth glide
the naive expression is perfectly smooth too, which is exactly why it survives a
listen and ships. Differentiating its phase gives an instantaneous rate of
**`f(t) + t·f′(t)`**, so the error grows with elapsed time and with how fast the
curve is moving:

| at | you asked for | naive gives |
|---|---|---|
| 10 s | 8.67 Hz | 7.33 Hz |
| 30 s | 6.00 Hz | 2.00 Hz |
| 40 s | 4.67 Hz | −0.67 Hz |
| 60 s | 2.00 Hz | −6.00 Hz |

Past the halfway point it goes negative, which swaps which ear leads. Forty-five
minutes of that isn't a descent, it's a rate nobody chose. Where the curve does
have a *step* in it, the naive form clicks as well — seventeen times the largest
legitimate sample-to-sample move. Both failure modes have a test.

### Why a closed form, not an accumulator

The usual fix is `phase += 2π·f/sr` carried across blocks. That works, and it
makes the oscillator stateful, which costs three things this package has been
careful to keep: it must be rendered strictly in order, it accumulates rounding
error over an hour, and rendering at two block sizes stops giving identical
samples — summing a million numbers in chunks of a thousand isn't the same
arithmetic as summing them in chunks of ten thousand.

A piecewise-linear or piecewise-exponential curve has an integral in closed
form, so it's evaluated directly from absolute time like everything else here.
No state, no drift, and block-size invariance that's exact rather than close.

Verified on the real 45-minute render: the beat measured by FFT at ten points
from minute 1 to minute 44 tracks the curve to within **0.007 Hz**.

### One thing it won't do

A moving beat and a seamless loop are refused together. The rate at the end of a
descent isn't the rate at the start, so however smooth the waveform is at the
join, the pulse jumps. An honest "these two cannot coexist" beats a muddy
compromise — a *flat* curve loops fine.

## A pulse you can hear on a speaker

Everything above puts its rate *between* the ears. Amplitude modulation puts it
in the loudness instead: a band of the mix dips and recovers many times a
second, the same in both ears. It needs no headphones, it survives lossy
streaming, and it can be laid over any music at all — including music you
already like.

This is what Brain.fm sells, and they say outright that they do not use
binaural beats. Their evidence is one paper: Woods et al. (2024),
*Communications Biology* 7:1376. What it actually shows:

- **Who:** two of the authors work for Brain.fm and hold equity; the company
  made the music tested. Funded by an NSF small-business grant. No independent
  replication found at the time of writing.
- **What helped:** 16 Hz modulation, confined to roughly 200 Hz–1 kHz and
  aligned to the music's beat, improved a sustained-attention task — mostly for
  participants who rated themselves high on ADHD symptoms. The key interaction
  came in at p = 0.048.
- **What did not:** varying modulation *depth* had no significant effect at
  all. The whole-brain fMRI group comparison was not significant. In the first
  experiment the benefit depended on hearing the modulated track first.
- **Caveats:** 21–48% of online participants were excluded, and the early
  experiments' comparison tracks also differed in key and instrumentation.

Promising for a subgroup, not settled for anyone. Which is exactly what the
trial below is for.

```sh
uv run violet render focus-music --music track.flac --bpm 84 --minutes 50
uv run violet modulate track.flac --bpm 84            # one file in, one out
uv run violet render deep-focus --out work.flac         # no music needed
uv run violet render ocean --depth 0.3 --hz beta        # modulate any preset
```

### How it is built

`Modulated` is a layer that wraps other layers. To the engine it is one more
layer; the engine did not change. It sums what it wraps, isolates the
200 Hz–1 kHz band with a fourth-order Butterworth whose state is carried block
to block like every other filter here, and subtracts a raised-cosine fraction
of that band:

    out = x - depth * m(t) * band(x)

Two reasons for the band. Modulate the bass and a kick drum starts pumping;
modulate the top and the hats flutter. Both sound like the music breaking rather
than like texture. And it is what the study did.

Two reasons for writing it as a subtraction rather than `rest + gain * band`:
at `depth = 0` it subtracts exactly zero, so **the placebo is the original,
sample for sample** — not a reconstruction of it, and there is a test that
compares bytes. That is what makes a blinded trial of the modulation exact.

The pulse is at full level on every cycle boundary. With `--bpm`, the rate is
moved to the nearest duple or triple subdivision of the tempo — 16 Hz at 84 bpm
becomes 16.8, twelve pulses to the beat — so the drum hits land at the top of a
pulse and the dips fall between them. `--offset` moves the first pulse onto the
track's first downbeat if it doesn't start on one. The rate can also move over
a render, with the same closed-form phase integral the beat curves use.

`Track` streams WAV, FLAC, MP3 or Ogg from disk, a block at a time. WAV, FLAC
and MP3 seek to the exact sample, so the layer is stateless for them. Ogg
Vorbis does not — libsndfile lands within a packet of the target, which an
early version of this layer turned into samples up to 0.47 apart at different
block sizes — so Ogg is read straight through and a jump is refused rather than
misread. It renders at the file's own sample rate rather than resampling,
repeats to fill a longer session, and measures its own peak instead of
estimating one.

### Headroom, measured

The analytic bound for a modulated mix is pessimistic: a band-pass can ring past
its input by up to its impulse response's L1 norm, about 1.76 here, and the bound
has to assume that happens against the full signal at the bottom of a pulse. On
a mastered track that would cost nearly 6 dB for nothing. So a modulated preset
without an explicit gain is rendered once into nothing, its real peak measured,
and attenuated only as far as that peak needs. Rendering is deterministic, so
the second pass produces exactly the samples the first one measured. In a
trial, the gain is decided once from the treated arm and used for both, so
loudness cannot give the arm away.

### What it won't do

A modulated mix cannot be a seamless loop. The band filter's memory can't be
made periodic by rounding a frequency, and crossfading the whole mix would
comb-filter every tone in it. The presets refuse the combination and say why.

The `creativity` preset's rate and depth are a guess. Nothing published says
what suits creative work; it is a lighter touch than `deep-focus` because a
heavier one is more intrusive, and the study's own 8 Hz condition did no better
than no modulation. Trial it before believing it.

### Music to put under it

Use what you like listening to; that matters more than anything in this
section. For files you can freely modify, the licences worth knowing (as of
October 2026, so check before relying on them):

- **CC0 / public domain lo-fi** — HoliznaCC0 on the Free Music Archive has
  several hundred tracks. FMA mixes CC0 and BY with NC and ND licences, so
  check each track. The site has no API and its terms forbid scraping.
- **CC BY** — Kevin MacLeod (incompetech.com) explicitly allows modification;
  attribution required. Calm and ambient rather than lo-fi.
- **Internet Archive netlabels** — a searchable API with a licence field.
  Filter out anything `nd`.
- **Generate it** — ACE-Step 1.5 (MIT weights, outputs usable commercially,
  runs in under 4 GB of VRAM) or Stable Audio 3 Small (community licence) make
  multi-minute instrumentals locally.

Avoid anything "no derivatives", and anything licensed only for YouTube or
Twitch use — Chillhop's creator programme and StreamBeats both forbid
modification.

## Finding out for yourself

The null condition is one parameter set to zero. Two files from the same
configuration, the same seed, the same progression, the same waves, the same
noise — one with `beat = 4.0` and one with `beat = 0.0`. With no detune there is
no interaural difference and so no binaural beat at all, and everything else
about the two files is identical. The ocean beds are *bit-identical*, and
there's a test asserting it. That is as close to a placebo as audio gets.

```sh
uv run violet trial start sleep-4hz --preset descent --minutes 45
uv run violet trial next sleep-4hz          # tells you which to play tonight
uv run violet trial log sleep-4hz A --state 4 --note "fell asleep quickly"
uv run violet trial status sleep-4hz        # progress, giving nothing away
uv run violet trial reveal sleep-4hz --i-am-done
```

Four things the design is careful about, each for a reason:

- **The assignment doesn't come from the seed.** It comes from `secrets`. Drawn
  from the render seed, you could recover it by re-running the walk, and every
  visible byte in the trial directory would tell you the answer.
- **Trial renders are WAV, not FLAC.** A compressed format leaks the arm through
  its own file size: with no detune the two channels of a tonal preset are
  identical, FLAC's mid/side stage reduces the difference channel to nothing,
  and the null file comes out **half the size**. Measured: 0.50 for `tones`,
  0.71 for `layered`, 0.91 for `ocean`. One glance at a directory listing and
  the trial is over. Uncompressed, both files are byte-for-byte the same length.
- **Which arm to play is chosen for you.** Left to pick freely you'll choose by
  mood, and mood is exactly what's being measured.
- **Unblinding is deliberately awkward.** `reveal` wants `--i-am-done`, refuses
  below six sessions per arm without `--force`, and records that it happened —
  anything logged afterwards is marked as no longer blind and excluded.

And one thing it is honest about: **at realistic session counts this cannot
settle anything.** Detecting even a *large* effect with a rank test at 80% power
takes about 27 sessions per arm; a medium one takes 68. `reveal` says so, and
tailors the caveat to the direction — a significant result from a small sample
gets told it will shrink with more data, a null result gets told that absence of
a difference isn't evidence there isn't one. It also can't stop you looking at
the numbers, stopping because they're interesting, and then quoting the p-value
as though the stopping rule had been fixed in advance. Decide how many sessions
you're doing before you start.

## The phone app

`web/` is violet as a home-screen app: music with the pulse applied live in the
browser, a binaural layer tuned to your base frequency underneath, a work/break
timer, playlists, offline playback, and a blind test that flips a coin each
session and only tells you which arm you got after you've rated it. Three
modes: **Deep Focus** (16 Hz pulse), **Creativity** (10 Hz, lighter), and
**Meditate** (no pulse; a binaural glide from alpha to theta over the plain
bed).

It runs on Cloudflare's free tier — a worker, R2 for the audio, D1 for
everything else — and Cloudflare Access handles sign-in, so a friend needs
only an email address you've allowed. Each person gets their own settings,
playlists, ratings and history; the library and its tags are shared.
[`web/README.md`](web/README.md) is the contract between the app, its worker
and `violet library`, and records what an iPhone was measured to do.

```sh
cd web
npm test                       # node --test, no dependencies
npx wrangler dev               # local, with DEV_EMAIL in .dev.vars
npx wrangler deploy
```

The binaural layer on a speaker is not the binaural percept — the two tones
meet in the air and beat as a monaural tremolo, which is a different thing (see
[Playback](#playback)). On headphones each ear gets its own tone. The blind
test switches the pulse and the binaural layer together, and it is honest about
its limit: the binaural tone is audible, so it hides the label from you, not
the sound.

### Music for the app

`violet library` gathers, measures and publishes what the app plays.

```sh
uv run violet library fetch --hours 40      # Creative Commons netlabels, Internet Archive
uv run violet library add track.mp3 --title ... --artist ...     --license "CC BY 4.0" --license-url ... --source-url ...
uv run violet library list --tag focus
uv run violet library tag 0123456789abcdef +vocals -focus
uv run violet library publish               # new files to R2, rows to D1
```

`fetch` fills three pools — steady-beat instrumentals, beatless ambient and
piano, other calm instrumentals — keeping only CC0, public domain, BY, BY-SA,
BY-NC and BY-NC-SA, never ND, and only tracks whose measured beat suits the
pool. Spoken word, rap, DJ mixes and rips with false licence claims are turned
away by keyword; the Internet Archive has all four. `add` takes music you own
(MP3 as is; FLAC, WAV, Ogg or AIFF re-encoded).

Every track is measured once: BS.1770-4 loudness, checked against EBU Tech
3341, and a playback gain to −16 LUFS so the pulse's depth means the same on
every track; tempo and first beat, so the pulse locks to the music; how clearly
there is a beat at all; how busy and how bright. Then it is tagged by the rules
in `web/README.md`. `publish` is safe to run again, and a tag someone removed in
the app stays removed. Everything lives in `library/`, which git ignores.

[docs/finding-music.md](docs/finding-music.md) says where to get more, which
licences the app can use, and how to add what you find.

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
uv run pytest                  # 753 tests, a few minutes
uv run pytest --cov            # 99% covered, with a 97% floor that CI enforces
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
    curves.py   beat rates that move, and their phase integral in closed form
    modulation.py  amplitude modulation: rate, depth, band, tempo locking
    osc.py      sine pairs
    noise.py    pink generators with persistent state
    filters.py  typed wrappers over scipy IIR, one state per channel
    env.py      fades, swells, slow LFOs
  layers.py     the Layer protocol: BinauralPair, Pedal, ChordBed, Ocean, Air,
                Track, and Modulated, which wraps the others
  engine.py     block-streaming renderer, sinks, loop closing
  presets.py    frozen config, TOML loading
  trial.py      blinded trials: sealed assignment, logging, honest statistics
  analysis.py   loudness (BS.1770-4), tempo, first beat, beat strength, feel
  archive.py    Internet Archive search, licence parsing, downloads
  library.py    the local music library, auto tags, publishing to R2 and D1
  library_cli.py  `violet library`
  data/         presets.toml — where the base frequency actually lives
  cli.py        typer CLI
web/
  worker/       Access sign-in, the JSON API, range-serving media from R2
  public/       the app: vanilla ES modules, a service worker, no build step
  migrations/   the D1 schema
  test/         node --test
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

`tests/test_validation.py` covers the error paths on purpose. Every message in
it is one somebody will read at three in the morning wondering why their render
stopped, and a validation branch that has never run is a branch that might not
raise, or might raise the wrong thing.

`tests/test_architecture.py` fails if any module in `src/violet` ever grows a
literal base frequency, in code, a default argument, a docstring or a comment.
A documented default is still a default. That is why 83.949 appears in this
README and in `data/presets.toml` and nowhere in the Python.

## Roadmap

1. ~~**Blinded self-experiment mode.**~~ Done — see
   [Finding out for yourself](#finding-out-for-yourself).
2. ~~**Beat automation curves.**~~ Done — see [A beat that moves](#a-beat-that-moves).
3. **Real-time engine.** Swap the file sink for `sounddevice`. The engine is
   already block-streaming and stateful, so this is a new sink plus a callback.
4. **Spatial movement, and why it fights the beat.** Orbiting the voices around
   the head uses interaural phase and time difference — exactly what the beat
   uses. They degrade each other. Worth building as an explicit either/or with
   the tradeoff documented, not as a default.
5. **Loudness normalisation.** Land every preset on the same LUFS target,
   replacing the hand-tuned gain constants. `violet.analysis` now measures
   BS.1770-4 loudness for the music library, so the measuring half exists.
6. **Progression DSL.** `Em:45 C:50 G:40 Am:55` parsed into the same chord
   objects the walk produces, so the walk becomes one generator among several.
7. **The violet companion.** A generated page at the mapped colour, brightness
   driven by the same wave envelope. 406 nm is outside sRGB, so it will be an
   approximation, and that is worth saying on the page itself.
8. ~~**Isochronic and monaural modes.**~~ Superseded by amplitude
   modulation — see [A pulse you can hear on a speaker](#a-pulse-you-can-hear-on-a-speaker).
9. ~~**A phone player.**~~ Done — see [The phone app](#the-phone-app).

## Licence

MIT. See [LICENSE](LICENSE).

The three prototype scripts this package was refactored from are kept unchanged
in `reference/`. They are the specification, and the parity tests compare
against them directly rather than against anyone's memory of what they did.
