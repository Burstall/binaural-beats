# violet on a phone

A home-screen web app that plays music with violet's pulse applied live, adds a
binaural layer tuned to your base frequency, and keeps a blinded record of
whether any of it helps you. Hosted on Cloudflare's free tier: static files and
a small worker, music in R2, everything else in D1, sign-in by Cloudflare
Access.

This file is the contract between the three parts — the Python ingest
(`violet library`), the worker, and the app. Change it first.

## Layout

    web/
      wrangler.jsonc        worker, assets, D1 and R2 bindings
      migrations/           D1 schema (the source of truth for the tables)
      worker/               the API and the media proxy (ES modules)
      public/               the app: index.html, css, js/, sw.js, icons
      test/                 node --test, no dependencies
      package.json          scripts: test, dev, deploy (wrangler pinned)

## Sign-in

Every request reaches the worker through Cloudflare Access. Access puts a
signed JWT in the `Cf-Access-Jwt-Assertion` header; the worker verifies it
(RS256 against `https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`,
`aud` must contain `ACCESS_AUD`, `iss` must equal the team domain, `exp` in the
future; keys cached for an hour) and takes the person from its `email` claim.
It never trusts the plain `Cf-Access-Authenticated-User-Email` header.

- `ACCESS_TEAM_DOMAIN` and `ACCESS_AUD` are vars in `wrangler.jsonc`. While
  either is empty, `/api/*` and `/media/*` answer 503 `{"error": "access not
  configured"}`.
- For `wrangler dev` only: if `DEV_EMAIL` is set (in `.dev.vars`, which is
  ignored by git) and the request comes from localhost, that address is used
  and no JWT is required.
- When the Access session has expired, fetches come back as a redirect to the
  Access login or an opaque error. The app treats any non-JSON response or
  network failure on `/api/me` while online as "signed out" and shows a SIGN IN
  button that does a top-level navigation to `/`.

Friends need no Cloudflare account: the Access policy lists their email
addresses and they sign in with a one-time code sent to that address.

## API

JSON in and out. Errors are `{"error": "message"}` with a 4xx/5xx status.
Every response carries `Cache-Control: no-store` except `/media`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/me` | | `{email, profile}` — `profile` is the stored JSON or `null` |
| PUT | `/api/profile` | the profile JSON (≤ 64 KB) | `{ok: true, updated_at}` |
| GET | `/api/library` | | `{tracks: [Track], updated_at}` |
| PUT | `/api/tracks/:id/tags` | `{add: [tag], remove: [tag]}` | `{tags: [tag]}` |
| GET | `/api/ratings` | | `{ratings: {trackId: -1 or 1}}` |
| PUT | `/api/ratings/:trackId` | `{value: -1, 0 or 1}` (0 deletes) | `{ok: true}` |
| GET | `/api/playlists` | | `{playlists: [{id, name, track_ids, updated_at}]}` |
| PUT | `/api/playlists/:id` | `{name, track_ids}` | `{ok: true, updated_at}` |
| DELETE | `/api/playlists/:id` | | `{ok: true}` |
| GET | `/api/sessions?since=ISO` | | `{sessions: [Session]}` (default: last 365 days) |
| PUT | `/api/sessions/:id` | Session fields (upsert) | `{ok: true}` |
| GET | `/media/<r2 key>` | | the object, with `Range` support |

`Track` is a row of `tracks` with `tags: [string]` added, minus `r2_key`, plus
`url: "/media/<r2_key>"`. Field names are the column names.

`Session` is `{id, mode, started_at, minutes, arm, rating, note, track_ids}`.

Rules the worker enforces: tags match `^[a-z0-9][a-z0-9-]{0,23}$`, at most 32
per track; playlist names ≤ 80 chars, ≤ 2000 tracks; ids match
`^[A-Za-z0-9_-]{8,40}$`; a person can only read or change their own profile,
ratings, playlists and sessions. Unknown track ids in ratings or tags are 404.

### /media

Serves R2 objects under `tracks/` and `beds/` only. Honours `Range` (single
range: `bytes=a-b`, `bytes=a-`, `bytes=-n`) with 206 and `Content-Range`, 416
when unsatisfiable, and answers `HEAD`. Sets `Accept-Ranges: bytes`, the
object's content type and ETag, and `Cache-Control: private, max-age=31536000,
immutable` — keys under `tracks/` are content hashes and never change meaning.
iOS Safari will not play media from a server that ignores ranges; that was
measured, not assumed.

## The library

Audio lives in the R2 bucket `violet-music`:

- `tracks/<id>.<ext>` — the music, as uploaded by `violet library publish`.
  `<id>` is the first 16 hex digits of the SHA-256 of the file's bytes.
  Formats: MP3 or AAC (`.m4a`) only — what iOS plays natively.
- `beds/meditate.mp3` — violet's `meditate` preset rendered with beat 0: the
  chords, pedal and ocean, no binaural pair (the app adds that live, tuned to
  the listener). 20 minutes.
- `beds/ocean-loop.flac` — the ocean alone, one minute, closed into a loop
  with violet's 12 s fold, rendered from a one-off preset (`kind = "ocean"`,
  level 0.42, gain 12, beat 0, 32 kHz). Not the built-in `ocean-loop` preset:
  that one carries chords, a pedal and a theta binaural pair, which would
  fight every mode's own layers. Measured seam: 0.0004 across the wrap against
  a typical sample-to-sample step of 0.058. FLAC because MP3 and AAC pad the
  ends and would click at the loop.

### Tags

Lower-case, hyphenated. Auto tags come from the analyser at publish time and
from the source's own subjects; people add and remove tags in the app.

| Kind | Tags | Set by |
|---|---|---|
| Mode fit | `focus`, `create`, `meditate` | analyser, people |
| Rhythm | `beat`, `beatless` | analyser (`beat_strength`) |
| Tempo | `slow` (< 75 bpm), `mid` (75–105), `upbeat` (> 105) | analyser |
| Feel | `calm`, `lively`; `dark`, `bright` | analyser (energy, brightness terciles) |
| Voice | `vocals`, `instrumental` | people (and source subjects when explicit) |
| Genre | `lofi`, `chillhop`, `ambient`, `downtempo`, `jazz`, `piano`, `electronic`, `acoustic`, `classical`, … | source subjects |

Mode-fit rules for the analyser (people correct it):

- `focus`: a clear steady beat (`beat_strength` ≥ 0.35), 60–110 bpm, energy in
  the bottom two-thirds of the library.
- `create`: a beat at any tempo, or beatless with energy above the bottom
  third.
- `meditate`: beatless (`beat_strength` < 0.2) and energy in the bottom third.

A track can fit more than one mode, or none.

### Loudness

The analyser measures integrated loudness (ITU-R BS.1770-4, gated) and sample
peak. `gain_db` brings the track to −16 LUFS but never past a sample peak of
−1 dBFS. The app applies it, so every track plays at the same level and the
pulse's depth means the same thing on all of them.

## The profile

Owned by the app, stored opaque by the worker. Version 1:

```json
{
  "v": 1,
  "name": "Rich",
  "base_hz": 83.949,
  "modes": {
    "focus":    {"pulse_hz": 16, "depth": 0.35, "binaural_hz": 16, "binaural_level": 0.5,
                 "tags": ["focus"], "exclude": ["vocals"], "playlist": null},
    "create":   {"pulse_hz": 10, "depth": 0.2,  "binaural_hz": 10, "binaural_level": 0.5,
                 "tags": ["create"], "exclude": ["vocals"], "playlist": null},
    "meditate": {"binaural_from_hz": 10, "binaural_to_hz": 5, "glide_minutes": 12, "minutes": 20,
                 "binaural_level": 0.6, "music": false, "tags": ["meditate"], "playlist": null}
  },
  "background": {"kind": "off", "level": 0.4},
  "timer": {"work_minutes": 50, "break_minutes": 10, "cycles": 1},
  "crossfade_seconds": 6,
  "blind": {"enabled": true},
  "last_mode": "focus"
}
```

`binaural_level` is a 0..1 slider mapped to −40..−18 dB under the music; 0 is
silent rather than −40 dB. On
speakers the two tones meet in the air and beat audibly (a monaural beat); on
headphones each ear gets its own. Either way it is quiet. The carrier is the
base frequency shifted by whole octaves into 250–520 Hz — the same rule as
`violet.tuning.carrier_for` — so 83.949 Hz gives 335.796 Hz.

Missing fields take these defaults; unknown fields are kept, so an older app
never destroys a newer app's settings.

- `timer.work_minutes` of 0 means open-ended: no breaks, no end, no chime.
  Meditate ignores `timer` and runs for `modes.meditate.minutes`.
- `background.kind` is `off`, `ocean` (the bed above) or `noise` (soft pink
  noise generated in the browser as a seamless 20 s loop); `level` 0..1 maps
  to gain `level²`. The background keeps playing through breaks.

## The sound

One `AudioContext`, built inside the first tap (iOS requires it), with
`navigator.audioSession.type = "playback"` where supported.

    deck A: <audio> → MediaElementSource → norm gain → fade gain ─┐
    deck B: <audio> → MediaElementSource → norm gain → fade gain ─┤
                                                                   ├→ [pulse] → music gain ─┐
    binaural: two oscillators → ChannelMerger(L, R) → binaural gain ───────────────────────┤
    background: AudioBufferSource(loop) → background gain ─────────────────────────────────┤
    meditate bed: <audio> → MediaElementSource → bed gain ─────────────────────────────────┤
                                                                     master gain → limiter → destination

- **The pulse** is violet's, per deck so each track pulses on its own beat
  grid: `out = x − depth·m(t)·band(x)`, band = 2nd-order high-pass at 200 Hz
  then 2nd-order low-pass at 1 kHz (Q = 1/√2), `m(t) = (1 − cos 2πf(t − t₀))/2`
  from an oscillator with a cosine `PeriodicWave`, started at `t₀` = the
  track's first beat in context time. `f` is the mode's rate locked to the
  track's tempo exactly as `violet.dsp.modulation.tempo_locked` does (nearest
  of subdivisions 1,2,3,4,6,8,12,16,24,32 per beat by ratio); unlocked when
  the track has no bpm.
- **Crossfades** between decks over `crossfade_seconds`, equal-power. The next
  track is chosen and loaded when the current one has 30 s left, triggered
  from `timeupdate`, and the fade is scheduled on the audio clock.
- **Fades and the timer** are scheduled on the audio clock at session start,
  not with `setTimeout`, because timers are throttled while the phone is
  locked and audio is not. A break lowers the music gain to zero (the tracks
  keep moving underneath) and rings the chime; the next work block brings it
  back. The chime is synthesised: a few inharmonic partials with exponential
  decay.
- **Meditate** plays the bed (or, if the profile says `music: true`, tracks
  tagged `meditate`) with no pulse, and glides the binaural beat from
  `binaural_from_hz` to `binaural_to_hz` exponentially over `glide_minutes`,
  then holds. Web Audio integrates frequency into phase itself, so writing
  each ear's frequency curve with `setValueCurveAtTime` cannot produce the
  wrong-rate bug Stage 8 fixed in Python.
- **The blind test.** With `blind.enabled`, each session is assigned `on` or
  `off` with probability ½, by `crypto.getRandomValues`, before anything
  plays. `off` sets pulse depth and binaural level to zero; everything else is
  identical, including the loudness. The arm is stored with the session but
  not shown until the person rates it: PLAY hides the pulse and binaural
  readouts for blind sessions, and LOG shows only "blind" beside an unrated
  one. Honest limit: the binaural tone is audible, so a listener can often
  guess the arm. The test blinds the label, not the ear. Rating (1–5) is asked when a session
  ends or is stopped after at least 10 minutes. The History view shows, per
  mode, the mean rating in each arm, the difference, and a bootstrap 95%
  interval — and says plainly when there are too few sessions to say anything
  (fewer than 10 rated sessions in either arm). Skipping the rating is
  allowed; any session of 10 minutes or more can be rated later from LOG.
  `minutes` counts the whole session, breaks included; pausing stops it.
- **Shuffle** picks from tracks that have all of the mode's `tags`, none of its
  `exclude`, and no thumbs-down from this person; thumbs-up triples a track's
  weight; a track is not repeated within the session until all have played.

## Offline

A service worker caches the app shell (cache-first, versioned by `VERSION` in
`sw.js` — bump it whenever a shell file changes; `test/sw.test.js` checks the
list matches the files on disk, and leaves out `/index.html`, which redirects
to `/`) and the last
`/api/library` response (network-first). "Keep offline" on a playlist stores
its tracks whole in Cache Storage; the service worker then answers `/media/`
requests for them from the cache, slicing for `Range` itself and answering 206.
Offline sessions are queued in IndexedDB and sent when the app is next online.

## What an iPhone was measured to do

On iOS 26.6, as a home-screen app: a media element routed through Web Audio,
an `AudioBufferSourceNode`, and a plain `<audio>` element all kept playing for
4–6 minutes with the screen locked, the Web Audio graph's clock advancing in
step with the wall clock. Not yet measured: changing tracks while locked, and
the Access sign-in flow inside a home-screen app.

## Running it

    cd web
    npm install                     # wrangler, for dev and deploy only
    npm test                        # node --test; no dependencies
    echo DEV_EMAIL=you@example.com > .dev.vars
    npx wrangler d1 migrations apply violet --local
    npx wrangler dev                # http://localhost:8787
    npx wrangler deploy

The tests run the worker against `node:sqlite` loaded with the real
migrations, an in-memory R2, and RSA keys generated per run for the Access
JWTs, so the SQL and the crypto under test are the ones that ship.
