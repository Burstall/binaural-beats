# Finding more music for violet

*As of 14 October 2026. Licences and terms change; check a source's own page
before relying on what this says about it.*

The app starts with 40 hours of Creative Commons music, but real lo-fi is the
gap: 32 of 543 tracks. The best ways to fill it are HoliznaCC0 for music you
can share with friends, and Bandcamp for music only you will hear — both added
with `violet library add`.

## What's in the library now

| Pool | Hours | What it is |
|---|---|---|
| Steady beats | 24.1 | Instrumental hip-hop, downtempo, trip-hop, some lo-fi |
| Ambient | 10.0 | Beatless ambient, piano, drone, neoclassical |
| Other calm | 6.0 | Jazz, acoustic, calm electronic |

543 tracks from 111 artists, all MP3, 4.3 GB in R2. By mode tag: 201 tracks fit
Deep Focus (13.7 h), 436 Creativity (31.6 h), 81 Meditate (6.7 h); a track can
fit more than one. Most of it (29 h) is CC BY-NC-SA.

Nobody has listened to all of it. Expect some tracks with vocals in the beats
pool — tag them `vocals` or give them a thumbs-down in the app and they drop
out of Deep Focus.

## Where to get more

| Source | Good for | Licence | How | The catch |
|---|---|---|---|---|
| [HoliznaCC0](https://freemusicarchive.org/music/holiznacc0) on the Free Music Archive | Real lo-fi, about 690 tracks | CC0 (public domain) on most; check each | Download by hand, then `violet library add` | FMA's terms forbid scraping, so no automatic fetch; some FMA tracks are ND or NC |
| [Bandcamp](https://bandcamp.com) | The best lo-fi there is; most lo-fi labels sell here | Personal use (you bought it) | Buy, download FLAC or MP3, `violet library add` | Only for you — see [Music you bought](#music-you-bought) |
| Internet Archive netlabels | More of what's already in the library | Mostly CC BY-NC-SA | `violet library fetch` | Thin on lo-fi; some uploads claim licences they don't have |
| [Kevin MacLeod / incompetech](https://incompetech.com/music/royalty-free/faq.html) | Calm, ambient, cinematic | CC BY 4.0 | Download by hand, `violet library add` | Attribution required (the app's Credits list does it) |
| iTunes Store purchases | Anything you already own there | Personal use | Convert to MP3, then `violet library add` | Apple Music streams are encrypted and can't be used at all |
| Generate it: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) or [Stable Audio 3 Small](https://huggingface.co/stabilityai/stable-audio-3-small-music) | Endless instrumentals in any style | Yours (MIT weights / community licence) | Runs on your own computer | Needs a GPU; check each take for vocals and odd artefacts |

Avoid anything marked "no derivatives", "YouTube/Twitch only" or "no copyright
music" without a real licence. Chillhop's creator programme, StreamBeats and Lofi
Girl all forbid what the app does, and Dusted Wax Kingdom, the best-known lo-fi
netlabel, is entirely ND.

## Adding music

1. Download the files into any folder. MP3, FLAC, WAV, Ogg and AIFF work; FLAC
   and the rest are re-encoded to MP3, because that is what an iPhone plays.
   AAC (`.m4a`, e.g. from iTunes) is refused — convert it first:

   ```sh
   ffmpeg -i song.m4a -c:a libmp3lame -q:a 2 song.mp3
   ```

2. Add them with their credit. One command can take a whole album:

   ```sh
   uv run violet library add downloads/holizna/*.mp3 \
       --artist "HoliznaCC0" --album "<album name>" \
       --license "CC0 1.0" \
       --license-url "https://creativecommons.org/publicdomain/zero/1.0/" \
       --source-url "https://freemusicarchive.org/music/holiznacc0" \
       --tag lofi --tag instrumental
   ```

   Each track is measured as it goes in — loudness, tempo, first beat, how
   clearly there's a beat — and tagged for the modes it suits.

3. Check what it decided, and correct it if you like:

   ```sh
   uv run violet library list --tag lofi
   uv run violet library tag 0123456789abcdef +focus -create
   ```

4. Publish. New files go to R2 and new rows to the database; running it again
   uploads nothing twice:

   ```sh
   uv run violet library publish
   ```

5. Open the app. The tracks are in LIBRARY straight away, and in shuffle for
   every mode whose tag they carry. Tags can be changed in the app too, by
   anyone signed in.

### More from the Internet Archive

```sh
uv run violet library fetch --hours 60
uv run violet library fetch --hours 5 --query 'collection:netlabels AND subject:"lofi"'
uv run violet library publish
```

`--hours` is the total you want in the pools, not how much to add, so 60 adds
about 20 to the current 40. With `--query` it fills a separate pool of that
size from your own search. It can be interrupted and picks up where it left
off.

## Licences

The app plays the original file and adds the pulse on your phone, which counts
as modifying it, and it plays to everyone you've allowed to sign in, which
counts as sharing it. So:

| Licence | Usable | Condition |
|---|---|---|
| CC0, Public Domain Mark | Yes | None |
| CC BY | Yes | Credit the artist — the app's Credits list does |
| CC BY-SA | Yes | Credit; share alike if you ever publish a modified file |
| CC BY-NC, BY-NC-SA | Yes | Non-commercial only. Friends, free, is fine; charging is not |
| Anything ND ("no derivatives") | No | The pulse is a derivative |
| "No copyright" with no real licence | No | Usually a rip someone had no right to give away |
| Apple Music, Spotify, YouTube Music | No | Encrypted; the app can't get at the sound |

### Music you bought

A Bandcamp or iTunes purchase is licensed to you for personal use. Putting it in
the shared library means everyone you've allowed to sign in can stream it,
which a personal licence doesn't cover. Right now the library has no private
tracks — every track is visible to everyone signed in. Until it does, keep
purchases out of the shared library if friends use the app, or ask for a
"private to me" flag on tracks; it's a small change to the worker and the
publish step.

## Internet Archive: what's worth searching

The archive's netlabel collections are where the starter library came from.
Labels that turned out well, by the identifier prefix the archive uses:

- **Instrumental hip-hop:** Fwonk (`FW…`), dystopiaq, true type (`20k…`), Lee
  Rosevere's lo-fi album.
- **Downtempo and trip-hop:** footnote (`foot…`), Take Pills (`tpd…`), Pueblo
  Nuevo (`pn…`), polygon, netwaves (`nws…`), Soisloscerdos (`slc…`).
- **Ambient, piano, neoclassical:** Musictrade (`mt…`), TFN, ansiform, Sirona,
  SCL (Kai Engel).
- **Calm instrumentals:** no-source (`NS…`), 12rec, weirdomusic (`WM…`),
  Incompetech, Kevin MacLeod's Classical Sampler.

And what to skip:

- The "meditation" subject — mostly spoken word: affirmations, guided
  meditations, lectures.
- "hip-hop" without "instrumental" — mostly rap.
- Uploads that claim a Creative Commons licence over a major-label mixtape, a
  YouTube rip or a cassette rip. The fetcher already refuses the ones it found;
  more will exist.

## Sources

- [Free Music Archive: HoliznaCC0](https://freemusicarchive.org/music/holiznacc0)
- [incompetech royalty-free FAQ](https://incompetech.com/music/royalty-free/faq.html)
- [Internet Archive advanced search](https://archive.org/advancedsearch.php)
- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
- [Stable Audio 3 Small](https://huggingface.co/stabilityai/stable-audio-3-small-music)
- [Creative Commons licence list](https://creativecommons.org/licenses/list.en)
