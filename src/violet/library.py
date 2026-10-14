"""
The music library: files on disk, a manifest, tags, and publishing to the app.

A library is a folder (``library/`` by default, ignored by git) holding
``tracks/<id>.<ext>`` and ``library.json``, the manifest. Each entry carries
every column of the app's ``tracks`` table (see ``web/migrations``), its tags
and who set them, and where the file came from. ``id`` is the first 16 hex
digits of the SHA-256 of the stored file, so a file cannot be added twice and
its R2 key never changes meaning.

Tracks come in two ways:

- :func:`fetch` gathers Creative Commons music from the Internet Archive
  (:mod:`violet.archive`), several hours at a time, measuring each track and
  keeping it only if it fits the pool it was fetched for;
- :func:`add_file` takes a file you have, a purchase say, with its credit.

:func:`publish` uploads new files to R2 and upserts rows into D1, both through
``wrangler``. Analysis columns are rewritten on every publish; tags are only
written for tracks D1 has not seen before, so a tag someone removed in the app
stays removed. Tag edits made here to tracks already in D1 are queued and
applied by the next publish.

The rules that turn measurements into tags are the ones ``web/README.md``
specifies; :func:`auto_tags` is their only implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from violet.analysis import NO_BEAT, Analysis, analyse
from violet.archive import (
    UNSUITABLE,
    Http,
    Item,
    Pool,
    RemoteTrack,
    interleave,
    item_tracks,
    search_all,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

MANIFEST = "library.json"
FETCH_STATE = "fetch-state.json"

#: What ``tracks.mime`` may say, by extension. Both play natively on iOS.
MIME = {".mp3": "audio/mpeg", ".m4a": "audio/mp4"}

#: What :func:`add_file` will convert to MP3 on the way in.
TRANSCODABLE = (".flac", ".wav", ".ogg", ".aiff", ".aif")

#: The contract's tag shape.
TAG = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")

#: Beat strength at or above which a track has ``beat``.
BEAT = 0.35

#: Track lengths the fetcher keeps, in seconds.
LENGTHS = (90.0, 1200.0)

#: Fetch caps, in seconds: no one item or artist dominates the library.
ITEM_CAP = 1.5 * 3600
ARTIST_CAP = 2.0 * 3600

#: Tags the analyser owns. Recomputed freely until a track is in D1.
ANALYSER_TAGS = frozenset(
    {
        "focus",
        "create",
        "meditate",
        "beat",
        "beatless",
        "slow",
        "mid",
        "upbeat",
        "calm",
        "lively",
        "dark",
        "bright",
    },
)

#: Source subjects that map to a genre tag. A subject matches a phrase when
#: the phrase appears in it as whole words.
GENRES: dict[str, tuple[str, ...]] = {
    "lofi": ("lofi", "lo-fi", "lo fi", "lofibeats", "lofihiphop"),
    "chillhop": ("chillhop", "chill hop", "jazzhop", "jazz hop", "jazzy hip hop"),
    "hip-hop": (
        "instrumental hip-hop",
        "instrumental hip hop",
        "hip hop",
        "hip-hop",
        "boom bap",
        "beat tape",
    ),
    "ambient": ("ambient",),
    "drone": ("drone",),
    "downtempo": (
        "downtempo",
        "down-tempo",
        "down tempo",
        "chillout",
        "chill out",
        "chill-out",
    ),
    "trip-hop": ("trip-hop", "trip hop", "triphop"),
    "jazz": ("jazz",),
    "lounge": ("lounge", "easy listening"),
    "piano": ("piano",),
    "electronic": ("electronic", "electronica", "idm"),
    "acoustic": ("acoustic", "guitar"),
    "classical": ("classical", "neo-classical", "neoclassical"),
    "instrumental": ("instrumental",),
}


def any_of(name: str, *terms: str) -> str:
    """An Archive search clause: ``name`` matches any of ``terms``."""
    quoted = (f'"{t}"' if " " in t else t for t in terms)
    return f"{name}:({' OR '.join(quoted)})"


AUDIO = "mediatype:audio"
NETLABELS = "collection:netlabels"

#: The starter library: three pools, in hours out of forty. ``ACCEPT`` is
#: checked against each downloaded track's analysis, so a "beats" query that
#: turns up a beatless interlude does not get to count it.
DEFAULT_POOLS = (
    Pool(
        "beats",
        24.0,
        (
            " AND ".join(
                (
                    AUDIO,
                    any_of(
                        "subject",
                        "instrumental hip-hop",
                        "instrumental hip hop",
                        "chillhop",
                        "jazzhop",
                        "jazz hop",
                        "lofi hip hop",
                        "lo-fi hip hop",
                        "lofibeats",
                        "boom bap",
                        "beat tape",
                        "chill beats",
                    ),
                )
            ),
            " AND ".join(
                (
                    AUDIO,
                    any_of("subject", "lofi", "lo-fi"),
                    any_of(
                        "subject",
                        "instrumental",
                        "beats",
                        "hip hop",
                        "hip-hop",
                        "chillhop",
                    ),
                )
            ),
            " AND ".join(
                (
                    AUDIO,
                    NETLABELS,
                    any_of(
                        "subject",
                        "downtempo",
                        "trip-hop",
                        "trip hop",
                        "triphop",
                        "chillout",
                        "chill out",
                        "nu jazz",
                        "acid jazz",
                    ),
                )
            ),
        ),
        "steady-beat instrumentals: the focus and creativity pool",
    ),
    Pool(
        "ambient",
        10.0,
        (
            " AND ".join(
                (
                    AUDIO,
                    NETLABELS,
                    any_of("subject", "ambient", "drone"),
                    any_of(
                        "subject",
                        "calm",
                        "relaxing",
                        "meditation",
                        "meditative",
                        "soothing",
                        "peaceful",
                        "minimal",
                        "sleep",
                    ),
                )
            ),
            " AND ".join(
                (
                    AUDIO,
                    any_of(
                        "subject",
                        "meditation",
                        "relaxing music",
                        "sleep music",
                        "meditation music",
                    ),
                    "licenseurl:*creativecommons*",
                )
            ),
            " AND ".join(
                (
                    AUDIO,
                    NETLABELS,
                    any_of(
                        "subject",
                        "piano",
                        "solo piano",
                        "neoclassical",
                        "neo-classical",
                    ),
                )
            ),
        ),
        "beatless ambient, piano and drone: the meditate pool",
    ),
    Pool(
        "calm",
        6.0,
        (
            " AND ".join(
                (
                    AUDIO,
                    any_of(
                        "creator",
                        "Kevin MacLeod",
                        "Lee Rosevere",
                        "Chad Crouch",
                        "Podington Bear",
                        "Kai Engel",
                        "Chris Zabriskie",
                        "Blue Dot Sessions",
                        "Komiku",
                        "Lobo Loco",
                        "Scott Holmes",
                    ),
                )
            ),
            " AND ".join(
                (
                    AUDIO,
                    NETLABELS,
                    any_of("subject", "jazz", "lounge", "acoustic"),
                    any_of("subject", "instrumental"),
                )
            ),
        ),
        "other calm instrumentals: jazz, acoustic, electronic",
    ),
)

#: The tempo range a "beats" track must fall in to count: the focus range,
#: with a little room above for the creativity end.
BEATS_BPM = (60.0, 115.0)

#: The tempo range that fits ``focus``, per the contract.
FOCUS_BPM = (60.0, 110.0)


def _accept_beats(a: Analysis) -> bool:
    if a.bpm is None or a.beat_strength < BEAT:
        return False
    return BEATS_BPM[0] <= a.bpm <= BEATS_BPM[1]


def _accept_ambient(a: Analysis) -> bool:
    return a.beat_strength < BEAT - 0.1


def _accept_any(_: Analysis) -> bool:
    return True


#: Whether a measured track belongs in the pool it was fetched for.
ACCEPT: dict[str, Callable[[Analysis], bool]] = {
    "beats": _accept_beats,
    "ambient": _accept_ambient,
    "calm": _accept_any,
}


def now() -> str:
    """The time, as the manifest and D1 write it: ISO 8601 UTC to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """
    One track in the library.

    The first block of fields is the ``tracks`` row, less ``energy``, which is
    a rank within the whole library and so only computed when publishing.
    ``energy_raw`` is what :mod:`violet.analysis` measured.
    """

    id: str
    file: str
    title: str
    artist: str
    album: str | None
    duration: float
    bytes: int
    mime: str
    loudness: float | None
    peak: float
    gain_db: float
    bpm: float | None
    beat_offset: float | None
    beat_strength: float
    energy_raw: float
    brightness: float
    license: str
    license_url: str
    source_url: str
    attribution: str
    added_at: str
    tags: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    uploaded_at: str | None = None
    in_d1: bool = False
    pending_add: list[str] = field(default_factory=list)
    pending_remove: list[str] = field(default_factory=list)

    @property
    def r2_key(self) -> str:
        """Where the file lives in the bucket."""
        return f"tracks/{self.id}{Path(self.file).suffix}"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Entry:
        """Read a manifest entry, ignoring fields this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def credit(title: str, artist: str, license_name: str, source_url: str) -> str:
    """The line a credits page prints for a track."""
    return f'"{title}" by {artist} ({license_name}) - {source_url}'


@dataclass
class Library:
    """A library folder and its manifest."""

    root: Path
    entries: dict[str, Entry] = field(default_factory=dict)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @classmethod
    def open(cls, root: Path) -> Library:
        """Open ``root``, creating it if needed; no manifest is an empty library."""
        (root / "tracks").mkdir(parents=True, exist_ok=True)
        library = cls(root)
        path = root / MANIFEST
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for raw in data.get("tracks", []):
                entry = Entry.from_json(raw)
                library.entries[entry.id] = entry
        return library

    def save(self) -> None:
        """Write the manifest, atomically: a crash leaves the old one whole."""
        with self._lock:
            ordered = sorted(self.entries.values(), key=lambda e: (e.added_at, e.id))
            text = json.dumps(
                {"version": 1, "tracks": [asdict(e) for e in ordered]},
                indent=1,
                ensure_ascii=False,
            )
            temporary = self.root / (MANIFEST + ".tmp")
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(self.root / MANIFEST)

    def add(self, entry: Entry) -> None:
        """Add an entry (thread-safe)."""
        with self._lock:
            self.entries[entry.id] = entry

    def path(self, entry: Entry) -> Path:
        """The entry's file on disk."""
        return self.root / entry.file

    def find(self, prefix: str) -> Entry:
        """The one entry whose id starts with ``prefix``."""
        matches = [e for e in self.entries.values() if e.id.startswith(prefix.lower())]
        if len(matches) != 1:
            what = "no track" if not matches else f"{len(matches)} tracks"
            msg = f"{what} with an id starting {prefix!r}"
            raise KeyError(msg)
        return matches[0]

    def seconds(self, *, pool: str | None = None) -> float:
        """Total length, optionally of one fetch pool only."""
        return sum(
            e.duration
            for e in self.entries.values()
            if pool is None or e.provenance.get("pool") == pool
        )

    def has_source(self, identifier: str, name: str) -> bool:
        """Whether a file from the Archive is already here."""
        return any(
            e.provenance.get("identifier") == identifier
            and e.provenance.get("file") == name
            for e in self.entries.values()
        )


def file_id(path: Path) -> str:
    """The first 16 hex digits of the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def transcode_to_mp3(source: Path, dest: Path) -> None:
    """
    Re-encode anything soundfile can read to MP3, in 4096-frame chunks.

    Chunked because writing a whole track to MP3 in one call overflows the
    stack inside libsndfile's encoder on Windows (0xC00000FD).
    """
    with (
        sf.SoundFile(str(source)) as reader,
        sf.SoundFile(
            str(dest),
            "w",
            samplerate=reader.samplerate,
            channels=min(reader.channels, 2),
            format="MP3",
            subtype="MPEG_LAYER_III",
            compression_level=0.2,
            bitrate_mode="VARIABLE",
        ) as writer,
    ):
        for block in reader.blocks(blocksize=4096, dtype="float32", always_2d=True):
            writer.write(block[:, :2])


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def subject_tags(subjects: Iterable[str]) -> set[str]:
    """Genre and voice tags from a source's own subject keywords."""
    tags: set[str] = set()
    for subject in subjects:
        text = subject.lower()
        for tag, phrases in GENRES.items():
            if any(
                re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", text) for p in phrases
            ):
                tags.add(tag)
    return tags


@dataclass(frozen=True)
class Scale:
    """Where a track sits in the whole library: energy rank, brightness terciles."""

    energy_rank: dict[str, float]
    bright_low: float
    bright_high: float


def scale(entries: Sequence[Entry]) -> Scale:
    """
    Library-relative positions for energy and brightness.

    Energy becomes a rank from 0 (calmest) to 1 (busiest); brightness is cut
    at its terciles. Both are relative because absolute values of either say
    little across genres — they only order tracks.
    """
    if not entries:
        return Scale({}, 0.0, 0.0)
    order = sorted(entries, key=lambda e: (e.energy_raw, e.id))
    span = max(1, len(order) - 1)
    ranks = {e.id: (i / span if len(order) > 1 else 0.5) for i, e in enumerate(order)}
    low, high = np.quantile([e.brightness for e in entries], [1 / 3, 2 / 3])
    return Scale(ranks, float(low), float(high))


def auto_tags(entry: Entry, where: Scale) -> set[str]:  # noqa: C901, PLR0912 - one rule per tag, in the contract's order
    """
    The analyser's tags for one track, per ``web/README.md``.

    Rhythm from beat strength; tempo bands from bpm; calm/lively and
    dark/bright from library terciles; then the mode-fit rules.
    """
    tags: set[str] = set()
    rank = where.energy_rank.get(entry.id, 0.5)
    strength = entry.beat_strength
    has_beat = strength >= BEAT
    beatless = strength < NO_BEAT
    if has_beat:
        tags.add("beat")
    if beatless:
        tags.add("beatless")
    if entry.bpm is not None:
        if entry.bpm < 75.0:  # noqa: PLR2004
            tags.add("slow")
        elif entry.bpm <= 105.0:  # noqa: PLR2004
            tags.add("mid")
        else:
            tags.add("upbeat")
    if rank < 1 / 3:
        tags.add("calm")
    elif rank >= 2 / 3:
        tags.add("lively")
    if entry.brightness < where.bright_low:
        tags.add("dark")
    elif entry.brightness >= where.bright_high:
        tags.add("bright")
    if (
        has_beat
        and entry.bpm is not None
        and FOCUS_BPM[0] <= entry.bpm <= FOCUS_BPM[1]
        and rank < 2 / 3
    ):
        tags.add("focus")
    if has_beat or (beatless and rank >= 1 / 3):
        tags.add("create")
    if beatless and rank < 1 / 3:
        tags.add("meditate")
    return tags


def classify(library: Library) -> Scale:
    """
    Recompute analyser tags for every track not yet in D1.

    Tracks already published keep their tags: from then on the app's people
    own them.
    """
    where = scale(list(library.entries.values()))
    for entry in library.entries.values():
        if entry.in_d1:
            continue
        kept = {
            t: s
            for t, s in entry.tags.items()
            if not (s == "auto" and t in ANALYSER_TAGS)
        }
        for tag in auto_tags(entry, where):
            kept.setdefault(tag, "auto")
        entry.tags = kept
    return where


def _not_a_tag(tag: str) -> str:
    return f"{tag!r} is not a tag: lower-case letters, digits and hyphens, at most 24"


def edit_tags(entry: Entry, add: Iterable[str], remove: Iterable[str]) -> None:
    """
    Add and remove tags by hand.

    On a track already in D1 the change is also queued for the next publish.
    """
    add, remove = list(add), list(remove)
    for tag in (*add, *remove):
        if not TAG.match(tag):
            raise ValueError(_not_a_tag(tag))
    for tag in remove:
        entry.tags.pop(tag, None)
        if entry.in_d1:
            entry.pending_add = [t for t in entry.pending_add if t != tag]
            entry.pending_remove = sorted({*entry.pending_remove, tag})
    for tag in add:
        entry.tags[tag] = "user"
        if entry.in_d1:
            entry.pending_remove = [t for t in entry.pending_remove if t != tag]
            entry.pending_add = sorted({*entry.pending_add, tag})


# ---------------------------------------------------------------------------
# adding a file you have
# ---------------------------------------------------------------------------


def _store(library: Library, staging: Path) -> tuple[str, Path]:
    """Move a staged MP3 into ``tracks/`` under its content id."""
    ident = file_id(staging)
    dest = library.root / "tracks" / f"{ident}{staging.suffix.lower()}"
    if dest.exists():
        staging.unlink()
    else:
        staging.replace(dest)
    return ident, dest


def _entry(  # noqa: PLR0913 - a manifest row is wide
    ident: str,
    stored: Path,
    library: Library,
    measured: Analysis,
    *,
    title: str,
    artist: str,
    album: str | None,
    license_name: str,
    license_url: str,
    source_url: str,
    tags: dict[str, str],
    provenance: dict[str, str],
) -> Entry:
    return Entry(
        id=ident,
        file=stored.relative_to(library.root).as_posix(),
        title=title,
        artist=artist,
        album=album,
        duration=measured.duration,
        bytes=stored.stat().st_size,
        mime=MIME[stored.suffix.lower()],
        loudness=measured.loudness,
        peak=measured.peak,
        gain_db=measured.gain_db,
        bpm=measured.bpm,
        beat_offset=measured.beat_offset,
        beat_strength=measured.beat_strength,
        energy_raw=measured.energy,
        brightness=measured.brightness,
        license=license_name,
        license_url=license_url,
        source_url=source_url,
        attribution=credit(title, artist, license_name, source_url),
        added_at=now(),
        tags=tags,
        provenance=provenance,
    )


def add_file(  # noqa: PLR0913 - every one is part of the credit
    library: Library,
    path: Path,
    *,
    title: str | None,
    artist: str,
    album: str | None,
    license_name: str,
    license_url: str,
    source_url: str,
    tags: Iterable[str] = (),
    measure: Callable[[Path], Analysis] = analyse,
) -> Entry:
    """
    Add a file you have to the library: MP3 as it is, lossless re-encoded.

    AAC (``.m4a``) is refused even though the app plays it: nothing here can
    decode it to measure it, and an unmeasured track has no loudness, tempo
    or beat. Convert it to MP3 or FLAC first.
    """
    suffix = path.suffix.lower()
    if suffix in {".m4a", ".aac", ".mp4"}:
        msg = (
            f"{path.name}: AAC cannot be decoded here to measure it;"
            " convert it to MP3 or FLAC first"
        )
        raise ValueError(msg)
    if suffix != ".mp3" and suffix not in TRANSCODABLE:
        msg = f"{path.name}: expected MP3 or one of {', '.join(TRANSCODABLE)}"
        raise ValueError(msg)
    tag_list = list(tags)
    for tag in tag_list:
        if not TAG.match(tag):
            raise ValueError(_not_a_tag(tag))
    staging = library.root / "tracks" / f".incoming{'.mp3'}"
    if suffix == ".mp3":
        shutil.copyfile(path, staging)
    else:
        transcode_to_mp3(path, staging)
    ident, stored = _store(library, staging)
    if ident in library.entries:
        return library.entries[ident]
    entry = _entry(
        ident,
        stored,
        library,
        measure(stored),
        title=title or path.stem,
        artist=artist,
        album=album,
        license_name=license_name,
        license_url=license_url,
        source_url=source_url,
        tags=dict.fromkeys(tag_list, "user"),
        provenance={"kind": "file", "original": path.name},
    )
    library.add(entry)
    return entry


def reanalyse(
    library: Library,
    measure: Callable[[Path], Analysis] = analyse,
    log: Callable[[str], None] = lambda _: None,
) -> int:
    """
    Measure every track again with the analyser as it is now.

    For when the analyser improves. Only the measurements change; the next
    publish carries them to D1, and the analyser's tags follow them on
    tracks D1 has not seen yet (on the rest, people own the tags).
    """
    for count, entry in enumerate(
        sorted(library.entries.values(), key=lambda e: e.id), 1
    ):
        measured = measure(library.path(entry))
        entry.duration = measured.duration
        entry.loudness = measured.loudness
        entry.peak = measured.peak
        entry.gain_db = measured.gain_db
        entry.bpm = measured.bpm
        entry.beat_offset = measured.beat_offset
        entry.beat_strength = measured.beat_strength
        entry.energy_raw = measured.energy
        entry.brightness = measured.brightness
        if count % 50 == 0:
            library.save()
            log(f"  measured {count} of {len(library.entries)}")
    library.save()
    return len(library.entries)


# ---------------------------------------------------------------------------
# fetching from the Internet Archive
# ---------------------------------------------------------------------------


@dataclass
class FetchState:
    """What the fetcher has already looked at and turned down, kept between runs."""

    rejected: set[str] = field(default_factory=set)
    done_items: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, root: Path) -> FetchState:
        """Read the state file, or start fresh."""
        path = root / FETCH_STATE
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(set(data.get("rejected", [])), set(data.get("done_items", [])))

    def save(self, root: Path) -> None:
        """Write the state file."""
        data = {
            "rejected": sorted(self.rejected),
            "done_items": sorted(self.done_items),
        }
        (root / FETCH_STATE).write_text(
            json.dumps(data, indent=1) + "\n", encoding="utf-8"
        )


@dataclass
class FetchReport:
    """What a fetch did, pool by pool."""

    added: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    rejected: int = 0
    items: int = 0


def _artist_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _download_and_measure(
    library: Library,
    http: Http,
    track: RemoteTrack,
    measure: Callable[[Path], Analysis],
) -> tuple[Path, Analysis] | None:
    """
    Fetch one remote track to a staging file (as MP3) and measure it.

    ``None`` means the file arrived but could not be decoded, which is a
    property of the file: it is turned down for good. A network failure
    raises :class:`OSError` instead, and is only skipped for this run.
    """
    staging_dir = library.root / "incoming"
    staging_dir.mkdir(exist_ok=True)
    stem = hashlib.sha256(f"{track.identifier}/{track.name}".encode()).hexdigest()[:16]
    raw = staging_dir / f"{stem}{Path(track.name).suffix.lower()}"
    http.download(track.url, raw)
    try:
        if track.needs_transcode:
            mp3 = staging_dir / f"{stem}.mp3"
            try:
                transcode_to_mp3(raw, mp3)
            finally:
                raw.unlink(missing_ok=True)
            raw = mp3
        return raw, measure(raw)
    except (RuntimeError, ValueError, sf.LibsndfileError):
        raw.unlink(missing_ok=True)
        return None


#: A candidate the network failed on: try it again next run.
_RETRY = "retry"


def _attempt(
    library: Library,
    http: Http,
    track: RemoteTrack,
    measure: Callable[[Path], Analysis],
    log: Callable[[str], None],
) -> tuple[Path, Analysis] | str | None:
    try:
        return _download_and_measure(library, http, track, measure)
    except OSError as error:
        log(f"  failed {track.identifier}/{track.name}: {error}")
        return _RETRY


def _fits(analysis: Analysis) -> bool:
    return LENGTHS[0] <= analysis.duration <= LENGTHS[1]


def _candidates(
    item: Item,
    tracks: list[RemoteTrack],
    library: Library,
    state: FetchState,
    budget: float,
) -> list[RemoteTrack]:
    """The tracks of an item worth downloading, up to ``budget`` seconds."""
    keep: list[RemoteTrack] = []
    planned = 0.0
    for track in tracks:
        key = f"{item.identifier}/{track.name}"
        if key in state.rejected or library.has_source(item.identifier, track.name):
            continue
        if UNSUITABLE.search(track.title) or (
            track.length is not None and not LENGTHS[0] <= track.length <= LENGTHS[1]
        ):
            state.rejected.add(key)
            continue
        planned += track.length or 240.0
        if planned > budget:
            break
        keep.append(track)
    return keep


@dataclass
class _PoolRun:
    """One pool being filled: its target, progress, and who is in it."""

    pool: Pool
    target: float
    have: float
    rule: Callable[[Analysis], bool]
    artists: dict[str, float]
    report: FetchReport

    def full(self) -> bool:
        """Whether the pool has its hours."""
        return self.have >= self.target


def _keep(
    run: _PoolRun,
    item: Item,
    track: RemoteTrack,
    library: Library,
    staged: Path,
    measured: Analysis,
) -> None:
    """File a downloaded track into the library and count it."""
    ident, stored = _store(library, staged)
    if ident in library.entries:
        return
    library.add(
        _entry(
            ident,
            stored,
            library,
            measured,
            title=track.title,
            artist=track.artist,
            album=track.album,
            license_name=item.license.name,
            license_url=item.license.url,
            source_url=item.url,
            tags=dict.fromkeys(sorted(subject_tags(item.subjects)), "auto"),
            provenance={
                "kind": "archive",
                "identifier": item.identifier,
                "file": track.name,
                "pool": run.pool.name,
            },
        )
    )
    artist = _artist_key(track.artist)
    run.artists[artist] = run.artists.get(artist, 0.0) + measured.duration
    run.have += measured.duration
    run.report.added[run.pool.name] += 1


def _take_item(
    run: _PoolRun,
    item: Item,
    library: Library,
    http: Http,
    state: FetchState,
    measure: Callable[[Path], Analysis],
    workers: int,
    log: Callable[[str], None],
) -> bool:
    """
    Download, measure and keep or turn down an item's tracks.

    Returns whether the item is finished with: every candidate kept or turned
    down for a reason that will not change. A track skipped because the pool
    filled up, or because the network failed, leaves it unfinished, so a
    later run with more hours, or a better connection, comes back to it.
    """
    tracks = item_tracks(http, item)
    budget = min(ITEM_CAP, 2.0 * (run.target - run.have) + 600.0)
    todo = [
        t
        for t in _candidates(item, tracks, library, state, budget)
        if run.artists.get(_artist_key(t.artist), 0.0) < ARTIST_CAP
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(lambda t: (t, _attempt(library, http, t, measure, log)), todo)
        )
    finished = True
    for track, result in results:
        key = f"{item.identifier}/{track.name}"
        if isinstance(result, str):
            finished = False
            continue
        if result is None:
            state.rejected.add(key)
            run.report.rejected += 1
            continue
        staged, measured = result
        if run.full():
            staged.unlink(missing_ok=True)
            finished = False
            continue
        artist = _artist_key(track.artist)
        over_cap = run.artists.get(artist, 0.0) + measured.duration > ARTIST_CAP
        if not _fits(measured) or not run.rule(measured) or over_cap:
            staged.unlink(missing_ok=True)
            state.rejected.add(key)
            run.report.rejected += 1
            continue
        _keep(run, item, track, library, staged, measured)
    return finished


def fetch(
    library: Library,
    http: Http,
    pools: Sequence[Pool] = DEFAULT_POOLS,
    *,
    scale_hours: float = 1.0,
    accept: dict[str, Callable[[Analysis], bool]] | None = None,
    measure: Callable[[Path], Analysis] = analyse,
    workers: int = 3,
    log: Callable[[str], None] = lambda _: None,
) -> FetchReport:
    """
    Fill each pool to its hours (times ``scale_hours``) from the Archive.

    Items are taken most-downloaded first, interleaved across the pool's
    queries. Each candidate track is downloaded, measured, and kept only if
    its length is sensible and the pool's ``accept`` rule passes; everything
    turned down is remembered in ``fetch-state.json`` so a second run does
    not fetch it again. No item contributes more than 1.5 hours and no artist
    more than 2. Safe to interrupt: the manifest is saved after every item.
    """
    rules = ACCEPT if accept is None else accept
    state = FetchState.load(library.root)
    report = FetchReport()
    artists: dict[str, float] = {}
    for entry in library.entries.values():
        key = _artist_key(entry.artist)
        artists[key] = artists.get(key, 0.0) + entry.duration
    for pool in pools:
        run = _PoolRun(
            pool,
            pool.hours * scale_hours * 3600.0,
            library.seconds(pool=pool.name),
            rules.get(pool.name, _accept_any),
            artists,
            report,
        )
        report.added.setdefault(pool.name, 0)
        report.seconds[pool.name] = run.have
        if run.full():
            continue
        hits = [list(search_all(http, q)) for q in pool.queries]
        for item in interleave(hits):
            if run.full():
                break
            if item.identifier in state.done_items or item.unsuitable():
                continue
            report.items += 1
            try:
                finished = _take_item(
                    run, item, library, http, state, measure, workers, log
                )
            except OSError as error:
                log(f"  skip {item.identifier}: {error}")
                continue
            if finished:
                state.done_items.add(item.identifier)
            library.save()
            state.save(library.root)
            report.seconds[pool.name] = run.have
            log(
                f"  {pool.name}: {run.have / 3600:.2f} of {run.target / 3600:.1f} h"
                f" after {item.identifier}"
            )
    return report


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------

WRANGLER = "wrangler@4.151.0"
BUCKET = "violet-music"
DATABASE = "violet"

#: Columns of ``tracks``, in the migration's order.
TRACK_COLUMNS = (
    "id",
    "r2_key",
    "title",
    "artist",
    "album",
    "duration",
    "bytes",
    "mime",
    "loudness",
    "peak",
    "gain_db",
    "bpm",
    "beat_offset",
    "beat_strength",
    "energy",
    "brightness",
    "license",
    "license_url",
    "source_url",
    "attribution",
    "added_at",
)

#: Rewritten on every publish. Not the key, the R2 key, or when it was added.
UPDATED_COLUMNS = tuple(
    c for c in TRACK_COLUMNS if c not in {"id", "r2_key", "added_at"}
)


def sql_literal(value: object) -> str:
    """One SQLite literal. Strings are quoted with doubled quotes; NUL is dropped."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else "NULL"
    text = str(value).replace("\x00", "")
    return "'" + text.replace("'", "''") + "'"


def track_row(entry: Entry, energy: float) -> dict[str, object]:
    """The ``tracks`` row for an entry."""
    row = {
        name: getattr(entry, name)
        for name in TRACK_COLUMNS
        if name not in {"r2_key", "energy"}
    }
    row["r2_key"] = entry.r2_key
    row["energy"] = round(energy, 4)
    return {name: row[name] for name in TRACK_COLUMNS}


def publish_sql(entries: Sequence[Entry], where: Scale) -> str:
    """
    The SQL that brings D1 up to date with ``entries``.

    Every track is upserted. Tracks D1 has not seen get all their tags;
    tracks it has get only the tag edits queued since the last publish.
    """
    lines: list[str] = []
    columns = ", ".join(TRACK_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in UPDATED_COLUMNS)
    for entry in entries:
        row = track_row(entry, where.energy_rank.get(entry.id, 0.5))
        values = ", ".join(sql_literal(v) for v in row.values())
        lines.append(
            f"INSERT INTO tracks ({columns}) VALUES ({values})"  # noqa: S608 - every value is sql_literal
            f" ON CONFLICT(id) DO UPDATE SET {updates};"
        )
        ident = sql_literal(entry.id)
        if not entry.in_d1:
            lines.extend(
                "INSERT OR IGNORE INTO track_tags (track_id, tag, source) "  # noqa: S608
                f"VALUES ({ident}, {sql_literal(tag)}, {sql_literal(source)});"
                for tag, source in sorted(entry.tags.items())
            )
            continue
        lines.extend(
            "INSERT INTO track_tags (track_id, tag, source) "  # noqa: S608
            f"VALUES ({ident}, {sql_literal(tag)}, 'user')"
            " ON CONFLICT(track_id, tag) DO UPDATE SET source = 'user';"
            for tag in entry.pending_add
        )
        lines.extend(
            f"DELETE FROM track_tags WHERE track_id = {ident}"  # noqa: S608
            f" AND tag = {sql_literal(tag)};"
            for tag in entry.pending_remove
        )
    return "\n".join(lines) + "\n"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed program and arguments built here
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@dataclass
class Wrangler:
    """The two Cloudflare operations publishing needs, through ``npx wrangler``."""

    web: Path
    run: Callable[[list[str], Path], subprocess.CompletedProcess[str]] = _run
    #: Each call is tried this many times. Cloudflare's API now and then
    #: refuses one request out of hundreds (the first starter publish died on
    #: a 7403 that did not recur), and every call here is safe to repeat: an
    #: upload overwrites, the SQL upserts.
    attempts: int = 3

    def _command(self, *args: str) -> list[str]:
        npx = shutil.which("npx")
        if npx is None:
            msg = "npx was not found; install Node.js to publish"
            raise RuntimeError(msg)
        return [npx, "--yes", WRANGLER, *args]

    def _call(self, command: list[str], what: str) -> str:
        for _ in range(self.attempts - 1):
            result = self.run(command, self.web)
            if result.returncode == 0:
                return result.stdout
        result = self.run(command, self.web)
        if result.returncode == 0:
            return result.stdout
        detail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        msg = f"{what} failed:\n" + "\n".join(detail)
        raise RuntimeError(msg)

    def put(self, key: str, path: Path, mime: str) -> None:
        """Upload one file to the bucket."""
        command = self._command(
            "r2",
            "object",
            "put",
            f"{BUCKET}/{key}",
            "--file",
            str(path.resolve()),
            "--content-type",
            mime,
            "--remote",
        )
        self._call(command, f"upload of {key}")

    def track_ids(self) -> set[str]:
        """The ids D1 already has."""
        command = self._command(
            "d1",
            "execute",
            DATABASE,
            "--remote",
            "--command",
            "SELECT id FROM tracks",
            "--json",
        )
        out = self._call(command, "reading track ids from D1")
        start = out.find("[")
        if start < 0:
            msg = f"unexpected output from wrangler: {out[:200]!r}"
            raise RuntimeError(msg)
        data = json.loads(out[start:])
        return {str(row["id"]) for block in data for row in block.get("results", [])}

    def execute(self, sql: Path) -> None:
        """Run a SQL file against D1."""
        command = self._command(
            "d1", "execute", DATABASE, "--remote", "--file", str(sql.resolve())
        )
        self._call(command, f"running {sql.name} on D1")


@dataclass
class PublishReport:
    """What a publish did."""

    uploaded: int = 0
    rows: int = 0
    new_rows: int = 0
    tag_edits: int = 0


def publish(
    library: Library,
    cloud: Wrangler,
    *,
    workers: int = 6,
    batch: int = 100,
    log: Callable[[str], None] = lambda _: None,
) -> PublishReport:
    """
    Upload what R2 lacks and bring D1 up to date. Safe to run again.

    D1 is asked which ids it has, so a manifest copied from elsewhere does
    not resend tags. Files are uploaded only once (``uploaded_at`` records
    it). The manifest is saved after every upload, so an interrupted publish
    resumes where it stopped.
    """
    report = PublishReport()
    known = cloud.track_ids()
    for entry in library.entries.values():
        entry.in_d1 = entry.id in known
    where = classify(library)

    def upload(entry: Entry) -> None:
        cloud.put(entry.r2_key, library.path(entry), entry.mime)
        entry.uploaded_at = now()
        library.save()
        log(f"  uploaded {entry.r2_key}  {entry.title}")

    pending = [e for e in library.entries.values() if e.uploaded_at is None]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(upload, pending))
    report.uploaded = len(pending)

    ready = sorted(
        (e for e in library.entries.values() if e.uploaded_at is not None),
        key=lambda e: e.id,
    )
    for start in range(0, len(ready), batch):
        chunk = ready[start : start + batch]
        sql = library.root / f"publish-{start // batch:03d}.sql"
        sql.write_text(publish_sql(chunk, where), encoding="utf-8")
        cloud.execute(sql)
        sql.unlink()
        for entry in chunk:
            report.rows += 1
            report.new_rows += 0 if entry.in_d1 else 1
            report.tag_edits += len(entry.pending_add) + len(entry.pending_remove)
            entry.in_d1 = True
            entry.pending_add, entry.pending_remove = [], []
        library.save()
        log(f"  D1: {min(start + batch, len(ready))} of {len(ready)} rows")
    return report
