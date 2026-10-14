"""
The Internet Archive as a source of music you are allowed to change.

The Archive's netlabel collections hold tens of thousands of releases under
Creative Commons licences, searchable through a documented API
(``advancedsearch.php``) that reports each item's licence, and downloadable
file by file. That makes it the one large source of modifiable music that can
be gathered by program without breaking anyone's terms.

What is accepted is decided by the licence alone, item by item:

- **Accepted:** CC0, the Public Domain Mark (and the old CC public-domain
  dedication), CC BY, BY-SA, BY-NC and BY-NC-SA, any version. The app is
  personal and non-commercial, shared among friends, so NC is fine; every
  one of these requires a credit, which the library keeps.
- **Rejected:** anything with ND ("no derivatives" — the pulse *is* a
  derivative), anything without a licence URL, and anything this module
  cannot parse. An unknown licence is a no, not a maybe.

Network access goes through the :class:`Http` protocol so that every rule here
is tested without touching the network.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Who is asking. Descriptive, and deliberately without any personal detail.
USER_AGENT = "violet-library/0.1 (personal, non-commercial)"

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{name}"
DETAILS_URL = "https://archive.org/details/{identifier}"

#: MP3 derivatives, most wanted first. iOS plays all of them natively.
#: Around 160-190 kbps first: the Archive's VBR derivative is that, and it is
#: transparent on earbuds at about half the size of 320 kbps — which matters
#: twice, once in the bucket and again on a phone keeping playlists offline.
#: The first starter library preferred 320 and came to 4 GB for 40 hours.
MP3_FORMATS = (
    "VBR MP3",
    "192Kbps MP3",
    "160Kbps MP3",
    "224Kbps MP3",
    "256Kbps MP3",
    "320Kbps MP3",
    "MP3",
    "128Kbps MP3",
    "96Kbps MP3",
    "64Kbps MP3",
)

#: Formats that can be decoded here and re-encoded to MP3 when an item has no
#: MP3 of a track at all. Best first.
LOSSLESS_FORMATS = ("Flac", "24bit Flac", "WAVE", "Ogg Vorbis")

#: Words in an item's subjects or title, or a track's title, that mark it as
#: wrong for this library: voices and spoken word; DJ mixes, mash-ups and
#: rips, whose Creative Commons claim cannot be the uploader's to make; harsh
#: genres; and other people's binaural beats (the app adds its own). Each of
#: the less obvious ones is here because the first starter fetch let one in.
UNSUITABLE = re.compile(
    r"\b("
    r"vocals?|vox|sung|singer|songwriter|rap|rapper|mc|acapella|a cappella|"
    r"feat\.?|ft\.|"
    r"spoken|poetry|podcast|interview|audiobook|librivox|lecture|sermon|radio|"
    r"affirmations?|guided|hypnosis|chants?|chanting|mantras?|prayers?|"
    r"comedy|humou?r|karaoke|"
    r"mixtape|dj mix|bootleg|mash ?-?ups?|"
    r"no copyright|copyright free|youtube|yt-?\w*|(cassette|vinyl|tape) rip|"
    r"binaural|isochronic|solfeggio|"
    r"noise|harsh|metal|punk|grindcore|hardcore|industrial|"
    r"dubstep|trap|drum ?(and|&|n) ?bass|dnb|jungle|gabber|"
    r"chiptune|8-?bit|1-?bit"
    r")\b",
    re.IGNORECASE,
)

#: Hip-hop on the Archive is as often rapped as not. An item that calls itself
#: hip-hop has to say it is instrumental, or beats, as well.
HIP_HOP = re.compile(r"\b(hip[ -]?hop|boom ?bap)\b", re.IGNORECASE)
INSTRUMENTAL = re.compile(r"\b(instrumentals?|beats|beat ?tapes?)\b", re.IGNORECASE)


class Http(Protocol):
    """The two things this module asks of the network."""

    def get_json(self, url: str) -> Any:  # noqa: ANN401 - it is whatever the API sent
        """GET ``url`` and decode it as JSON."""
        ...

    def download(self, url: str, dest: Path) -> None:
        """GET ``url`` into the file ``dest``."""
        ...


@dataclass
class UrllibHttp:
    """:class:`Http` over the standard library, polite and patient."""

    timeout: float = 60.0
    retries: int = 3
    backoff: float = 2.0

    def _open(self, url: str) -> Any:  # noqa: ANN401 - an HTTPResponse
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - https URLs built here
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310
            except OSError as error:  # URLError and timeouts are OSErrors
                last = error
                time.sleep(self.backoff * (attempt + 1))
        assert last is not None  # noqa: S101 - the loop ran at least once
        raise last

    def get_json(self, url: str) -> Any:  # noqa: ANN401
        """GET ``url`` and decode it as JSON."""
        with self._open(url) as response:
            return json.load(response)

    def download(self, url: str, dest: Path) -> None:
        """GET ``url`` into ``dest``, via a temporary name so a cut is never kept."""
        partial = dest.with_name(dest.name + ".part")
        with self._open(url) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 16):
                out.write(chunk)
        partial.replace(dest)


# ---------------------------------------------------------------------------
# licences
# ---------------------------------------------------------------------------

_CC = re.compile(
    r"creativecommons\.org/licenses/(?P<kind>[a-z-]+)/(?P<version>\d(?:\.\d)?)(?:/(?P<port>[a-z]{2})(?=/|$))?",
)
_ACCEPTED_KINDS = {
    "by": "BY",
    "by-sa": "BY-SA",
    "by-nc": "BY-NC",
    "by-nc-sa": "BY-NC-SA",
    "by-sa-nc": "BY-NC-SA",
}


@dataclass(frozen=True)
class License:
    """A licence this library accepts: its short name and canonical URL."""

    name: str
    url: str


#: Public-domain tools, by a fragment of their URL.
_PUBLIC_DOMAIN = (
    ("publicdomain/zero/1.0", "CC0 1.0", "publicdomain/zero/1.0/"),
    ("publicdomain/mark/1.0", "Public Domain Mark 1.0", "publicdomain/mark/1.0/"),
    ("licenses/publicdomain", "Public Domain", "licenses/publicdomain/"),
)


def parse_license(url: str | None) -> License | None:
    """
    The licence behind a ``licenseurl``, or ``None`` if it is not acceptable.

    ``None`` covers both "not allowed" (any ND) and "not understood". The
    returned URL is the canonical https form, without any ``deed`` suffix.
    """
    text = (url or "").strip().lower()
    if "creativecommons.org/" not in text:
        return None
    for fragment, name, path in _PUBLIC_DOMAIN:
        if fragment in text:
            return License(name, f"https://creativecommons.org/{path}")
    match = _CC.search(text)
    kind = _ACCEPTED_KINDS.get(match["kind"]) if match else None
    if match is None or kind is None:
        return None
    version, port = match["version"], match["port"]
    name = f"CC {kind} {version}" + (f" {port.upper()}" if port else "")
    path = f"{kind.lower()}/{version}/" + (f"{port}/" if port else "")
    return License(name, f"https://creativecommons.org/licenses/{path}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _strings(value: object) -> list[str]:
    """IA fields are a string, a list of strings, or absent."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


@dataclass(frozen=True)
class Item:
    """One search hit: an Archive item with an accepted licence."""

    identifier: str
    title: str
    creator: str | None
    license: License
    subjects: tuple[str, ...]
    downloads: int

    @property
    def url(self) -> str:
        """The item's page, which is what a credit should point to."""
        return DETAILS_URL.format(identifier=self.identifier)

    def unsuitable(self) -> bool:
        """Whether the title or any subject says this is not for the library."""
        text = " | ".join((self.title, *self.subjects))
        if UNSUITABLE.search(text):
            return True
        return bool(HIP_HOP.search(text)) and not INSTRUMENTAL.search(text)


def search(http: Http, query: str, *, rows: int = 200, page: int = 1) -> list[Item]:
    """
    One page of an advanced search, most downloaded first, licence-filtered.

    Items without an acceptable licence are dropped here, before anything
    else looks at them.
    """
    fields = ("identifier", "title", "creator", "licenseurl", "subject", "downloads")
    params: list[tuple[str, str]] = [("q", query)]
    params += [("fl[]", f) for f in fields]
    params += [
        ("sort[]", "downloads desc"),
        ("rows", str(rows)),
        ("page", str(page)),
        ("output", "json"),
    ]
    data = http.get_json(SEARCH_URL + "?" + urllib.parse.urlencode(params))
    items: list[Item] = []
    for doc in data.get("response", {}).get("docs", []):
        licence = parse_license(next(iter(_strings(doc.get("licenseurl"))), None))
        if licence is None or "identifier" not in doc:
            continue
        creators = _strings(doc.get("creator"))
        items.append(
            Item(
                identifier=str(doc["identifier"]),
                title=next(iter(_strings(doc.get("title"))), str(doc["identifier"])),
                creator=creators[0] if creators else None,
                license=licence,
                subjects=tuple(_strings(doc.get("subject"))),
                downloads=int(doc.get("downloads") or 0),
            ),
        )
    return items


def search_all(
    http: Http, query: str, *, limit: int = 600, rows: int = 200
) -> Iterator[Item]:
    """Every accepted hit for ``query``, page by page, up to ``limit`` hits seen."""
    seen = 0
    page = 1
    while seen < limit:
        batch = search(http, query, rows=rows, page=page)
        yield from batch
        # A short raw page means the end; the licence filter makes `batch`
        # shorter than `rows` on most pages, so count pages, not hits.
        seen += rows
        page += 1
        if not batch:
            return


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def parse_length(value: object) -> float | None:
    """IA's ``length``: seconds as ``"254.3"``, or ``"4:14"`` / ``"1:02:03"``."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            seconds = 0.0
            for part in text.split(":"):
                seconds = seconds * 60.0 + float(part)
            return seconds
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class RemoteTrack:
    """One track in an item, and the best file to fetch it from."""

    identifier: str
    name: str
    format: str
    title: str
    artist: str
    album: str
    length: float | None
    size: int | None

    @property
    def url(self) -> str:
        """Where to download it."""
        return DOWNLOAD_URL.format(
            identifier=self.identifier, name=urllib.parse.quote(self.name)
        )

    @property
    def needs_transcode(self) -> bool:
        """True when the only copy is in a format iOS will not play."""
        return self.format not in MP3_FORMATS


def _rank(fmt: str) -> int:
    order = MP3_FORMATS + LOSSLESS_FORMATS
    return order.index(fmt) if fmt in order else len(order)


def item_tracks(http: Http, item: Item) -> list[RemoteTrack]:
    """
    The tracks in ``item``, one file each, in the item's own order.

    An item lists every file it has: the uploaded original and the MP3s and
    Oggs derived from it. Files are grouped by the original they came from,
    and each group contributes its best MP3, or failing that its best
    lossless or Ogg file, to be transcoded.
    """
    data = http.get_json(
        METADATA_URL.format(identifier=urllib.parse.quote(item.identifier))
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in data.get("files", []):
        fmt = str(entry.get("format", ""))
        if _rank(fmt) == len(MP3_FORMATS) + len(LOSSLESS_FORMATS):
            continue
        key = str(entry.get("original") or entry.get("name"))
        groups.setdefault(key, []).append(entry)
    tracks: list[RemoteTrack] = []
    for key in sorted(groups, key=_track_order(groups)):
        best = min(groups[key], key=lambda e: _rank(str(e.get("format", ""))))
        name = str(best["name"])
        stem = Path(name).stem
        title = next((str(e["title"]) for e in groups[key] if e.get("title")), stem)
        artist = next(
            (str(e["artist"]) for e in groups[key] if e.get("artist")),
            item.creator or "Unknown artist",
        )
        album = next(
            (str(e["album"]) for e in groups[key] if e.get("album")), item.title
        )
        length = next(
            (parse_length(e["length"]) for e in groups[key] if e.get("length")), None
        )
        size = int(best["size"]) if str(best.get("size", "")).isdigit() else None
        tracks.append(
            RemoteTrack(
                item.identifier,
                name,
                str(best["format"]),
                title,
                artist,
                album,
                length,
                size,
            )
        )
    return tracks


def _track_order(groups: dict[str, list[dict[str, Any]]]) -> Any:  # noqa: ANN401 - a sort key
    """Sort by the item's track numbers where it gives them, then by name."""

    def key(name: str) -> tuple[int, str]:
        numbers = [str(e.get("track", "")) for e in groups[name]]
        digits = next(
            (re.match(r"\d+", n) for n in numbers if re.match(r"\d+", n)), None
        )
        return (int(digits.group()) if digits else 10**6, name)

    return key


@dataclass
class Pool:
    """
    A share of the library: some hours of one kind of music, and where to look.

    ``queries`` are tried in turn, their hits interleaved so no one query's
    most popular label takes the whole pool.
    """

    name: str
    hours: float
    queries: Sequence[str]
    note: str = ""


def interleave(lists: Sequence[Sequence[Item]]) -> Iterator[Item]:
    """Round-robin through several hit lists, skipping repeats."""
    seen: set[str] = set()
    for row in range(max((len(x) for x in lists), default=0)):
        for hits in lists:
            if row < len(hits) and hits[row].identifier not in seen:
                seen.add(hits[row].identifier)
                yield hits[row]
