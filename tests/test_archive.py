"""The Internet Archive source: licences, search, and picking files, offline."""

from __future__ import annotations

import io
import json
import urllib.parse
from typing import TYPE_CHECKING, Any

import pytest

from violet import archive
from violet.archive import (
    UNSUITABLE,
    Item,
    License,
    UrllibHttp,
    interleave,
    item_tracks,
    parse_length,
    parse_license,
    search,
    search_all,
)

if TYPE_CHECKING:
    from pathlib import Path

CC = "http://creativecommons.org"


class FakeHttp:
    """Answers JSON by URL prefix, and writes the URL itself as a 'download'."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = pages
        self.asked: list[str] = []

    def get_json(self, url: str) -> Any:  # noqa: ANN401 - whatever the page holds
        """Answer from the first page whose prefix matches."""
        self.asked.append(url)
        for prefix, page in self.pages.items():
            if url.startswith(prefix):
                return page(url) if callable(page) else page
        raise OSError(url)

    def download(self, url: str, dest: Path) -> None:
        """Write the URL as the file's content: unique per file, so unique ids."""
        dest.write_bytes(url.encode())


# ---------------------------------------------------------------------------
# licences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "name", "canonical"),
    [
        (
            f"{CC}/publicdomain/zero/1.0/",
            "CC0 1.0",
            "https://creativecommons.org/publicdomain/zero/1.0/",
        ),
        (
            "https://creativecommons.org/publicdomain/mark/1.0/",
            "Public Domain Mark 1.0",
            "https://creativecommons.org/publicdomain/mark/1.0/",
        ),
        (
            f"{CC}/licenses/publicdomain/",
            "Public Domain",
            "https://creativecommons.org/licenses/publicdomain/",
        ),
        (
            f"{CC}/licenses/by/4.0/",
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        ),
        (
            f"{CC}/licenses/by-sa/3.0/deed.en",
            "CC BY-SA 3.0",
            "https://creativecommons.org/licenses/by-sa/3.0/",
        ),
        (
            f"{CC}/licenses/by-nc/2.5/",
            "CC BY-NC 2.5",
            "https://creativecommons.org/licenses/by-nc/2.5/",
        ),
        (
            f"{CC}/licenses/by-nc-sa/2.0/uk/",
            "CC BY-NC-SA 2.0 UK",
            "https://creativecommons.org/licenses/by-nc-sa/2.0/uk/",
        ),
        (
            f"{CC}/licenses/by-sa-nc/1.0/",
            "CC BY-NC-SA 1.0",
            "https://creativecommons.org/licenses/by-nc-sa/1.0/",
        ),
        (
            " HTTP://CreativeCommons.org/Licenses/BY/3.0 ",
            "CC BY 3.0",
            "https://creativecommons.org/licenses/by/3.0/",
        ),
    ],
)
def test_accepted_licences(url: str, name: str, canonical: str) -> None:
    assert parse_license(url) == License(name, canonical)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        f"{CC}/licenses/by-nd/4.0/",
        f"{CC}/licenses/by-nc-nd/3.0/",
        f"{CC}/licenses/nd/1.0/",
        f"{CC}/licenses/sampling+/1.0/",
        f"{CC}/licenses/",
        "https://example.com/licenses/by/4.0/",
        "All rights reserved",
    ],
)
def test_rejected_licences(url: str | None) -> None:
    """No derivatives, nothing, or anything unrecognised: all a no."""
    assert parse_license(url) is None


# ---------------------------------------------------------------------------
# suitability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Vocals",
        "with vocal",
        "Rap",
        "feat. Somebody",
        "Spoken Word",
        "Binaural beats",
        "Harsh Noise",
        "drum & bass",
        "Drum and Bass",
        "chiptune",
        "8-bit",
        "LibriVox",
        "Humour",
        "Positive Affirmations",
        "Guided meditation",
        "Sacred Songs and Chants",
        "Underground Boom Bap Mixtape",
        "remix; bootleg; mash up",
        "radio sampling",
        "[No Copyright] 10 Minutes LOFI",
        "yt-5s.com",
        "(Cassette Rip)",
    ],
)
def test_unsuitable(text: str) -> None:
    assert UNSUITABLE.search(text)


@pytest.mark.parametrize(
    "text",
    ["instrumental hip-hop", "lofi", "ambient", "trip-hop", "piano", "Rapture"],
)
def test_suitable(text: str) -> None:
    assert not UNSUITABLE.search(text)


def make_item(**changes: object) -> Item:
    base: dict[str, Any] = {
        "identifier": "x1",
        "title": "Calm Tape",
        "creator": "Someone",
        "license": License("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
        "subjects": ("lofi",),
        "downloads": 5,
    }
    base.update(changes)
    return Item(**base)


def test_item_suitability_reads_title_and_subjects() -> None:
    assert not make_item().unsuitable()
    assert make_item(subjects=("lofi", "rap")).unsuitable()
    assert make_item(title="Beats feat. MC Somebody").unsuitable()


@pytest.mark.parametrize(
    ("subjects", "unsuitable"),
    [
        (("hip-hop", "boom bap"), True),
        (("Hip Hop",), True),
        (("hip-hop", "instrumental"), False),
        (("instrumental hip-hop",), False),
        (("hip-hop", "beats"), False),
        (("boom bap", "beat tape"), False),
        (("lofi",), False),
    ],
)
def test_hip_hop_has_to_say_it_is_instrumental(
    subjects: tuple[str, ...],
    unsuitable: bool,  # noqa: FBT001
) -> None:
    assert make_item(title="Tape", subjects=subjects).unsuitable() is unsuitable


def test_item_url() -> None:
    assert make_item().url == "https://archive.org/details/x1"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_page(*docs: dict[str, Any]) -> dict[str, Any]:
    return {"response": {"numFound": len(docs), "docs": list(docs)}}


def test_search_keeps_only_accepted_licences() -> None:
    http = FakeHttp(
        {
            archive.SEARCH_URL: search_page(
                {
                    "identifier": "ok",
                    "title": "Good",
                    "creator": ["Ann", "Bo"],
                    "licenseurl": f"{CC}/licenses/by/3.0/",
                    "subject": "lofi",
                    "downloads": 12,
                },
                {"identifier": "nd", "licenseurl": f"{CC}/licenses/by-nd/3.0/"},
                {"identifier": "none"},
                {"licenseurl": f"{CC}/licenses/by/3.0/"},
                {
                    "identifier": "bare",
                    "licenseurl": [f"{CC}/publicdomain/zero/1.0/"],
                },
            ),
        },
    )
    items = search(http, "q")
    assert [i.identifier for i in items] == ["ok", "bare"]
    assert items[0].creator == "Ann"
    assert items[0].subjects == ("lofi",)
    assert items[0].downloads == 12
    assert items[1].title == "bare"
    assert items[1].creator is None
    assert items[1].downloads == 0
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(http.asked[0]).query)
    assert query["q"] == ["q"]
    assert query["sort[]"] == ["downloads desc"]


def test_search_all_pages_until_empty() -> None:
    def page(url: str) -> dict[str, Any]:
        number = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page"][0])
        if number > 2:
            return search_page()
        return search_page(
            {"identifier": f"p{number}", "licenseurl": f"{CC}/licenses/by/4.0/"}
        )

    http = FakeHttp({archive.SEARCH_URL: page})
    assert [i.identifier for i in search_all(http, "q", limit=10_000)] == ["p1", "p2"]
    assert len(http.asked) == 3


def test_search_all_stops_at_the_limit() -> None:
    http = FakeHttp(
        {
            archive.SEARCH_URL: search_page(
                {"identifier": "a", "licenseurl": f"{CC}/licenses/by/4.0/"}
            ),
        },
    )
    assert len(list(search_all(http, "q", limit=400, rows=200))) == 2
    assert len(http.asked) == 2


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("254.3", 254.3),
        ("4:14", 254.0),
        ("1:02:03", 3723.0),
        (300, 300.0),
        (None, None),
        ("soon", None),
    ],
)
def test_parse_length(value: object, seconds: float | None) -> None:
    assert parse_length(value) == seconds


def metadata(*files: dict[str, Any]) -> dict[str, Any]:
    return {"metadata": {}, "files": list(files)}


def test_item_tracks_picks_one_file_per_track() -> None:
    http = FakeHttp(
        {
            "https://archive.org/metadata/x1": metadata(
                {
                    "name": "02 b.flac",
                    "format": "Flac",
                    "source": "original",
                    "track": "2",
                },
                {
                    "name": "02 b.mp3",
                    "format": "VBR MP3",
                    "original": "02 b.flac",
                    "title": "Bee",
                    "length": "3:00",
                    "size": "123",
                },
                {
                    "name": "02 b_64kb.mp3",
                    "format": "64Kbps MP3",
                    "original": "02 b.flac",
                },
                {
                    "name": "01 a.flac",
                    "format": "Flac",
                    "track": "1/9",
                    "artist": "Ay",
                    "album": "Disc",
                },
                {"name": "cover.jpg", "format": "JPEG"},
                {"name": "x1_meta.xml", "format": "Metadata"},
            ),
        },
    )
    tracks = item_tracks(http, make_item())
    assert [t.name for t in tracks] == ["01 a.flac", "02 b.mp3"]
    first, second = tracks
    assert first.needs_transcode
    assert (first.title, first.artist, first.album) == ("01 a", "Ay", "Disc")
    assert first.length is None
    assert first.size is None
    assert not second.needs_transcode
    assert (second.title, second.artist, second.album) == (
        "Bee",
        "Someone",
        "Calm Tape",
    )
    assert second.length == 180.0
    assert second.size == 123
    assert second.url == "https://archive.org/download/x1/02%20b.mp3"


def test_item_tracks_without_a_creator() -> None:
    http = FakeHttp(
        {
            "https://archive.org/metadata/x1": metadata(
                {"name": "a.mp3", "format": "MP3"}
            )
        }
    )
    [track] = item_tracks(http, make_item(creator=None))
    assert track.artist == "Unknown artist"


def test_item_tracks_without_numbers_sort_by_name() -> None:
    http = FakeHttp(
        {
            "https://archive.org/metadata/x1": metadata(
                {"name": "b.mp3", "format": "MP3"},
                {"name": "a.mp3", "format": "MP3"},
                {"name": "c.mp3", "format": "MP3", "track": "1"},
            )
        }
    )
    assert [t.name for t in item_tracks(http, make_item())] == [
        "c.mp3",
        "a.mp3",
        "b.mp3",
    ]


def test_interleave_takes_turns_and_skips_repeats() -> None:
    a, b, c, d = (make_item(identifier=x) for x in "abcd")
    order = [i.identifier for i in interleave([[a, b, c], [b, d]])]
    assert order == ["a", "b", "d", "c"]


# ---------------------------------------------------------------------------
# the real HTTP client, against a fake urlopen
# ---------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    """Bytes that work as a context manager, like an HTTPResponse."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_urllib_http_sends_the_agent_and_reads_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def urlopen(request: Any, timeout: float) -> FakeResponse:  # noqa: ANN401
        seen.append((request, timeout))
        return FakeResponse(json.dumps({"ok": True}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert UrllibHttp(timeout=5).get_json("https://archive.org/x") == {"ok": True}
    request, timeout = seen[0]
    assert request.get_header("User-agent") == archive.USER_AGENT
    assert "@" not in archive.USER_AGENT
    assert timeout == 5


def test_urllib_http_downloads_via_a_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _r, timeout: FakeResponse(b"x" * 200_000),  # noqa: ARG005
    )
    dest = tmp_path / "t.mp3"
    UrllibHttp().download("https://archive.org/t.mp3", dest)
    assert dest.read_bytes() == b"x" * 200_000
    assert not (tmp_path / "t.mp3.part").exists()


def test_urllib_http_retries_then_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def urlopen(_request: Any, timeout: float) -> FakeResponse:  # noqa: ANN401, ARG001
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError
        return FakeResponse(b"[]")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    assert UrllibHttp(retries=3).get_json("https://archive.org/x") == []
    calls.clear()
    with pytest.raises(TimeoutError):
        UrllibHttp(retries=2).get_json("https://archive.org/x")


def test_item_tracks_prefer_a_phone_sized_mp3() -> None:
    http = FakeHttp(
        {
            "https://archive.org/metadata/x1": metadata(
                {"name": "a.flac", "format": "Flac"},
                {"name": "a_320.mp3", "format": "320Kbps MP3", "original": "a.flac"},
                {"name": "a_vbr.mp3", "format": "VBR MP3", "original": "a.flac"},
                {"name": "a_64.mp3", "format": "64Kbps MP3", "original": "a.flac"},
            )
        }
    )
    assert [t.name for t in item_tracks(http, make_item())] == ["a_vbr.mp3"]
