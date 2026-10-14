"""
The music library: manifest, tags, adding, fetching and publishing.

Publishing is tested end to end against the app's real schema: the fake
``wrangler`` below runs the SQL it is handed on an in-memory SQLite database
built from ``web/migrations``, which is what D1 is.
"""

from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

import music
from violet import library as lib
from violet import library_cli
from violet.analysis import Analysis, analyse
from violet.archive import SEARCH_URL, Pool
from violet.cli import app
from violet.library import (
    Entry,
    Library,
    Scale,
    Wrangler,
    add_file,
    auto_tags,
    classify,
    credit,
    edit_tags,
    fetch,
    file_id,
    publish,
    publish_sql,
    reanalyse,
    scale,
    sql_literal,
    subject_tags,
    transcode_to_mp3,
)

if TYPE_CHECKING:
    from collections.abc import Callable

MIGRATIONS = Path(__file__).resolve().parents[1] / "web" / "migrations"
BY = "http://creativecommons.org/licenses/by/4.0/"


def measured(**changes: Any) -> Analysis:  # noqa: ANN401 - field values
    base: dict[str, Any] = {
        "duration": 240.0,
        "sample_rate": 44100,
        "channels": 2,
        "loudness": -12.0,
        "peak": 0.9,
        "gain_db": -4.0,
        "bpm": 85.0,
        "beat_offset": 0.1,
        "beat_strength": 0.7,
        "energy": 1.0,
        "brightness": 1500.0,
    }
    base.update(changes)
    return Analysis(**base)


def entry(ident: str, **changes: Any) -> Entry:  # noqa: ANN401 - field values
    base: dict[str, Any] = {
        "id": ident,
        "file": f"tracks/{ident}.mp3",
        "title": f"Track {ident}",
        "artist": "Someone",
        "album": None,
        "duration": 240.0,
        "bytes": 1000,
        "mime": "audio/mpeg",
        "loudness": -12.0,
        "peak": 0.9,
        "gain_db": -4.0,
        "bpm": 85.0,
        "beat_offset": 0.1,
        "beat_strength": 0.7,
        "energy_raw": 1.0,
        "brightness": 1500.0,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://archive.org/details/x",
        "attribution": "credit",
        "added_at": "2026-10-14T00:00:00Z",
    }
    base.update(changes)
    return Entry(**base)


def flat(
    where_rank: dict[str, float], low: float = 1000.0, high: float = 2000.0
) -> Scale:
    return Scale(where_rank, low, high)


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    library = Library.open(tmp_path / "lib")
    library.add(entry("a1", tags={"lofi": "auto"}, provenance={"pool": "beats"}))
    library.save()
    again = Library.open(tmp_path / "lib")
    assert again.entries == library.entries
    assert not (tmp_path / "lib" / "library.json.tmp").exists()


def test_manifest_ignores_fields_it_does_not_know(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    Library.open(root)
    data = {"version": 1, "tracks": [{**entry("a1").__dict__, "from_the_future": 1}]}
    (root / "library.json").write_text(json.dumps(data), encoding="utf-8")
    assert Library.open(root).entries["a1"].title == "Track a1"


def test_find_by_prefix(tmp_path: Path) -> None:
    library = Library.open(tmp_path)
    for ident in ("abc1", "abd2"):
        library.add(entry(ident))
    assert library.find("ABC").id == "abc1"
    with pytest.raises(KeyError, match="2 tracks"):
        library.find("ab")
    with pytest.raises(KeyError, match="no track"):
        library.find("zz")


def test_seconds_and_sources(tmp_path: Path) -> None:
    library = Library.open(tmp_path)
    library.add(
        entry("a", provenance={"pool": "beats", "identifier": "i", "file": "f"})
    )
    library.add(entry("b", duration=60.0, provenance={"pool": "calm"}))
    assert library.seconds() == 300.0
    assert library.seconds(pool="beats") == 240.0
    assert library.has_source("i", "f")
    assert not library.has_source("i", "g")


def test_r2_key_and_credit() -> None:
    assert entry("0123").r2_key == "tracks/0123.mp3"
    assert entry("0123", file="tracks/0123.m4a").r2_key == "tracks/0123.m4a"
    assert credit("T", "A", "CC0 1.0", "u") == '"T" by A (CC0 1.0) - u'


def test_file_id_is_content_not_name(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"same")
    (tmp_path / "b").write_bytes(b"same")
    (tmp_path / "c").write_bytes(b"different")
    assert file_id(tmp_path / "a") == file_id(tmp_path / "b")
    assert file_id(tmp_path / "a") != file_id(tmp_path / "c")
    assert len(file_id(tmp_path / "a")) == 16


def test_transcode_keeps_the_audio(tmp_path: Path) -> None:
    dest = tmp_path / "t.mp3"
    transcode_to_mp3(music.track_path(), dest)
    info = sf.info(str(dest))
    assert info.format == "MP3"
    assert info.channels == 2
    assert info.duration == pytest.approx(music.SECONDS, abs=0.1)


def test_transcode_keeps_only_two_channels(tmp_path: Path) -> None:
    source = tmp_path / "quad.wav"
    sf.write(str(source), np.zeros((44100, 4)), 44100)
    transcode_to_mp3(source, tmp_path / "q.mp3")
    assert sf.info(str(tmp_path / "q.mp3")).channels == 2


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def test_subject_tags() -> None:
    tags = subject_tags(
        ["Instrumental Hip-Hop", "Lo-Fi", "jazzhop", "Chill Out", "Piano"]
    )
    assert tags == {"hip-hop", "instrumental", "lofi", "chillhop", "downtempo", "piano"}
    assert subject_tags(["Chillhopper"]) == set()
    assert subject_tags(["electronica"]) == {"electronic"}


def test_scale_ranks_energy_and_cuts_brightness() -> None:
    entries = [
        entry(str(i), energy_raw=float(i), brightness=100.0 * i) for i in range(1, 7)
    ]
    where = scale(entries)
    assert where.energy_rank["1"] == 0.0
    assert where.energy_rank["6"] == 1.0
    assert where.bright_low < where.bright_high
    assert scale([]).energy_rank == {}
    assert scale([entry("x")]).energy_rank == {"x": 0.5}


@pytest.mark.parametrize(
    ("changes", "rank", "has", "lacks"),
    [
        # a steady mid-tempo groove, calm: focus and create
        ({"bpm": 85.0}, 0.2, {"beat", "mid", "calm", "focus", "create"}, {"meditate"}),
        # busy: still creates, but not focus
        ({"bpm": 85.0}, 0.9, {"beat", "lively", "create"}, {"focus", "calm"}),
        # too fast for focus
        ({"bpm": 120.0}, 0.2, {"beat", "upbeat", "create"}, {"focus"}),
        # slow
        ({"bpm": 70.0}, 0.5, {"slow", "focus"}, {"calm", "lively"}),
        # beatless and still: meditate
        (
            {"bpm": None, "beat_strength": 0.1},
            0.1,
            {"beatless", "meditate", "calm"},
            {"beat", "create", "focus", "slow", "mid", "upbeat"},
        ),
        # beatless but not still: create, not meditate
        (
            {"bpm": None, "beat_strength": 0.1},
            0.5,
            {"beatless", "create"},
            {"meditate"},
        ),
        # a weak beat: neither beat nor beatless, so no mode at all
        (
            {"bpm": 90.0, "beat_strength": 0.3},
            0.5,
            {"mid"},
            {"beat", "beatless", "focus", "create", "meditate"},
        ),
    ],
)
def test_auto_tags_follow_the_contract(
    changes: dict[str, Any], rank: float, has: set[str], lacks: set[str]
) -> None:
    tags = auto_tags(entry("t", **changes), flat({"t": rank}))
    assert has <= tags
    assert not lacks & tags


def test_brightness_tags() -> None:
    where = flat({"t": 0.5}, low=1000.0, high=2000.0)
    assert "dark" in auto_tags(entry("t", brightness=500.0), where)
    assert "bright" in auto_tags(entry("t", brightness=2500.0), where)
    middle = auto_tags(entry("t", brightness=1500.0), where)
    assert not {"dark", "bright"} & middle


def test_classify_leaves_published_tracks_alone(tmp_path: Path) -> None:
    library = Library.open(tmp_path)
    library.add(
        entry("new", tags={"lofi": "auto", "focus": "auto", "mine": "user"}, bpm=120.0)
    )
    library.add(entry("old", in_d1=True, tags={"whatever": "user"}))
    classify(library)
    new = library.entries["new"].tags
    assert new["lofi"] == "auto"
    assert new["mine"] == "user"
    assert "focus" not in new  # 120 bpm is not focus: the stale tag went
    assert new["upbeat"] == "auto"
    assert library.entries["old"].tags == {"whatever": "user"}


def test_a_user_tag_is_not_overwritten_by_the_analyser(tmp_path: Path) -> None:
    library = Library.open(tmp_path)
    library.add(entry("t", tags={"focus": "user"}))
    classify(library)
    assert library.entries["t"].tags["focus"] == "user"


def test_edit_tags_before_and_after_publishing() -> None:
    local = entry("a", tags={"calm": "auto"})
    edit_tags(local, ["night"], ["calm"])
    assert local.tags == {"night": "user"}
    assert (local.pending_add, local.pending_remove) == ([], [])

    published = entry("b", in_d1=True, tags={"calm": "auto"})
    edit_tags(published, ["night", "rain"], ["calm"])
    assert published.pending_add == ["night", "rain"]
    assert published.pending_remove == ["calm"]
    edit_tags(published, ["calm"], ["rain"])
    assert published.pending_add == ["calm", "night"]
    assert published.pending_remove == ["rain"]


@pytest.mark.parametrize("bad", ["Upper", "-lead", "has space", "x" * 25, ""])
def test_edit_tags_refuses_non_tags(bad: str) -> None:
    with pytest.raises(ValueError, match="is not a tag"):
        edit_tags(entry("a"), [bad], [])


# ---------------------------------------------------------------------------
# adding files you have
# ---------------------------------------------------------------------------


def add(library: Library, path: Path, **changes: Any) -> Entry:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "title": None,
        "artist": "Me",
        "album": None,
        "license_name": "Personal use (purchased)",
        "license_url": "https://me.bandcamp.com/x",
        "source_url": "https://me.bandcamp.com/x",
        "measure": lambda _p: measured(),
    }
    kwargs.update(changes)
    return add_file(library, path, **kwargs)


def test_add_a_wav_converts_it(tmp_path: Path) -> None:
    library = Library.open(tmp_path / "lib")
    added = add(library, music.track_path(), tags=["night"])
    assert added.file == f"tracks/{added.id}.mp3"
    assert added.mime == "audio/mpeg"
    assert (library.root / added.file).is_file()
    assert added.title == music.track_path().stem
    assert added.tags == {"night": "user"}
    assert added.attribution.startswith(f'"{added.title}" by Me (Personal use')
    assert added.provenance == {"kind": "file", "original": music.track_path().name}
    assert added.bytes == (library.root / added.file).stat().st_size


def test_add_an_mp3_copies_it_and_measures_the_copy(tmp_path: Path) -> None:
    mp3 = tmp_path / "song.mp3"
    transcode_to_mp3(music.track_path(), mp3)
    library = Library.open(tmp_path / "lib")
    added = add(library, mp3, title="Song", measure=analyse)
    assert added.id == file_id(mp3)
    assert added.title == "Song"
    assert added.bpm == 120.0
    again = add(library, mp3, title="Other")
    assert again is added
    assert len(library.entries) == 1


@pytest.mark.parametrize(
    ("name", "match"), [("a.m4a", "AAC"), ("a.txt", "expected MP3")]
)
def test_add_refuses_what_it_cannot_measure(
    tmp_path: Path, name: str, match: str
) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match=match):
        add(Library.open(tmp_path / "lib"), path)


def test_add_refuses_bad_tags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a tag"):
        add(Library.open(tmp_path / "lib"), music.track_path(), tags=["Bad Tag"])


def test_reanalyse_updates_measurements_only(tmp_path: Path) -> None:
    library = Library.open(tmp_path)
    for i in range(55):
        library.add(entry(f"{i:02d}", title="kept", tags={"x": "user"}))
    count = reanalyse(
        library, lambda _p: measured(bpm=99.0, beat_offset=0.25, energy=2.0)
    )
    assert count == 55
    assert {e.bpm for e in library.entries.values()} == {99.0}
    assert {e.energy_raw for e in library.entries.values()} == {2.0}
    assert {e.title for e in library.entries.values()} == {"kept"}
    assert Library.open(tmp_path).entries["00"].beat_offset == 0.25


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


class Archive:
    """A small fake Internet Archive: searches, item metadata and downloads."""

    def __init__(self) -> None:
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.files: dict[str, list[dict[str, Any]]] = {}
        self.downloads: list[str] = []
        self.broken: set[str] = set()
        self.flac: set[str] = set()

    def item(self, ident: str, query: str, *tracks: dict[str, Any], **doc: Any) -> None:  # noqa: ANN401
        """List an item under a search query, with its files."""
        hit = {"identifier": ident, "title": ident, "creator": "Band", "licenseurl": BY}
        hit.update(doc)
        self.results.setdefault(query, []).append(hit)
        self.files[ident] = list(tracks)

    def get_json(self, url: str) -> Any:  # noqa: ANN401
        """Search pages and metadata."""
        if url.startswith(SEARCH_URL):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            page = int(query["page"][0])
            docs = self.results.get(query["q"][0], []) if page == 1 else []
            return {"response": {"docs": docs}}
        ident = urllib.parse.unquote(url.rsplit("/", 1)[1])
        if ident in self.broken:
            raise OSError(ident)
        return {"files": self.files[ident]}

    def download(self, url: str, dest: Path) -> None:
        """Write the URL as the content, or a real FLAC for the transcoding test."""
        if url in self.broken:
            raise OSError(url)
        self.downloads.append(url)
        if url in self.flac:
            data, rate = sf.read(str(music.track_path()), always_2d=True)
            sf.write(str(dest), data[: rate * 2], rate, format="FLAC")
        else:
            dest.write_bytes(url.encode())


def track(name: str, seconds: float = 240.0, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {"name": name, "format": "VBR MP3", "length": str(seconds), **extra}


def by_url(table: dict[str, Analysis], default: Analysis) -> Callable[[Path], Analysis]:
    """A fake analyser that knows each file by its content (the URL it came from)."""

    def measure(path: Path) -> Analysis:
        content = path.read_bytes()
        for key, value in table.items():
            if content.endswith(key.encode()):
                return value
        return default

    return measure


def run_fetch(
    library: Library,
    archive: Archive,
    pools: tuple[Pool, ...],
    measure: Callable[[Path], Analysis] = lambda _p: measured(),
) -> lib.FetchReport:
    rules = {"beats": lib.ACCEPT["beats"], "ambient": lib.ACCEPT["ambient"]}
    return fetch(library, archive, pools, accept=rules, measure=measure, workers=2)


def test_fetch_fills_pools_and_remembers_what_it_turned_down(tmp_path: Path) -> None:
    archive = Archive()
    archive.item(
        "tape",
        "qb",
        track("01 one.mp3", track="1"),
        track("02 two.mp3", track="2"),
        track("03 skit.mp3", seconds=30.0, track="3"),
        track("04 feat. MC.mp3", track="4"),
        track("05 drift.mp3", track="5"),
        subject=["lofi", "instrumental hip-hop"],
    )
    archive.item("rapper", "qb", track("r.mp3"), subject=["rap"])
    archive.item(
        "nd",
        "qb",
        track("n.mp3"),
        licenseurl="http://creativecommons.org/licenses/by-nd/4.0/",
    )
    archive.item("still", "qa", track("pad.mp3"), subject=["ambient", "piano"])
    measure = by_url(
        {
            "05%20drift.mp3": measured(bpm=None, beat_strength=0.05),
            "pad.mp3": measured(bpm=None, beat_strength=0.05, duration=600.0),
        },
        measured(),
    )
    library = Library.open(tmp_path / "lib")
    pools = (Pool("beats", 1.0, ("qb",)), Pool("ambient", 1.0, ("qa",)))
    report = run_fetch(library, archive, pools, measure)

    assert report.added == {"beats": 2, "ambient": 1}
    assert report.rejected == 1  # drift: beatless, wrong pool
    titles = sorted(e.title for e in library.entries.values())
    assert titles == ["01 one", "02 two", "pad"]
    beats = next(e for e in library.entries.values() if e.title == "01 one")
    assert beats.provenance == {
        "kind": "archive",
        "identifier": "tape",
        "file": "01 one.mp3",
        "pool": "beats",
    }
    assert beats.tags == {"lofi": "auto", "hip-hop": "auto", "instrumental": "auto"}
    assert beats.license == "CC BY 4.0"
    assert beats.source_url == "https://archive.org/details/tape"
    assert (
        beats.attribution
        == '"01 one" by Band (CC BY 4.0) - https://archive.org/details/tape'
    )
    assert not any("rapper" in url or "/nd/" in url for url in archive.downloads)
    assert not any("skit" in url or "feat" in url for url in archive.downloads)
    assert not list((library.root / "incoming").glob("*"))

    state = lib.FetchState.load(library.root)
    assert state.done_items == {"tape", "still"}
    assert "tape/05 drift.mp3" in state.rejected
    assert "tape/03 skit.mp3" in state.rejected

    before = len(archive.downloads)
    again = run_fetch(library, archive, (Pool("beats", 2.0, ("qb",)),), measure)
    assert len(archive.downloads) == before
    assert again.added == {"beats": 0}


def test_fetch_stops_when_the_pool_is_full_and_comes_back_for_more(
    tmp_path: Path,
) -> None:
    archive = Archive()
    archive.item("tape", "qb", *(track(f"{i}.mp3", track=str(i)) for i in range(1, 6)))
    archive.item("later", "qb", track("z.mp3"))
    library = Library.open(tmp_path / "lib")
    small = (Pool("beats", 400.0 / 3600.0, ("qb",)),)
    report = run_fetch(library, archive, small)
    assert report.added == {"beats": 2}
    assert not any("later" in url for url in archive.downloads)
    assert "tape" not in lib.FetchState.load(library.root).done_items
    assert lib.FetchState.load(library.root).rejected == set()

    bigger = (Pool("beats", 1.0, ("qb",)),)
    assert run_fetch(library, archive, bigger).added == {"beats": 4}
    assert "tape" in lib.FetchState.load(library.root).done_items


def test_fetch_caps_items_and_artists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lib, "ITEM_CAP", 500.0)
    monkeypatch.setattr(lib, "ARTIST_CAP", 700.0)
    archive = Archive()
    archive.item("one", "qb", *(track(f"a{i}.mp3", track=str(i)) for i in range(5)))
    archive.item("two", "qb", *(track(f"b{i}.mp3", track=str(i)) for i in range(5)))
    library = Library.open(tmp_path / "lib")
    report = run_fetch(library, archive, (Pool("beats", 5.0, ("qb",)),))
    # 240 s tracks: two per item by the item cap; two by the artist cap.
    assert report.added == {"beats": 2}
    assert {e.provenance["identifier"] for e in library.entries.values()} == {"one"}


def test_fetch_retries_the_network_but_not_bad_files(tmp_path: Path) -> None:
    archive = Archive()
    archive.item("tape", "qb", track("good.mp3"), track("flaky.mp3"), track("junk.mp3"))
    archive.item("gone", "qb", track("x.mp3"))
    archive.broken = {"https://archive.org/download/tape/flaky.mp3", "gone"}

    def measure(path: Path) -> Analysis:
        if path.read_bytes().endswith(b"junk.mp3"):
            msg = "cannot decode"
            raise ValueError(msg)
        return measured()

    library = Library.open(tmp_path / "lib")
    log: list[str] = []
    rules = {"beats": lib.ACCEPT["beats"]}
    fetch(
        library,
        archive,
        (Pool("beats", 5.0, ("qb",)),),
        accept=rules,
        measure=measure,
        log=log.append,
    )
    state = lib.FetchState.load(library.root)
    assert state.rejected == {"tape/junk.mp3"}
    assert state.done_items == set()
    assert any("failed tape/flaky.mp3" in line for line in log)
    assert any("skip gone" in line for line in log)
    assert len(library.entries) == 1

    archive.broken = set()
    fetch(
        library, archive, (Pool("beats", 5.0, ("qb",)),), accept=rules, measure=measure
    )
    assert len(library.entries) == 3


def test_fetch_transcodes_tracks_that_only_come_lossless(tmp_path: Path) -> None:
    archive = Archive()
    archive.item("tape", "qb", {"name": "a.flac", "format": "Flac"})
    archive.flac = {"https://archive.org/download/tape/a.flac"}
    library = Library.open(tmp_path / "lib")
    run_fetch(library, archive, (Pool("beats", 1.0, ("qb",)),))
    [only] = library.entries.values()
    assert only.file.endswith(".mp3")
    assert sf.info(str(library.path(only))).format == "MP3"


class Mirrors(Archive):
    """Every download is the same file, as when two items mirror one release."""

    def download(self, url: str, dest: Path) -> None:
        """Write the same bytes whatever was asked for."""
        self.downloads.append(url)
        dest.write_bytes(b"same bytes")


def test_fetch_keeps_one_copy_of_identical_files(tmp_path: Path) -> None:
    archive = Mirrors()
    archive.item("tape", "qb", track("a.mp3"))
    archive.item("copy", "qb", track("a.mp3"))
    library = Library.open(tmp_path / "lib")
    run_fetch(library, archive, (Pool("beats", 1.0, ("qb",)),))
    assert len(library.entries) == 1


def test_fetch_skips_full_pools_without_searching(tmp_path: Path) -> None:
    library = Library.open(tmp_path / "lib")
    library.add(entry("a", duration=3600.0, provenance={"pool": "beats"}))
    archive = Archive()
    report = run_fetch(library, archive, (Pool("beats", 1.0, ("qb",)),))
    assert report.seconds == {"beats": 3600.0}
    assert archive.downloads == []


@pytest.mark.parametrize(
    ("pool", "analysis", "kept"),
    [
        ("beats", measured(), True),
        ("beats", measured(bpm=130.0), False),
        ("beats", measured(beat_strength=0.3), False),
        ("beats", measured(bpm=None, beat_strength=0.5), False),
        ("ambient", measured(bpm=None, beat_strength=0.1), True),
        ("ambient", measured(beat_strength=0.3), False),
        ("calm", measured(beat_strength=0.3), True),
    ],
)
def test_pool_rules(pool: str, analysis: Analysis, kept: bool) -> None:  # noqa: FBT001
    assert lib.ACCEPT[pool](analysis) is kept


def test_default_pools_add_up_to_forty_hours() -> None:
    assert sum(p.hours for p in lib.DEFAULT_POOLS) == 40.0
    assert {p.name for p in lib.DEFAULT_POOLS} == set(lib.ACCEPT)
    for pool in lib.DEFAULT_POOLS:
        for query in pool.queries:
            assert query.startswith("mediatype:audio AND ")
            assert query.count("(") == query.count(")")


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------


class Cloud:
    """A fake ``npx wrangler``: R2 is a dict, D1 is SQLite with the real schema."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA foreign_keys = ON")
        for migration in sorted(MIGRATIONS.glob("*.sql")):
            self.db.executescript(migration.read_text(encoding="utf-8"))
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.commands: list[list[str]] = []
        self.fail: str | None = None

    def __call__(
        self, command: list[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        """Do what wrangler would, to the fakes."""
        self.commands.append(command)
        assert command[:3] == ["npx", "--yes", lib.WRANGLER]
        assert cwd.name == "web"
        if self.fail and self.fail in command:
            return subprocess.CompletedProcess(
                command, 1, "", "line\nauthentication error"
            )
        if command[3:5] == ["r2", "object"]:
            key = command[6].removeprefix(f"{lib.BUCKET}/")
            path = Path(command[command.index("--file") + 1])
            self.objects[key] = (
                path.read_bytes(),
                command[command.index("--content-type") + 1],
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--command" in command:
            rows = [{"id": r[0]} for r in self.db.execute("SELECT id FROM tracks")]
            out = "noise before the json\n" + json.dumps(
                [{"results": rows, "success": True}]
            )
            return subprocess.CompletedProcess(command, 0, out, "")
        sql = Path(command[command.index("--file") + 1]).read_text(encoding="utf-8")
        self.db.executescript(sql)
        return subprocess.CompletedProcess(command, 0, "", "")

    def tags(self, ident: str) -> dict[str, str]:
        """A track's tags in D1, with their sources."""
        rows = self.db.execute(
            "SELECT tag, source FROM track_tags WHERE track_id = ?", (ident,)
        )
        return dict(rows.fetchall())

    def row(self, ident: str) -> dict[str, Any]:
        """A track's row in D1, by column."""
        cursor = self.db.execute("SELECT * FROM tracks WHERE id = ?", (ident,))
        names = [d[0] for d in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Cloud:
    monkeypatch.setattr("shutil.which", lambda _name: "npx")
    return Cloud()


def stocked(root: Path, count: int = 3) -> Library:
    library = Library.open(root)
    for i in range(count):
        ident = f"{i:016x}"
        (root / "tracks" / f"{ident}.mp3").write_bytes(f"audio {i}".encode())
        library.add(
            entry(
                ident,
                energy_raw=float(i),
                brightness=1000.0 + 500.0 * i,
                tags={"lofi": "auto", "night": "user"},
            )
        )
    library.save()
    return library


def test_publish_uploads_and_inserts(tmp_path: Path, cloud: Cloud) -> None:
    library = stocked(tmp_path / "lib")
    report = publish(library, Wrangler(tmp_path / "web", run=cloud))
    assert (report.uploaded, report.rows, report.new_rows) == (3, 3, 3)
    assert set(cloud.objects) == {e.r2_key for e in library.entries.values()}
    assert cloud.objects["tracks/0000000000000001.mp3"] == (b"audio 1", "audio/mpeg")
    first = cloud.row("0000000000000000")
    assert first["r2_key"] == "tracks/0000000000000000.mp3"
    assert first["energy"] == 0.0
    assert cloud.row("0000000000000002")["energy"] == 1.0
    tags = cloud.tags("0000000000000000")
    assert tags["lofi"] == "auto"
    assert tags["night"] == "user"
    assert tags["focus"] == "auto"
    assert all(
        e.in_d1 and e.uploaded_at
        for e in Library.open(tmp_path / "lib").entries.values()
    )
    assert not list((tmp_path / "lib").glob("publish-*.sql"))


def test_a_second_publish_uploads_nothing_and_leaves_tags_alone(
    tmp_path: Path, cloud: Cloud
) -> None:
    library = stocked(tmp_path / "lib")
    cloud_run = Wrangler(tmp_path / "web", run=cloud)
    publish(library, cloud_run)
    # Someone removes a tag in the app, and the analyser is improved here.
    cloud.db.execute(
        "DELETE FROM track_tags WHERE track_id = '0000000000000000' AND tag = 'lofi'"
    )
    library.entries["0000000000000000"].bpm = 86.5
    report = publish(Library.open(tmp_path / "lib"), cloud_run)
    assert report.uploaded == 0
    assert report.new_rows == 0
    assert "lofi" not in cloud.tags("0000000000000000")
    assert (
        cloud.row("0000000000000000")["bpm"] == 85.0
    )  # the in-memory change was not saved
    library = Library.open(tmp_path / "lib")
    library.entries["0000000000000000"].bpm = 86.5
    publish(library, cloud_run)
    assert cloud.row("0000000000000000")["bpm"] == 86.5


def test_publish_applies_queued_tag_edits(tmp_path: Path, cloud: Cloud) -> None:
    run = Wrangler(tmp_path / "web", run=cloud)
    publish(stocked(tmp_path / "lib"), run)
    library = Library.open(tmp_path / "lib")
    target = library.entries["0000000000000001"]
    edit_tags(target, ["rain", "lofi"], ["night"])
    library.save()
    report = publish(library, run)
    assert report.tag_edits == 3
    tags = cloud.tags("0000000000000001")
    assert tags["rain"] == "user"
    assert tags["lofi"] == "user"
    assert "night" not in tags
    again = Library.open(tmp_path / "lib").entries["0000000000000001"]
    assert (again.pending_add, again.pending_remove) == ([], [])


def test_publish_trusts_d1_about_what_it_has(tmp_path: Path, cloud: Cloud) -> None:
    """A manifest that forgot it published (copied, restored) resends no tags."""
    run = Wrangler(tmp_path / "web", run=cloud)
    publish(stocked(tmp_path / "lib"), run)
    cloud.db.execute("DELETE FROM track_tags")
    library = Library.open(tmp_path / "lib")
    for item in library.entries.values():
        item.in_d1 = False
    publish(library, run)
    assert cloud.db.execute("SELECT count(*) FROM track_tags").fetchone() == (0,)


def test_publish_in_batches(tmp_path: Path, cloud: Cloud) -> None:
    library = stocked(tmp_path / "lib", count=5)
    publish(library, Wrangler(tmp_path / "web", run=cloud), batch=2)
    files = [c for c in cloud.commands if "--file" in c and "d1" in c]
    assert len(files) == 3
    assert cloud.db.execute("SELECT count(*) FROM tracks").fetchone() == (5,)


def test_publish_reports_wrangler_failures(tmp_path: Path, cloud: Cloud) -> None:
    cloud.fail = "--command"
    with pytest.raises(
        RuntimeError,
        match="reading track ids from D1 failed:\nline\nauthentication error",
    ):
        publish(stocked(tmp_path / "lib"), Wrangler(tmp_path / "web", run=cloud))


def test_publish_reports_a_failed_upload(tmp_path: Path, cloud: Cloud) -> None:
    cloud.fail = "r2"
    with pytest.raises(RuntimeError, match="upload of tracks/"):
        publish(
            stocked(tmp_path / "lib", count=1), Wrangler(tmp_path / "web", run=cloud)
        )


def test_publish_needs_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="npx was not found"):
        Wrangler(tmp_path / "web").track_ids()


def test_track_ids_needs_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "npx")

    def run(command: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "no json here", "")

    with pytest.raises(RuntimeError, match="unexpected output"):
        Wrangler(tmp_path / "web", run=run).track_ids()


def test_the_real_runner_runs_a_program(tmp_path: Path) -> None:
    result = lib._run(["git", "--version"], tmp_path)
    assert result.returncode == 0
    assert "git version" in result.stdout


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        (None, "NULL"),
        (True, "1"),
        (3, "3"),
        (0.25, "0.25"),
        (float("nan"), "NULL"),
        (float("inf"), "NULL"),
        ("it's", "'it''s'"),
        ("a\x00b", "'ab'"),
    ],
)
def test_sql_literal(value: object, literal: str) -> None:
    assert sql_literal(value) == literal


def test_awkward_text_survives_the_trip(tmp_path: Path, cloud: Cloud) -> None:
    library = Library.open(tmp_path / "lib")
    title = "Rock 'n' Roll\"; DROP TABLE tracks; --\x00 ünïcødé"
    (library.root / "tracks" / "00000000000000aa.mp3").write_bytes(b"x")
    library.add(entry("00000000000000aa", title=title))
    publish(library, Wrangler(tmp_path / "web", run=cloud))
    assert cloud.row("00000000000000aa")["title"] == title.replace("\x00", "")


def test_publish_sql_is_valid_on_its_own() -> None:
    db = sqlite3.connect(":memory:")
    for migration in sorted(MIGRATIONS.glob("*.sql")):
        db.executescript(migration.read_text(encoding="utf-8"))
    sql = publish_sql([entry("a", tags={"x": "auto"})], Scale({"a": 0.5}, 0.0, 1.0))
    db.executescript(sql)
    db.executescript(sql)  # twice: an upsert, not a duplicate
    assert db.execute("SELECT count(*) FROM tracks").fetchone() == (1,)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def invoke(*args: str) -> Any:  # noqa: ANN401 - a click Result
    return CliRunner().invoke(app, ["library", *args])


def test_cli_add_list_and_tag(tmp_path: Path) -> None:
    root = str(tmp_path / "lib")
    added = invoke(
        "add",
        str(music.track_path()),
        "--artist",
        "Me",
        "--source-url",
        "https://me.bandcamp.com",
        "--tag",
        "night",
        "--root",
        root,
    )
    assert added.exit_code == 0, added.output
    assert "120 bpm" in added.output
    ident = added.output.split()[0]

    listed = invoke("list", "--root", root)
    assert listed.exit_code == 0
    assert "Me - track-v1" in listed.output
    assert "1 tracks" in listed.output
    assert "0 tracks" in invoke("list", "--tag", "nonesuch", "--root", root).output

    tagged = invoke("tag", ident[:6], "+rain", "-night", "--root", root)
    assert tagged.exit_code == 0, tagged.output
    assert "rain" in tagged.output
    assert "night" not in tagged.output
    entry_ = Library.open(Path(root)).entries[ident]
    assert entry_.tags["rain"] == "user"
    assert entry_.license_url == "https://me.bandcamp.com"


def test_cli_add_refusals(tmp_path: Path) -> None:
    root = str(tmp_path / "lib")
    common = ("--artist", "Me", "--source-url", "u", "--root", root)
    two = invoke(
        "add",
        str(music.track_path()),
        str(music.track_path(22050)),
        "--title",
        "x",
        *common,
    )
    assert two.exit_code == 1
    assert "--title names one track" in two.output
    m4a = tmp_path / "a.m4a"
    m4a.write_bytes(b"x")
    assert "AAC" in invoke("add", str(m4a), *common).output


def test_cli_tag_refusals(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    library = Library.open(root)
    library.add(entry("abc"))
    library.save()
    assert "no track" in invoke("tag", "zz", "+a", "--root", str(root)).output
    assert "say +tag or -tag" in invoke("tag", "abc", "a", "--root", str(root)).output
    assert "is not a tag" in invoke("tag", "abc", "+A", "--root", str(root)).output


def test_cli_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fetch(
        _library: Library,
        _http: object,
        pools: tuple[Pool, ...],
        **kwargs: object,
    ) -> lib.FetchReport:
        calls.append({"pools": pools, **kwargs})
        report = lib.FetchReport()
        for pool in pools:
            report.added[pool.name] = 1
            report.seconds[pool.name] = 3600.0
        return report

    monkeypatch.setattr(library_cli, "fetch", fake_fetch)
    result = invoke("fetch", "--hours", "20", "--root", str(tmp_path / "lib"))
    assert result.exit_code == 0, result.output
    assert calls[0]["scale_hours"] == 0.5
    assert calls[0]["workers"] == 4
    assert "beats    +1 tracks, 1.0 h" in result.output

    custom = invoke(
        "fetch", "--hours", "3", "-q", "subject:piano", "--root", str(tmp_path / "lib")
    )
    assert custom.exit_code == 0
    [pool] = calls[1]["pools"]
    assert (pool.name, pool.hours, tuple(pool.queries)) == (
        "custom",
        3.0,
        ("subject:piano",),
    )


def test_cli_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    web = tmp_path / "web"
    web.mkdir()

    def fake_publish(
        _library: Library, cloud_: Wrangler, **_kw: object
    ) -> lib.PublishReport:
        assert cloud_.web == web
        return lib.PublishReport(uploaded=2, rows=5, new_rows=2, tag_edits=1)

    monkeypatch.setattr(library_cli, "publish", fake_publish)
    result = invoke("publish", "--web", str(web), "--root", str(tmp_path / "lib"))
    assert result.exit_code == 0, result.output
    assert (
        "uploaded 2 files; 5 rows in D1 (2 new, 1 tag edits applied)" in result.output
    )

    def failing(*_a: Any, **_kw: Any) -> lib.PublishReport:  # noqa: ANN401
        msg = "upload of tracks/x failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(library_cli, "publish", failing)
    failed = invoke("publish", "--web", str(web), "--root", str(tmp_path / "lib"))
    assert failed.exit_code == 1
    assert "upload of tracks/x failed" in failed.output


def test_cli_analyse_measures_again(tmp_path: Path) -> None:
    library = Library.open(tmp_path / "lib")
    added = add(library, music.track_path(), measure=lambda _p: measured(bpm=91.0))
    library.save()
    result = invoke("analyse", "--root", str(tmp_path / "lib"))
    assert result.exit_code == 0, result.output
    assert "measured 1 tracks" in result.output
    assert Library.open(tmp_path / "lib").entries[added.id].bpm == 120.0


def test_wrangler_tries_again_before_giving_up(tmp_path: Path, cloud: Cloud) -> None:
    hiccups = [1]

    def flaky(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if hiccups:
            hiccups.pop()
            return subprocess.CompletedProcess(command, 1, "", "APIError 7403")
        return cloud(command, cwd)

    report = publish(stocked(tmp_path / "lib"), Wrangler(tmp_path / "web", run=flaky))
    assert report.rows == 3
    cloud.fail = "--command"
    before = sum("--command" in c for c in cloud.commands)
    with pytest.raises(RuntimeError, match="authentication error"):
        Wrangler(tmp_path / "web", run=cloud, attempts=2).track_ids()
    assert sum("--command" in c for c in cloud.commands) == before + 2


def test_cli_output_survives_a_narrow_console(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    narrow = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", narrow)
    library_cli.echo("♫ Ennui — été")
    narrow.flush()
    assert buffer.getvalue().decode("cp1252").startswith("? Ennui — été")
