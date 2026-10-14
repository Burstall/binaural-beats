-- The app's whole database. Tracks and their tags are shared by everyone who
-- can sign in; profiles, ratings, playlists and sessions belong to one person,
-- keyed by the email address Cloudflare Access vouched for.

-- One row per audio file in the R2 bucket. Written by `violet library publish`;
-- the app only reads it. `id` is the first 16 hex digits of the SHA-256 of the
-- uploaded file, so the same file can never be listed twice and its R2 key
-- never changes meaning.
CREATE TABLE tracks (
  id            TEXT PRIMARY KEY,
  r2_key        TEXT NOT NULL UNIQUE,   -- e.g. tracks/0123456789abcdef.mp3
  title         TEXT NOT NULL,
  artist        TEXT NOT NULL,
  album         TEXT,
  duration      REAL NOT NULL,          -- seconds
  bytes         INTEGER NOT NULL,
  mime          TEXT NOT NULL,          -- audio/mpeg, audio/mp4, ...
  loudness      REAL,                   -- integrated loudness, LUFS (BS.1770)
  peak          REAL,                   -- sample peak, linear
  gain_db       REAL NOT NULL DEFAULT 0,-- playback gain to the library target
  bpm           REAL,                   -- NULL when there is no steady beat
  beat_offset   REAL,                   -- seconds to the first beat
  beat_strength REAL,                   -- 0..1, how clearly there is a beat
  energy        REAL,                   -- 0..1, relative within the library
  brightness    REAL,                   -- spectral centroid, Hz
  license       TEXT NOT NULL,          -- short name, e.g. "CC BY 4.0"
  license_url   TEXT NOT NULL,
  source_url    TEXT NOT NULL,
  attribution   TEXT NOT NULL,          -- ready-to-print credit line
  added_at      TEXT NOT NULL           -- ISO 8601 UTC
);

-- Shared tags. `source` says who put it there: the analyser at publish time,
-- or a person in the app. Publishing only ever adds auto tags to tracks it is
-- inserting for the first time, so a tag someone removed stays removed.
CREATE TABLE track_tags (
  track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  tag      TEXT NOT NULL,
  source   TEXT NOT NULL CHECK (source IN ('auto', 'user')),
  PRIMARY KEY (track_id, tag)
);
CREATE INDEX track_tags_by_tag ON track_tags(tag);

-- Everything else about a person's settings: an opaque JSON document the app
-- owns. The worker checks only that it is JSON and not huge.
CREATE TABLE profiles (
  email      TEXT PRIMARY KEY,
  data       TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE ratings (
  email      TEXT NOT NULL,
  track_id   TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  value      INTEGER NOT NULL CHECK (value IN (-1, 1)),
  updated_at TEXT NOT NULL,
  PRIMARY KEY (email, track_id)
);

CREATE TABLE playlists (
  id         TEXT PRIMARY KEY,          -- chosen by the client, random
  email      TEXT NOT NULL,
  name       TEXT NOT NULL,
  track_ids  TEXT NOT NULL,             -- JSON array of track ids, in order
  updated_at TEXT NOT NULL
);
CREATE INDEX playlists_by_email ON playlists(email);

-- One row per listening session. `arm` is the blinded condition: 'on' had the
-- pulse and binaural layer, 'off' had neither, NULL means the session was not
-- part of the blind test. The app does not show the arm until it is rated.
CREATE TABLE sessions (
  id         TEXT PRIMARY KEY,          -- chosen by the client, random
  email      TEXT NOT NULL,
  mode       TEXT NOT NULL CHECK (mode IN ('focus', 'create', 'meditate')),
  started_at TEXT NOT NULL,
  minutes    REAL NOT NULL DEFAULT 0,
  arm        TEXT CHECK (arm IN ('on', 'off')),
  rating     INTEGER CHECK (rating BETWEEN 1 AND 5),
  note       TEXT,
  track_ids  TEXT NOT NULL DEFAULT '[]' -- JSON array, in the order played
);
CREATE INDEX sessions_by_email ON sessions(email, started_at);
