// The JSON API. Every handler gets the signed-in email and touches only that
// person's rows; tracks and tags are shared.

import { HttpError, json, readJson } from "./http.js";

export const TAG_RULE = /^[a-z0-9][a-z0-9-]{0,23}$/;
export const ID_RULE = /^[A-Za-z0-9_-]{8,40}$/;
export const MAX_TAGS = 32;
export const MAX_PLAYLIST_TRACKS = 2000;
export const MAX_NAME = 80;
export const MAX_NOTE = 2000;
export const PROFILE_LIMIT = 65536;
const MODES = new Set(["focus", "create", "meditate"]);

const nowIso = () => new Date().toISOString();

function isIso(s) {
  return typeof s === "string" && s.length <= 40 && /^\d{4}-\d{2}-\d{2}T/.test(s) && Number.isFinite(Date.parse(s));
}

function requireId(id, what = "id") {
  if (!ID_RULE.test(id)) throw new HttpError(400, `bad ${what}`);
  return id;
}

function requireObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new HttpError(400, "body must be a JSON object");
  }
  return value;
}

function idList(value, max, what) {
  if (!Array.isArray(value)) throw new HttpError(400, `${what} must be an array`);
  if (value.length > max) throw new HttpError(400, `too many ${what}`);
  for (const id of value) if (typeof id !== "string" || !ID_RULE.test(id)) throw new HttpError(400, `bad id in ${what}`);
  return value;
}

function tagList(value, what) {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > MAX_TAGS) throw new HttpError(400, `${what} must be an array of at most ${MAX_TAGS} tags`);
  for (const tag of value) if (typeof tag !== "string" || !TAG_RULE.test(tag)) throw new HttpError(400, `bad tag ${JSON.stringify(tag)}`);
  return [...new Set(value)];
}

async function trackExists(db, id) {
  const row = await db.prepare("SELECT 1 AS ok FROM tracks WHERE id = ?").bind(id).first();
  if (!row) throw new HttpError(404, "no such track");
}

function parseList(text) {
  try {
    const value = JSON.parse(text);
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------

async function getMe({ db, email }) {
  const row = await db.prepare("SELECT data FROM profiles WHERE email = ?").bind(email).first();
  let profile = null;
  if (row) {
    try {
      profile = JSON.parse(row.data);
    } catch {
      profile = null;
    }
  }
  return json({ email, profile });
}

async function putProfile({ db, email, request }) {
  const { value } = await readJson(request, PROFILE_LIMIT);
  requireObject(value);
  const updated = nowIso();
  await db
    .prepare(
      "INSERT INTO profiles (email, data, updated_at) VALUES (?, ?, ?) " +
        "ON CONFLICT (email) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
    )
    .bind(email, JSON.stringify(value), updated)
    .run();
  return json({ ok: true, updated_at: updated });
}

export function trackOut(row, tags) {
  const { r2_key: key, ...rest } = row;
  return { ...rest, tags: tags ?? [], url: `/media/${key}` };
}

async function getLibrary({ db }) {
  const [tracks, tags] = await db.batch([
    db.prepare("SELECT * FROM tracks ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE"),
    db.prepare("SELECT track_id, tag FROM track_tags ORDER BY tag"),
  ]);
  const byTrack = new Map();
  for (const { track_id: id, tag } of tags.results) {
    if (!byTrack.has(id)) byTrack.set(id, []);
    byTrack.get(id).push(tag);
  }
  let updated = null;
  const out = tracks.results.map((row) => {
    if (!updated || row.added_at > updated) updated = row.added_at;
    return trackOut(row, byTrack.get(row.id));
  });
  return json({ tracks: out, updated_at: updated });
}

async function putTags({ db, request, params }) {
  const id = requireId(params.id, "track id");
  const body = requireObject((await readJson(request)).value);
  const add = tagList(body.add, "add");
  const remove = tagList(body.remove, "remove");
  await trackExists(db, id);
  const current = (await db.prepare("SELECT tag FROM track_tags WHERE track_id = ?").bind(id).all()).results.map((r) => r.tag);
  const next = new Set(current);
  for (const tag of remove) next.delete(tag);
  for (const tag of add) next.add(tag);
  if (next.size > MAX_TAGS) throw new HttpError(400, `a track can carry at most ${MAX_TAGS} tags`);
  const statements = [];
  for (const tag of remove) {
    if (!add.includes(tag)) statements.push(db.prepare("DELETE FROM track_tags WHERE track_id = ? AND tag = ?").bind(id, tag));
  }
  for (const tag of add) {
    statements.push(
      db
        .prepare("INSERT INTO track_tags (track_id, tag, source) VALUES (?, ?, 'user') ON CONFLICT (track_id, tag) DO NOTHING")
        .bind(id, tag),
    );
  }
  if (statements.length) await db.batch(statements);
  return json({ tags: [...next].sort() });
}

async function getRatings({ db, email }) {
  const { results } = await db.prepare("SELECT track_id, value FROM ratings WHERE email = ?").bind(email).all();
  const ratings = {};
  for (const r of results) ratings[r.track_id] = r.value;
  return json({ ratings });
}

async function putRating({ db, email, request, params }) {
  const id = requireId(params.id, "track id");
  const body = requireObject((await readJson(request)).value);
  if (![-1, 0, 1].includes(body.value)) throw new HttpError(400, "value must be -1, 0 or 1");
  await trackExists(db, id);
  if (body.value === 0) {
    await db.prepare("DELETE FROM ratings WHERE email = ? AND track_id = ?").bind(email, id).run();
  } else {
    await db
      .prepare(
        "INSERT INTO ratings (email, track_id, value, updated_at) VALUES (?, ?, ?, ?) " +
          "ON CONFLICT (email, track_id) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
      )
      .bind(email, id, body.value, nowIso())
      .run();
  }
  return json({ ok: true });
}

async function getPlaylists({ db, email }) {
  const { results } = await db
    .prepare("SELECT id, name, track_ids, updated_at FROM playlists WHERE email = ? ORDER BY name COLLATE NOCASE")
    .bind(email)
    .all();
  return json({ playlists: results.map((r) => ({ ...r, track_ids: parseList(r.track_ids) })) });
}

/** 404 when the row exists but belongs to someone else, so ids reveal nothing. */
async function assertOwner(db, table, id, email) {
  const row = await db.prepare(`SELECT email FROM ${table} WHERE id = ?`).bind(id).first();
  if (row && row.email !== email) throw new HttpError(404, "not found");
}

async function putPlaylist({ db, email, request, params }) {
  const id = requireId(params.id, "playlist id");
  const body = requireObject((await readJson(request, 131072)).value);
  if (typeof body.name !== "string" || !body.name.trim() || body.name.length > MAX_NAME) {
    throw new HttpError(400, `name must be 1-${MAX_NAME} characters`);
  }
  const trackIds = idList(body.track_ids, MAX_PLAYLIST_TRACKS, "track_ids");
  await assertOwner(db, "playlists", id, email);
  const updated = nowIso();
  await db
    .prepare(
      "INSERT INTO playlists (id, email, name, track_ids, updated_at) VALUES (?, ?, ?, ?, ?) " +
        "ON CONFLICT (id) DO UPDATE SET name = excluded.name, track_ids = excluded.track_ids, updated_at = excluded.updated_at",
    )
    .bind(id, email, body.name.trim(), JSON.stringify(trackIds), updated)
    .run();
  return json({ ok: true, updated_at: updated });
}

async function deletePlaylist({ db, email, params }) {
  const id = requireId(params.id, "playlist id");
  await db.prepare("DELETE FROM playlists WHERE id = ? AND email = ?").bind(id, email).run();
  return json({ ok: true });
}

async function getSessions({ db, email, url }) {
  let since = url.searchParams.get("since");
  if (since === null) since = new Date(Date.now() - 365 * 86400000).toISOString();
  else if (!isIso(since)) throw new HttpError(400, "since must be an ISO date-time");
  const { results } = await db
    .prepare(
      "SELECT id, mode, started_at, minutes, arm, rating, note, track_ids FROM sessions " +
        "WHERE email = ? AND started_at >= ? ORDER BY started_at",
    )
    .bind(email, since)
    .all();
  return json({ sessions: results.map((r) => ({ ...r, track_ids: parseList(r.track_ids) })) });
}

async function putSession({ db, email, request, params }) {
  const id = requireId(params.id, "session id");
  const s = requireObject((await readJson(request, 131072)).value);
  if (!MODES.has(s.mode)) throw new HttpError(400, "mode must be focus, create or meditate");
  if (!isIso(s.started_at)) throw new HttpError(400, "started_at must be an ISO date-time");
  const minutes = s.minutes ?? 0;
  if (typeof minutes !== "number" || !(minutes >= 0 && minutes <= 1440)) throw new HttpError(400, "minutes must be 0-1440");
  const arm = s.arm ?? null;
  if (arm !== null && arm !== "on" && arm !== "off") throw new HttpError(400, "arm must be on, off or null");
  const rating = s.rating ?? null;
  if (rating !== null && !(Number.isInteger(rating) && rating >= 1 && rating <= 5)) throw new HttpError(400, "rating must be 1-5 or null");
  const note = s.note ?? null;
  if (note !== null && (typeof note !== "string" || note.length > MAX_NOTE)) throw new HttpError(400, `note must be at most ${MAX_NOTE} characters`);
  const trackIds = idList(s.track_ids ?? [], MAX_PLAYLIST_TRACKS, "track_ids");
  await assertOwner(db, "sessions", id, email);
  await db
    .prepare(
      "INSERT INTO sessions (id, email, mode, started_at, minutes, arm, rating, note, track_ids) " +
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET mode = excluded.mode, " +
        "started_at = excluded.started_at, minutes = excluded.minutes, arm = excluded.arm, " +
        "rating = excluded.rating, note = excluded.note, track_ids = excluded.track_ids",
    )
    .bind(id, email, s.mode, s.started_at, minutes, arm, rating, note, JSON.stringify(trackIds))
    .run();
  return json({ ok: true });
}

// ---------------------------------------------------------------------------

const ROUTES = [
  ["GET", /^\/api\/me$/, getMe],
  ["PUT", /^\/api\/profile$/, putProfile],
  ["GET", /^\/api\/library$/, getLibrary],
  ["PUT", /^\/api\/tracks\/(?<id>[^/]+)\/tags$/, putTags],
  ["GET", /^\/api\/ratings$/, getRatings],
  ["PUT", /^\/api\/ratings\/(?<id>[^/]+)$/, putRating],
  ["GET", /^\/api\/playlists$/, getPlaylists],
  ["PUT", /^\/api\/playlists\/(?<id>[^/]+)$/, putPlaylist],
  ["DELETE", /^\/api\/playlists\/(?<id>[^/]+)$/, deletePlaylist],
  ["GET", /^\/api\/sessions$/, getSessions],
  ["PUT", /^\/api\/sessions\/(?<id>[^/]+)$/, putSession],
];

export async function routeApi(request, env, url, email) {
  let pathMatched = false;
  for (const [method, pattern, handler] of ROUTES) {
    const m = pattern.exec(url.pathname);
    if (!m) continue;
    pathMatched = true;
    if (method !== request.method) continue;
    let params = {};
    try {
      params = Object.fromEntries(Object.entries(m.groups ?? {}).map(([k, v]) => [k, decodeURIComponent(v)]));
    } catch {
      throw new HttpError(400, "bad path");
    }
    return handler({ db: env.DB, email, request, params, url });
  }
  if (pathMatched) throw new HttpError(405, "method not allowed");
  throw new HttpError(404, "no such endpoint");
}
