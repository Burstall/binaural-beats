import { test } from "node:test";
import assert from "node:assert/strict";
import { handle } from "../worker/index.js";
import { addTrack, makeEnv, req } from "./helpers.js";

async function call(env, path, opts) {
  const res = await handle(req(path, opts), env);
  const text = await res.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body, headers: res.headers };
}

const A = "aaaaaaaaaaaaaaaa";
const B = "bbbbbbbbbbbbbbbb";

test("refuses everything when Access is not configured and there is no dev email", async () => {
  const env = makeEnv({ DEV_EMAIL: "" });
  for (const path of ["/api/me", "/api/library", "/media/tracks/x.mp3"]) {
    const r = await call(env, path);
    assert.equal(r.status, 503, path);
    assert.deepEqual(r.body, { error: "access not configured" });
  }
});

test("the dev email only works from localhost", async () => {
  const env = makeEnv();
  const local = await call(env, "/api/me");
  assert.equal(local.status, 200);
  assert.equal(local.body.email, "me@example.com");
  const remote = await call(env, "/api/me", { host: "https://violet.example.workers.dev" });
  assert.equal(remote.status, 503);
});

test("static paths go to the assets binding", async () => {
  const r = await call(makeEnv(), "/index.html");
  assert.equal(r.status, 200);
  assert.match(r.body, /app/);
});

test("profile round-trips, is per person, and has a size limit", async () => {
  const env = makeEnv();
  assert.equal((await call(env, "/api/me")).body.profile, null);
  const put = await call(env, "/api/profile", { method: "PUT", body: { v: 1, name: "Rich", future: { x: 1 } } });
  assert.equal(put.status, 200);
  assert.ok(put.body.updated_at);
  const me = await call(env, "/api/me");
  assert.deepEqual(me.body.profile, { v: 1, name: "Rich", future: { x: 1 } });
  assert.equal(me.headers.get("cache-control"), "no-store");

  const other = makeEnv({ DB: env.DB, DEV_EMAIL: "friend@example.com" });
  assert.equal((await call(other, "/api/me")).body.profile, null);

  assert.equal((await call(env, "/api/profile", { method: "PUT", body: [1, 2] })).status, 400);
  assert.equal((await call(env, "/api/profile", { method: "PUT", body: "not json" })).status, 400);
  const huge = { pad: "x".repeat(70000) };
  assert.equal((await call(env, "/api/profile", { method: "PUT", body: huge })).status, 413);
});

test("library lists tracks with tags and urls, without r2 keys", async () => {
  const env = makeEnv();
  addTrack(env, A, { artist: "Zed" });
  addTrack(env, B, { artist: "amy", added_at: "2026-10-15T00:00:00.000Z" });
  env.DB.sqlite.prepare("INSERT INTO track_tags VALUES (?, ?, 'auto')").run(A, "focus");
  env.DB.sqlite.prepare("INSERT INTO track_tags VALUES (?, ?, 'auto')").run(A, "beat");
  const r = await call(env, "/api/library");
  assert.equal(r.status, 200);
  assert.deepEqual(r.body.tracks.map((t) => t.id), [B, A], "sorted by artist, case-insensitively");
  const a = r.body.tracks[1];
  assert.deepEqual(a.tags, ["beat", "focus"]);
  assert.equal(a.url, `/media/tracks/${A}.mp3`);
  assert.equal("r2_key" in a, false);
  assert.deepEqual(r.body.tracks[0].tags, []);
  assert.equal(r.body.updated_at, "2026-10-15T00:00:00.000Z");
});

test("tags: add, remove, validate, cap, unknown track", async () => {
  const env = makeEnv();
  addTrack(env, A);
  env.DB.sqlite.prepare("INSERT INTO track_tags VALUES (?, 'focus', 'auto')").run(A);
  let r = await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { add: ["vocals", "focus"], remove: [] } });
  assert.deepEqual(r.body.tags, ["focus", "vocals"]);
  const source = env.DB.sqlite.prepare("SELECT source FROM track_tags WHERE tag = 'focus'").get().source;
  assert.equal(source, "auto", "adding an existing tag keeps its source");
  r = await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { remove: ["focus"] } });
  assert.deepEqual(r.body.tags, ["vocals"]);
  assert.equal((await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { add: ["Bad Tag"] } })).status, 400);
  assert.equal((await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { add: "focus" } })).status, 400);
  assert.equal((await call(env, `/api/tracks/${B}/tags`, { method: "PUT", body: { add: ["x"] } })).status, 404);
  const many = Array.from({ length: 32 }, (_, i) => `t${i}`);
  assert.equal((await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { add: many } })).status, 400, "33 tags");
  assert.deepEqual((await call(env, `/api/tracks/${A}/tags`, { method: "PUT", body: { add: many.slice(0, 31) } })).body.tags.length, 32);
});

test("ratings are per person; 0 deletes; unknown track 404", async () => {
  const env = makeEnv();
  addTrack(env, A);
  assert.equal((await call(env, `/api/ratings/${A}`, { method: "PUT", body: { value: 1 } })).status, 200);
  assert.deepEqual((await call(env, "/api/ratings")).body.ratings, { [A]: 1 });
  await call(env, `/api/ratings/${A}`, { method: "PUT", body: { value: -1 } });
  assert.deepEqual((await call(env, "/api/ratings")).body.ratings, { [A]: -1 });
  const other = makeEnv({ DB: env.DB, DEV_EMAIL: "friend@example.com" });
  assert.deepEqual((await call(other, "/api/ratings")).body.ratings, {});
  await call(env, `/api/ratings/${A}`, { method: "PUT", body: { value: 0 } });
  assert.deepEqual((await call(env, "/api/ratings")).body.ratings, {});
  assert.equal((await call(env, `/api/ratings/${A}`, { method: "PUT", body: { value: 2 } })).status, 400);
  assert.equal((await call(env, `/api/ratings/${B}`, { method: "PUT", body: { value: 1 } })).status, 404);
});

test("playlists: upsert, list, ownership, delete, validation", async () => {
  const env = makeEnv();
  const id = "plist-0001";
  let r = await call(env, `/api/playlists/${id}`, { method: "PUT", body: { name: " Morning ", track_ids: [A, B] } });
  assert.equal(r.status, 200);
  r = await call(env, "/api/playlists");
  assert.deepEqual(r.body.playlists.map((p) => [p.id, p.name, p.track_ids]), [[id, "Morning", [A, B]]]);

  const other = makeEnv({ DB: env.DB, DEV_EMAIL: "friend@example.com" });
  assert.deepEqual((await call(other, "/api/playlists")).body.playlists, []);
  assert.equal((await call(other, `/api/playlists/${id}`, { method: "PUT", body: { name: "mine", track_ids: [] } })).status, 404);
  await call(other, `/api/playlists/${id}`, { method: "DELETE" });
  assert.equal((await call(env, "/api/playlists")).body.playlists.length, 1, "someone else's delete does nothing");

  assert.equal((await call(env, "/api/playlists/short", { method: "PUT", body: { name: "x", track_ids: [] } })).status, 400);
  assert.equal((await call(env, `/api/playlists/${id}`, { method: "PUT", body: { name: "", track_ids: [] } })).status, 400);
  assert.equal((await call(env, `/api/playlists/${id}`, { method: "PUT", body: { name: "x".repeat(81), track_ids: [] } })).status, 400);
  assert.equal((await call(env, `/api/playlists/${id}`, { method: "PUT", body: { name: "x", track_ids: ["no"] } })).status, 400);

  assert.equal((await call(env, `/api/playlists/${id}`, { method: "DELETE" })).status, 200);
  assert.deepEqual((await call(env, "/api/playlists")).body.playlists, []);
});

test("sessions: upsert, since filter, validation, ownership", async () => {
  const env = makeEnv();
  const s = { mode: "focus", started_at: "2026-10-14T09:00:00.000Z", minutes: 25.5, arm: "on", rating: null, note: null, track_ids: [A] };
  assert.equal((await call(env, "/api/sessions/sess-0001", { method: "PUT", body: s })).status, 200);
  await call(env, "/api/sessions/sess-0001", { method: "PUT", body: { ...s, rating: 4, note: "good" } });
  let r = await call(env, "/api/sessions?since=2026-10-01T00:00:00Z");
  assert.equal(r.body.sessions.length, 1);
  assert.equal(r.body.sessions[0].rating, 4);
  assert.deepEqual(r.body.sessions[0].track_ids, [A]);
  r = await call(env, "/api/sessions?since=2026-10-15T00:00:00Z");
  assert.equal(r.body.sessions.length, 0);
  assert.equal((await call(env, "/api/sessions?since=yesterday")).status, 400);

  const bad = [
    { ...s, mode: "sleep" },
    { ...s, started_at: "now" },
    { ...s, minutes: -1 },
    { ...s, arm: "maybe" },
    { ...s, rating: 6 },
    { ...s, rating: 2.5 },
    { ...s, note: "x".repeat(2001) },
  ];
  for (const body of bad) assert.equal((await call(env, "/api/sessions/sess-0002", { method: "PUT", body })).status, 400, JSON.stringify(body).slice(0, 60));

  const other = makeEnv({ DB: env.DB, DEV_EMAIL: "friend@example.com" });
  assert.equal((await call(other, "/api/sessions/sess-0001", { method: "PUT", body: s })).status, 404);
  assert.deepEqual((await call(other, "/api/sessions?since=2020-01-01T00:00:00Z")).body.sessions, []);
});

test("unknown endpoints 404, wrong methods 405", async () => {
  const env = makeEnv();
  assert.equal((await call(env, "/api/nope")).status, 404);
  assert.equal((await call(env, "/api/me", { method: "POST", body: {} })).status, 405);
  assert.equal((await call(env, "/api/playlists/abcdefgh", { method: "GET" })).status, 405);
});
