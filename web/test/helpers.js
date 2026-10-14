// Fakes for the worker's bindings. D1 is played by node:sqlite running the
// real migrations, so the SQL under test is the SQL that ships.

import { readFileSync, readdirSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
export const WEB = join(here, "..");

class Statement {
  constructor(db, sql, args = []) {
    this.db = db;
    this.sql = sql;
    this.args = args;
  }

  bind(...args) {
    return new Statement(this.db, this.sql, args);
  }

  async all() {
    return { results: this.db.prepare(this.sql).all(...this.args).map((r) => ({ ...r })), success: true };
  }

  async first() {
    const row = this.db.prepare(this.sql).get(...this.args);
    return row ? { ...row } : null;
  }

  async run() {
    const info = this.db.prepare(this.sql).run(...this.args);
    return { success: true, meta: { changes: Number(info.changes) } };
  }

  runSync() {
    const stmt = this.db.prepare(this.sql);
    if (/^\s*select/i.test(this.sql)) return { results: stmt.all(...this.args).map((r) => ({ ...r })) };
    stmt.run(...this.args);
    return { results: [] };
  }
}

export class FakeD1 {
  constructor() {
    this.sqlite = new DatabaseSync(":memory:");
    this.sqlite.exec("PRAGMA foreign_keys = ON");
    const dir = join(WEB, "migrations");
    for (const file of readdirSync(dir).filter((f) => f.endsWith(".sql")).sort()) {
      this.sqlite.exec(readFileSync(join(dir, file), "utf8"));
    }
  }

  prepare(sql) {
    return new Statement(this.sqlite, sql);
  }

  async batch(statements) {
    this.sqlite.exec("BEGIN");
    try {
      const out = statements.map((s) => s.runSync());
      this.sqlite.exec("COMMIT");
      return out;
    } catch (err) {
      this.sqlite.exec("ROLLBACK");
      throw err;
    }
  }
}

export class FakeR2 {
  constructor() {
    this.objects = new Map();
    this.gets = [];
  }

  put(key, bytes, contentType) {
    this.objects.set(key, { bytes: new Uint8Array(bytes), contentType, etag: `"etag-${key.length}-${bytes.length}"` });
  }

  meta(key) {
    const o = this.objects.get(key);
    if (!o) return null;
    return {
      key,
      size: o.bytes.length,
      httpEtag: o.etag,
      writeHttpMetadata(headers) {
        if (o.contentType) headers.set("content-type", o.contentType);
      },
    };
  }

  async head(key) {
    return this.meta(key);
  }

  async get(key, options = {}) {
    const meta = this.meta(key);
    if (!meta) return null;
    this.gets.push({ key, options });
    const { bytes } = this.objects.get(key);
    let slice = bytes;
    if (options.range) {
      const { offset = 0, length = bytes.length - offset } = options.range;
      slice = bytes.subarray(offset, offset + length);
    }
    return { ...meta, body: new Response(slice).body };
  }
}

export function makeEnv(overrides = {}) {
  return {
    DB: new FakeD1(),
    MUSIC: new FakeR2(),
    ASSETS: { fetch: async () => new Response("<!doctype html>app", { headers: { "content-type": "text/html" } }) },
    DEV_EMAIL: "me@example.com",
    ACCESS_TEAM_DOMAIN: "",
    ACCESS_AUD: "",
    ...overrides,
  };
}

export function req(path, { method = "GET", body, headers = {}, host = "http://localhost:8787" } = {}) {
  const init = { method, headers: { ...headers } };
  if (body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
    init.headers["content-type"] = "application/json";
  }
  return new Request(`${host}${path}`, init);
}

export function addTrack(env, id, extra = {}) {
  const row = {
    id,
    r2_key: `tracks/${id}.mp3`,
    title: `Title ${id}`,
    artist: "Artist",
    album: null,
    duration: 180,
    bytes: 1000,
    mime: "audio/mpeg",
    loudness: -14,
    peak: 0.9,
    gain_db: -2,
    bpm: 84,
    beat_offset: 0.1,
    beat_strength: 0.6,
    energy: 0.4,
    brightness: 1500,
    license: "CC BY 4.0",
    license_url: "https://creativecommons.org/licenses/by/4.0/",
    source_url: "https://archive.org/details/x",
    attribution: `"Title ${id}" by Artist, CC BY 4.0`,
    added_at: "2026-10-14T00:00:00.000Z",
    ...extra,
  };
  const cols = Object.keys(row);
  env.DB.sqlite.prepare(`INSERT INTO tracks (${cols.join(",")}) VALUES (${cols.map(() => "?").join(",")})`).run(...Object.values(row));
  return row;
}
