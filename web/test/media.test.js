import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { handle } from "../worker/index.js";
import { mediaKey } from "../worker/media.js";
import { parseRange } from "../public/js/range.js";
import { WEB, makeEnv, req } from "./helpers.js";

const BYTES = Uint8Array.from({ length: 1000 }, (_, i) => i % 251);

function env() {
  const e = makeEnv();
  e.MUSIC.put("tracks/abc.mp3", BYTES, "audio/mpeg");
  return e;
}

async function get(e, path, headers = {}, method = "GET") {
  const res = await handle(req(path, { headers, method }), e);
  const body = new Uint8Array(await res.arrayBuffer());
  return { res, body };
}

test("parseRange covers the forms Safari and the RFC use", () => {
  assert.equal(parseRange(null, 1000), null);
  assert.deepEqual(parseRange("bytes=0-1", 1000), { start: 0, end: 1 });
  assert.deepEqual(parseRange("bytes=100-", 1000), { start: 100, end: 999 });
  assert.deepEqual(parseRange("bytes=-100", 1000), { start: 900, end: 999 });
  assert.deepEqual(parseRange("bytes=-5000", 1000), { start: 0, end: 999 });
  assert.deepEqual(parseRange("bytes=990-5000", 1000), { start: 990, end: 999 });
  assert.equal(parseRange("bytes=1000-", 1000), "unsatisfiable");
  assert.equal(parseRange("bytes=-0", 1000), "unsatisfiable");
  assert.equal(parseRange("bytes=0-", 0), "unsatisfiable");
  assert.equal(parseRange("bytes=5-1", 1000), null, "last < first is ignored");
  assert.equal(parseRange("bytes=0-1,5-6", 1000), null, "multi-range served whole");
  assert.equal(parseRange("items=0-1", 1000), null);
  assert.equal(parseRange("bytes=-", 1000), null);
});

test("the service worker's copy of parseRange behaves identically", () => {
  const src = readFileSync(join(WEB, "public", "sw.js"), "utf8");
  const m = /\/\/ BEGIN parseRange\n([\s\S]*?)\/\/ END parseRange/.exec(src);
  assert.ok(m, "sw.js marks its parseRange copy");
  const swParse = new Function(`${m[1]}; return parseRange;`)();
  const cases = ["bytes=0-1", "bytes=100-", "bytes=-100", "bytes=-5000", "bytes=990-5000", "bytes=1000-", "bytes=-0", "bytes=5-1", "bytes=0-1,5-6", null, "bytes=-"];
  for (const c of cases) for (const size of [0, 1, 1000]) assert.deepEqual(swParse(c, size), parseRange(c, size), `${c} @ ${size}`);
});

test("whole file: 200, headers, body", async () => {
  const { res, body } = await get(env(), "/media/tracks/abc.mp3");
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("content-type"), "audio/mpeg");
  assert.equal(res.headers.get("accept-ranges"), "bytes");
  assert.equal(res.headers.get("content-length"), "1000");
  assert.match(res.headers.get("cache-control"), /immutable/);
  assert.deepEqual(body, BYTES);
});

test("ranges: 206 with the right slice, and R2 is asked for only that slice", async () => {
  const e = env();
  for (const [header, start, end] of [["bytes=0-1", 0, 1], ["bytes=10-19", 10, 19], ["bytes=990-", 990, 999], ["bytes=-3", 997, 999]]) {
    const { res, body } = await get(e, "/media/tracks/abc.mp3", { range: header });
    assert.equal(res.status, 206, header);
    assert.equal(res.headers.get("content-range"), `bytes ${start}-${end}/1000`);
    assert.equal(res.headers.get("content-length"), String(end - start + 1));
    assert.deepEqual(body, BYTES.subarray(start, end + 1));
    assert.deepEqual(e.MUSIC.gets.at(-1).options, { range: { offset: start, length: end - start + 1 } });
  }
});

test("unsatisfiable range: 416 with the size", async () => {
  const { res } = await get(env(), "/media/tracks/abc.mp3", { range: "bytes=5000-" });
  assert.equal(res.status, 416);
  assert.equal(res.headers.get("content-range"), "bytes */1000");
});

test("HEAD answers headers without reading the object", async () => {
  const e = env();
  const { res, body } = await get(e, "/media/tracks/abc.mp3", { range: "bytes=0-9" }, "HEAD");
  assert.equal(res.status, 206);
  assert.equal(res.headers.get("content-length"), "10");
  assert.equal(body.length, 0);
  assert.equal(e.MUSIC.gets.length, 0);
});

test("If-None-Match gives 304", async () => {
  const e = env();
  const first = await get(e, "/media/tracks/abc.mp3");
  const { res } = await get(e, "/media/tracks/abc.mp3", { "if-none-match": first.res.headers.get("etag") });
  assert.equal(res.status, 304);
});

test("only tracks/ and beds/ keys, no traversal, 404 when missing, 405 for writes", async () => {
  assert.equal(mediaKey("/media/tracks/abc.mp3"), "tracks/abc.mp3");
  assert.equal(mediaKey("/media/beds/meditate.mp3"), "beds/meditate.mp3");
  assert.equal(mediaKey("/media/secret/x.mp3"), null);
  assert.equal(mediaKey("/media/tracks/../x"), null);
  assert.equal(mediaKey("/media/tracks%2F..%2Fx"), null);
  assert.equal(mediaKey("/media/tracks/a/b.mp3"), null);
  assert.equal(mediaKey("/media/%E0%A4%A"), null);
  const e = env();
  assert.equal((await get(e, "/media/tracks/missing.mp3")).res.status, 404);
  assert.equal((await get(e, "/media/other/abc.mp3")).res.status, 404);
  assert.equal((await handle(req("/media/tracks/abc.mp3", { method: "PUT", body: "x" }), e)).status, 405);
});
