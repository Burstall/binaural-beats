// Audio out of R2, with byte ranges. iOS Safari asks for `bytes=0-1` first
// and refuses to play anything served whole in answer; that was measured on
// the lock-test page, not assumed.

import { parseRange } from "../public/js/range.js";
import { HttpError } from "./http.js";

const KEY_RULE = /^(tracks|beds)\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

export function mediaKey(pathname) {
  let key;
  try {
    key = decodeURIComponent(pathname.slice("/media/".length));
  } catch {
    return null;
  }
  if (!KEY_RULE.test(key) || key.includes("..")) return null;
  return key;
}

export async function serveMedia(request, env, url) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    throw new HttpError(405, "method not allowed");
  }
  const key = mediaKey(url.pathname);
  if (!key) throw new HttpError(404, "no such media");
  const head = await env.MUSIC.head(key);
  if (!head) throw new HttpError(404, "no such media");

  const size = head.size;
  const headers = new Headers();
  head.writeHttpMetadata(headers);
  if (!headers.has("content-type")) headers.set("content-type", "application/octet-stream");
  headers.set("etag", head.httpEtag);
  headers.set("accept-ranges", "bytes");
  headers.set("cache-control", "private, max-age=31536000, immutable");

  const inm = request.headers.get("if-none-match");
  if (inm && inm === head.httpEtag && !request.headers.get("range")) {
    return new Response(null, { status: 304, headers });
  }

  const range = parseRange(request.headers.get("range"), size);
  if (range === "unsatisfiable") {
    headers.set("content-range", `bytes */${size}`);
    return new Response(null, { status: 416, headers });
  }

  let status = 200;
  let length = size;
  if (range) {
    status = 206;
    length = range.end - range.start + 1;
    headers.set("content-range", `bytes ${range.start}-${range.end}/${size}`);
  }
  headers.set("content-length", String(length));
  if (request.method === "HEAD") return new Response(null, { status, headers });

  const object = await env.MUSIC.get(key, range ? { range: { offset: range.start, length } } : undefined);
  if (!object) throw new HttpError(404, "no such media");
  return new Response(object.body, { status, headers });
}
