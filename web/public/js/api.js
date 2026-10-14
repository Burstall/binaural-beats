// Talking to the worker. Reads fail loudly; writes made while offline go to
// the outbox and are sent, in order, when the network comes back.

import { enqueue, list, remove } from "./idb.js";

export class SignedOut extends Error {
  constructor() {
    super("signed out");
  }
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export class Offline extends Error {
  constructor() {
    super("offline");
  }
}

async function request(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      credentials: "same-origin",
      redirect: "manual",
      cache: "no-store",
      headers: body !== undefined ? { "content-type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Offline();
  }
  // An expired Access session answers with a redirect to its login page, or
  // with that page itself. Either way: not JSON, so not signed in.
  if (res.type === "opaqueredirect" || res.status === 401) throw new SignedOut();
  const type = res.headers.get("content-type") || "";
  if (!type.includes("application/json")) {
    if (res.status === 503) throw new Offline();
    throw new SignedOut();
  }
  const data = await res.json();
  if (!res.ok) {
    if (data.error === "offline") throw new Offline();
    throw new ApiError(res.status, data.error || `HTTP ${res.status}`);
  }
  return data;
}

/** A write that survives being offline: queued, then retried by flush(). */
async function write(method, path, body) {
  try {
    return await request(method, path, body);
  } catch (err) {
    if (err instanceof Offline) {
      await enqueue({ method, path, body });
      return { queued: true };
    }
    throw err;
  }
}

let flushing = false;

/** Sends queued writes in order. Stops at the first network failure. */
export async function flush() {
  if (flushing) return 0;
  flushing = true;
  let sent = 0;
  try {
    for (const entry of await list()) {
      try {
        await request(entry.method, entry.path, entry.body);
      } catch (err) {
        if (err instanceof Offline || err instanceof SignedOut) break;
        console.warn("dropping a queued write the server refused", entry, err);
      }
      await remove(entry.seq);
      sent += 1;
    }
  } finally {
    flushing = false;
  }
  return sent;
}

export async function pending() {
  return (await list()).length;
}

const enc = encodeURIComponent;

export const api = {
  me: () => request("GET", "/api/me"),
  putProfile: (profile) => write("PUT", "/api/profile", profile),
  library: () => request("GET", "/api/library"),
  putTags: (id, add, remove) => write("PUT", `/api/tracks/${enc(id)}/tags`, { add, remove }),
  ratings: () => request("GET", "/api/ratings"),
  putRating: (id, value) => write("PUT", `/api/ratings/${enc(id)}`, { value }),
  playlists: () => request("GET", "/api/playlists"),
  putPlaylist: (id, name, trackIds) => write("PUT", `/api/playlists/${enc(id)}`, { name, track_ids: trackIds }),
  deletePlaylist: (id) => write("DELETE", `/api/playlists/${enc(id)}`),
  sessions: (since) => request("GET", `/api/sessions${since ? `?since=${enc(since)}` : ""}`),
  putSession: (session) => {
    const { id, ...rest } = session;
    return write("PUT", `/api/sessions/${enc(id)}`, rest);
  },
};
