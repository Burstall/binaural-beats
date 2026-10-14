// The app's state and the only code that changes it. Views read `state` and
// subscribe to the events named below; they never call the API themselves.
//
// Events: "profile", "library", "ratings", "playlists", "sessions",
// "network", "player".

import { ApiError, Offline, SignedOut, api, flush, pending } from "./api.js";
import { mergeProfile } from "./profile.js";
import { randomId } from "./rng.js";
import { debounce } from "./ui.js";

const CACHE_KEY = "violet-cache-v1";

export const state = {
  email: null,
  profile: mergeProfile(null),
  tracks: [],
  byId: new Map(),
  ratings: {},
  playlists: [],
  sessions: [],
  online: navigator.onLine,
  signedOut: false,
  notConfigured: false,
  pending: 0,
  booted: false,
};

const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
  return () => listeners.get(event).delete(fn);
}

export function emit(event, detail) {
  for (const fn of listeners.get(event) ?? []) {
    try {
      fn(detail);
    } catch (err) {
      console.error(err);
    }
  }
}

// A local copy of the person's own data, so the app opens offline. Losing it
// costs nothing: the server has the truth.
function saveCache() {
  try {
    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({
        email: state.email,
        profile: state.profile,
        ratings: state.ratings,
        playlists: state.playlists,
        sessions: state.sessions.slice(-400),
      }),
    );
  } catch {
    /* private mode or full: fine */
  }
}

function loadCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function setLibrary(tracks) {
  state.tracks = tracks;
  state.byId = new Map(tracks.map((t) => [t.id, t]));
  emit("library");
}

async function refreshPending() {
  try {
    state.pending = await pending();
  } catch {
    state.pending = 0;
  }
  emit("network");
}

function handleAuth(err) {
  if (err instanceof SignedOut) {
    state.signedOut = true;
    emit("network");
    return true;
  }
  if (err instanceof ApiError && err.status === 503) {
    state.notConfigured = true;
    emit("network");
    return true;
  }
  return false;
}

/** Loads everything. Falls back to the local copy when offline. */
export async function boot() {
  const cached = loadCache();
  if (cached) {
    state.email = cached.email;
    state.profile = mergeProfile(cached.profile);
    state.ratings = cached.ratings ?? {};
    state.playlists = cached.playlists ?? [];
    state.sessions = cached.sessions ?? [];
  }
  try {
    const me = await api.me();
    state.email = me.email;
    state.profile = mergeProfile(me.profile);
    state.signedOut = false;
    await flush().catch(() => 0);
    const [lib, ratings, playlists, sessions] = await Promise.all([
      api.library(),
      api.ratings(),
      api.playlists(),
      api.sessions(),
    ]);
    setLibrary(lib.tracks);
    state.ratings = ratings.ratings;
    state.playlists = playlists.playlists;
    state.sessions = sessions.sessions;
    state.online = true;
    saveCache();
  } catch (err) {
    if (handleAuth(err)) {
      state.booted = true;
      return;
    }
    if (!(err instanceof Offline)) console.error(err);
    state.online = false;
    try {
      const lib = await api.library();
      setLibrary(lib.tracks);
    } catch {
      /* no library cached yet */
    }
  }
  state.booted = true;
  await refreshPending();
  for (const e of ["profile", "library", "ratings", "playlists", "sessions", "network"]) emit(e);
}

export async function goOnline() {
  state.online = true;
  try {
    await flush();
    const lib = await api.library();
    setLibrary(lib.tracks);
  } catch (err) {
    handleAuth(err);
  }
  await refreshPending();
}

export function goOffline() {
  state.online = false;
  emit("network");
}

async function write(promise) {
  try {
    const result = await promise;
    if (result?.queued) await refreshPending();
    return result;
  } catch (err) {
    if (!handleAuth(err)) throw err;
    return null;
  }
}

// --- profile --------------------------------------------------------------

const pushProfile = debounce(() => write(api.putProfile(state.profile)).catch((e) => console.error(e)), 800);

/** Applies `fn` to the profile, then saves it shortly afterwards. */
export function updateProfile(fn) {
  fn(state.profile);
  saveCache();
  emit("profile");
  pushProfile();
}

// --- tracks ---------------------------------------------------------------

export async function setTags(trackId, add = [], remove = []) {
  const track = state.byId.get(trackId);
  if (track) {
    const tags = new Set(track.tags);
    for (const t of remove) tags.delete(t);
    for (const t of add) tags.add(t);
    track.tags = [...tags].sort();
    emit("library");
  }
  const result = await write(api.putTags(trackId, add, remove));
  if (track && result?.tags) {
    track.tags = result.tags;
    emit("library");
  }
}

export async function rate(trackId, value) {
  if (value === 0) delete state.ratings[trackId];
  else state.ratings[trackId] = value;
  saveCache();
  emit("ratings");
  await write(api.putRating(trackId, value));
}

// --- playlists -------------------------------------------------------------

export function playlist(id) {
  return state.playlists.find((p) => p.id === id) ?? null;
}

export async function savePlaylist(id, name, trackIds) {
  const existing = playlist(id);
  const updated = new Date().toISOString();
  if (existing) Object.assign(existing, { name, track_ids: trackIds, updated_at: updated });
  else state.playlists.push({ id, name, track_ids: trackIds, updated_at: updated });
  state.playlists.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
  saveCache();
  emit("playlists");
  await write(api.putPlaylist(id, name, trackIds));
}

export async function createPlaylist(name) {
  const id = randomId();
  await savePlaylist(id, name, []);
  return id;
}

export async function deletePlaylist(id) {
  state.playlists = state.playlists.filter((p) => p.id !== id);
  updateProfile((p) => {
    for (const mode of Object.values(p.modes)) if (mode.playlist === id) mode.playlist = null;
  });
  saveCache();
  emit("playlists");
  await write(api.deletePlaylist(id));
}

export async function addToPlaylist(id, trackId) {
  const p = playlist(id);
  if (!p || p.track_ids.includes(trackId)) return false;
  await savePlaylist(id, p.name, [...p.track_ids, trackId]);
  return true;
}

// --- sessions --------------------------------------------------------------

export async function saveSession(session) {
  const i = state.sessions.findIndex((s) => s.id === session.id);
  const copy = { ...session, track_ids: [...session.track_ids] };
  if (i >= 0) state.sessions[i] = copy;
  else state.sessions.push(copy);
  saveCache();
  emit("sessions");
  await write(api.putSession(copy));
}
