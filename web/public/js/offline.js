// "Keep offline": a playlist's tracks stored whole in Cache Storage, where the
// service worker finds them and answers range requests from them.

const MEDIA_CACHE = "violet-media";
const KEPT_KEY = "violet-offline-lists";

function absolute(url) {
  return new URL(url, location.origin).href;
}

export function keptLists() {
  try {
    return new Set(JSON.parse(localStorage.getItem(KEPT_KEY) || "[]"));
  } catch {
    return new Set();
  }
}

function setKept(set) {
  try {
    localStorage.setItem(KEPT_KEY, JSON.stringify([...set]));
  } catch {
    /* fine */
  }
}

/** How many of these tracks are stored offline. */
export async function countCached(tracks) {
  if (!("caches" in window)) return 0;
  const cache = await caches.open(MEDIA_CACHE);
  let n = 0;
  for (const t of tracks) if (await cache.match(absolute(t.url))) n += 1;
  return n;
}

/**
 * Downloads every track not yet stored. `onProgress(done, total)` after each.
 * Stops at the first failure and reports it; what was stored stays stored.
 */
export async function keepOffline(listId, tracks, onProgress = () => {}) {
  const cache = await caches.open(MEDIA_CACHE);
  if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
  const kept = keptLists();
  kept.add(listId);
  setKept(kept);
  let done = 0;
  for (const t of tracks) {
    const url = absolute(t.url);
    if (!(await cache.match(url))) {
      const res = await fetch(url, { credentials: "same-origin" });
      if (res.status !== 200) throw new Error(`${t.title}: HTTP ${res.status}`);
      await cache.put(url, res);
    }
    done += 1;
    onProgress(done, tracks.length);
  }
  return done;
}

/** Forgets a list; deletes its tracks unless another kept list still has them. */
export async function dropOffline(listId, tracks, allLists) {
  const kept = keptLists();
  kept.delete(listId);
  setKept(kept);
  const stillNeeded = new Set();
  for (const list of allLists) if (kept.has(list.id)) for (const id of list.track_ids) stillNeeded.add(id);
  const cache = await caches.open(MEDIA_CACHE);
  for (const t of tracks) if (!stillNeeded.has(t.id)) await cache.delete(absolute(t.url));
}

export async function storageText() {
  if (!navigator.storage?.estimate) return "";
  const { usage = 0, quota = 0 } = await navigator.storage.estimate();
  const mb = (b) => `${Math.round(b / 1e6)} MB`;
  return quota ? `${mb(usage)} used of ${mb(quota)}` : `${mb(usage)} used`;
}
