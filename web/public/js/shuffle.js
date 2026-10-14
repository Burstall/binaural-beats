// Choosing what plays next.
//
// Shuffle draws from tracks that carry every one of the mode's tags, none of
// its excluded tags, and no thumbs-down from this person. A thumbs-up triples
// a track's weight. Nothing repeats until everything eligible has played, and
// a new round never opens with the track that closed the last one.

export const LIKED_WEIGHT = 3;

export function eligible(tracks, { tags = [], exclude = [] } = {}, ratings = {}) {
  return tracks.filter((t) => {
    if (ratings[t.id] === -1) return false;
    const own = new Set(t.tags ?? []);
    return tags.every((tag) => own.has(tag)) && !exclude.some((tag) => own.has(tag));
  });
}

export function weightOf(track, ratings = {}) {
  return ratings[track.id] === 1 ? LIKED_WEIGHT : 1;
}

/** One weighted draw from `items`; `random` returns floats in [0, 1). */
export function weightedPick(items, weights, random) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return undefined;
  let r = random() * total;
  for (let i = 0; i < items.length; i += 1) {
    r -= weights[i];
    if (r < 0) return items[i];
  }
  return items[items.length - 1];
}

/** A shuffled draw from a pool, without repeats until the pool is spent. */
export class Shuffler {
  constructor(pool, ratings = {}, random = Math.random) {
    this.pool = pool.slice();
    this.ratings = ratings;
    this.random = random;
    this.played = new Set();
    this.last = null;
  }

  get size() {
    return this.pool.length;
  }

  next() {
    if (this.pool.length === 0) return undefined;
    let remaining = this.pool.filter((t) => !this.played.has(t.id));
    if (remaining.length === 0) {
      this.played.clear();
      remaining = this.pool.filter((t) => t.id !== this.last?.id);
      if (remaining.length === 0) remaining = this.pool.slice();
    }
    const weights = remaining.map((t) => weightOf(t, this.ratings));
    const pick = weightedPick(remaining, weights, this.random);
    this.played.add(pick.id);
    this.last = pick;
    return pick;
  }
}

/** Plays a fixed list in order, then starts again from the top. */
export class InOrder {
  constructor(list) {
    this.list = list.slice();
    this.index = 0;
  }

  get size() {
    return this.list.length;
  }

  next() {
    if (this.list.length === 0) return undefined;
    const pick = this.list[this.index % this.list.length];
    this.index += 1;
    return pick;
  }
}

/** Resolves a playlist's ids to tracks, dropping any no longer in the library. */
export function resolvePlaylist(trackIds, byId) {
  return trackIds.map((id) => byId.get(id)).filter(Boolean);
}

/** Text search and tag filters for the library view. */
export function filterLibrary(tracks, { query = "", tags = [] } = {}) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return tracks.filter((t) => {
    const own = new Set(t.tags ?? []);
    if (!tags.every((tag) => own.has(tag))) return false;
    if (words.length === 0) return true;
    const hay = `${t.title} ${t.artist} ${t.album ?? ""} ${(t.tags ?? []).join(" ")}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
}

/** Tags by how many tracks carry them, most first, then alphabetically. */
export function tagCounts(tracks) {
  const counts = new Map();
  for (const t of tracks) for (const tag of t.tags ?? []) counts.set(tag, (counts.get(tag) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}
