// The profile: a person's settings, stored opaque by the worker. The app owns
// the shape. Missing fields take the defaults below; unknown fields are kept,
// so an older copy of the app never destroys a newer one's settings.

export const MODES = ["focus", "create", "meditate"];

export const MODE_NAMES = {
  focus: "Deep Focus",
  create: "Creativity",
  meditate: "Meditate",
};

export const DEFAULT_PROFILE = Object.freeze({
  v: 1,
  name: "",
  base_hz: 83.949,
  modes: {
    focus: {
      pulse_hz: 16, depth: 0.35, binaural_hz: 16, binaural_level: 0.5,
      tags: ["focus"], exclude: ["vocals"], playlist: null,
    },
    create: {
      pulse_hz: 10, depth: 0.2, binaural_hz: 10, binaural_level: 0.5,
      tags: ["create"], exclude: ["vocals"], playlist: null,
    },
    meditate: {
      binaural_from_hz: 10, binaural_to_hz: 5, glide_minutes: 12, minutes: 20,
      binaural_level: 0.6, music: false, tags: ["meditate"], playlist: null,
    },
  },
  background: { kind: "off", level: 0.4 },
  timer: { work_minutes: 50, break_minutes: 10, cycles: 1 },
  crossfade_seconds: 6,
  blind: { enabled: true },
  last_mode: "focus",
});

function isPlainObject(x) {
  return x !== null && typeof x === "object" && !Array.isArray(x);
}

/**
 * `stored` laid over `defaults`, recursively. A stored value wins when it has
 * the default's type (arrays count as their own type, and null is allowed
 * wherever the default is null or an object is expected); anything the
 * defaults don't mention is carried through untouched.
 */
export function mergeProfile(stored, defaults = DEFAULT_PROFILE) {
  const out = structuredClone(defaults);
  if (!isPlainObject(stored)) return out;
  for (const [key, value] of Object.entries(stored)) {
    if (!(key in out)) {
      out[key] = value;
      continue;
    }
    const fallback = out[key];
    if (isPlainObject(fallback)) {
      if (isPlainObject(value)) out[key] = mergeProfile(value, fallback);
    } else if (fallback === null) {
      out[key] = value;
    } else if (Array.isArray(fallback)) {
      if (Array.isArray(value) && value.every((v) => typeof v === "string")) out[key] = value;
    } else if (typeof value === typeof fallback && (typeof value !== "number" || Number.isFinite(value))) {
      out[key] = value;
    }
  }
  return out;
}

export const LIMITS = {
  base_hz: [1, 2000],
  pulse_hz: [1, 64],
  depth: [0, 1],
  binaural_hz: [0, 40],
  binaural_level: [0, 1],
  binaural_from_hz: [0.5, 40],
  binaural_to_hz: [0.5, 40],
  glide_minutes: [0, 120],
  minutes: [1, 240],
  level: [0, 1],
  work_minutes: [0, 240],
  break_minutes: [0, 60],
  cycles: [1, 12],
  crossfade_seconds: [0, 20],
};

export function clamp(name, value) {
  const [lo, hi] = LIMITS[name];
  const v = Number(value);
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, v));
}
