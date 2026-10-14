// The LOG view's arithmetic: minutes listened per day, and what the blinded
// sessions say. Pure, so it can be tested without a browser.

import { MODES } from "./profile.js";

export const MIN_PER_ARM = 10;

/** YYYY-MM-DD in the device's own time zone. */
export function localDayKey(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/**
 * Minutes per mode for each of the last `days` days, oldest first, ending
 * with the day `now` falls in. Days with nothing are present with zeros.
 */
export function dailyMinutes(sessions, { days = 30, now = new Date(), dayKey = localDayKey } = {}) {
  const out = [];
  const index = new Map();
  for (let i = days - 1; i >= 0; i -= 1) {
    const d = new Date(now.getTime() - i * 86400000);
    const key = dayKey(d);
    if (index.has(key)) continue;
    const row = { date: key, focus: 0, create: 0, meditate: 0 };
    index.set(key, row);
    out.push(row);
  }
  for (const s of sessions) {
    const row = index.get(dayKey(new Date(s.started_at)));
    if (row && MODES.includes(s.mode)) row[s.mode] += Number(s.minutes) || 0;
  }
  return out;
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Bootstrap 95% interval for mean(on) - mean(off). */
export function bootstrapDiff(on, off, random, resamples = 2000) {
  const diffs = new Float64Array(resamples);
  for (let r = 0; r < resamples; r += 1) {
    let a = 0;
    for (let i = 0; i < on.length; i += 1) a += on[Math.floor(random() * on.length)];
    let b = 0;
    for (let i = 0; i < off.length; i += 1) b += off[Math.floor(random() * off.length)];
    diffs[r] = a / on.length - b / off.length;
  }
  const sorted = Array.from(diffs).sort((x, y) => x - y);
  return [quantile(sorted, 0.025), quantile(sorted, 0.975)];
}

function fmt(x) {
  return (x >= 0 ? "+" : "") + x.toFixed(2);
}

/**
 * What the blinded, rated sessions of one mode say. Honest by construction:
 * below MIN_PER_ARM rated sessions in either arm it declines to say anything.
 */
export function blindSummary(sessions, mode, { random, resamples = 2000, minPerArm = MIN_PER_ARM } = {}) {
  const rated = sessions.filter((s) => s.mode === mode && (s.arm === "on" || s.arm === "off") && s.rating >= 1);
  const on = rated.filter((s) => s.arm === "on").map((s) => s.rating);
  const off = rated.filter((s) => s.arm === "off").map((s) => s.rating);
  const result = {
    mode,
    on: { n: on.length, mean: on.length ? mean(on) : null },
    off: { n: off.length, mean: off.length ? mean(off) : null },
    diff: on.length && off.length ? mean(on) - mean(off) : null,
    ci: null,
    enough: on.length >= minPerArm && off.length >= minPerArm,
    verdict: "too-few",
    message: "",
  };
  if (!result.enough) {
    result.message =
      `Too few rated sessions to say anything yet: ${on.length} with the enhancements, ` +
      `${off.length} without. ${minPerArm} of each is the least that means anything.`;
    return result;
  }
  const [lo, hi] = bootstrapDiff(on, off, random, resamples);
  result.ci = [lo, hi];
  const span = `${fmt(lo)} to ${fmt(hi)}`;
  if (lo > 0) {
    result.verdict = "better";
    result.message =
      `Sessions with the enhancements rated ${fmt(result.diff)} on average (95% interval ${span}). ` +
      `That is evidence for you, on your ratings — not proof, and it can shrink as sessions add up.`;
  } else if (hi < 0) {
    result.verdict = "worse";
    result.message =
      `Sessions with the enhancements rated ${fmt(result.diff)} on average (95% interval ${span}). ` +
      `For you, so far, they seem to get in the way.`;
  } else {
    result.verdict = "no-difference";
    result.message =
      `No clear difference: ${fmt(result.diff)} on average, and the 95% interval (${span}) includes zero.`;
  }
  return result;
}
