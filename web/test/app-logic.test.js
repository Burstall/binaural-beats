import { test } from "node:test";
import assert from "node:assert/strict";
import { autoOctaves, binauralGain, carrierFor, earFrequencies, nextBeatTime, pulseRate, tempoLocked } from "../public/js/tuning.js";
import { DEFAULT_PROFILE, clamp, mergeProfile } from "../public/js/profile.js";
import { InOrder, Shuffler, eligible, filterLibrary, resolvePlaylist, tagCounts, weightedPick } from "../public/js/shuffle.js";
import { assignArm, mulberry32, randomId } from "../public/js/rng.js";
import { blindSummary, bootstrapDiff, dailyMinutes } from "../public/js/stats.js";
import { clock, crossfadeCurves, earCurves, glideCurve, segmentAt, sessionPlan } from "../public/js/schedule.js";
import { noiseLoop, pinkNoise, seamlessLoop } from "../public/js/noise.js";

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// --- tuning: values printed by violet itself -------------------------------

test("carrierFor agrees with violet.tuning.carrier_for", () => {
  const expected = { 83.949: 335.796, 40: 320, 250: 250, 520: 520, 1000: 500, 7.83: 250.56, 432: 432, 136.1: 272.2 };
  for (const [f, c] of Object.entries(expected)) close(carrierFor(Number(f)), c, 1e-9);
  assert.equal(autoOctaves(83.949), 2);
  assert.throws(() => carrierFor(0), RangeError);
  assert.throws(() => carrierFor(450, 300, 400), RangeError, "a window narrower than an octave can miss");
});

test("tempoLocked agrees with violet.dsp.modulation.tempo_locked", () => {
  const cases = [[16, 84, 16.8], [16, 120, 16], [10, 90, 9], [16, 60, 16], [10, 72.5, 9.666666666666666], [16, 170, 17]];
  for (const [target, bpm, want] of cases) close(tempoLocked(target, bpm), want, 1e-9);
  assert.throws(() => tempoLocked(0, 90), RangeError);
  assert.throws(() => tempoLocked(16, 0), RangeError);
});

test("pulseRate locks only to a steady beat", () => {
  close(pulseRate(16, { bpm: 84, beat_strength: 0.6 }), 16.8);
  assert.equal(pulseRate(16, { bpm: 84, beat_strength: 0.1 }), 16);
  assert.equal(pulseRate(16, { bpm: null }), 16);
  assert.equal(pulseRate(16, null), 16);
});

test("nextBeatTime lands on the track's beat grid, never in the past", () => {
  const track = { bpm: 120, beat_offset: 0.25 };
  close(nextBeatTime(10, 0, track), 10.25);
  close(nextBeatTime(10, 0.25, track), 10);
  close(nextBeatTime(10, 0.3, track), 10.45);
  close(nextBeatTime(5, 61.1, track), 5 + (61.25 - 61.1));
  assert.equal(nextBeatTime(7, 3, { bpm: null }), 7);
});

test("binaural level: 0 is silent, 0..1 spans -40..-18 dB", () => {
  assert.equal(binauralGain(0), 0);
  close(binauralGain(1), 10 ** (-18 / 20));
  close(binauralGain(0.5), 10 ** (-29 / 20));
  close(binauralGain(5), 10 ** (-18 / 20), 1e-12);
  assert.deepEqual(earFrequencies(335.796, 16), [327.796, 343.796]);
});

// --- profile ---------------------------------------------------------------

test("mergeProfile fills defaults and keeps unknown fields", () => {
  const merged = mergeProfile({ name: "Rich", modes: { focus: { depth: 0.5, future: 1 } }, extra: { a: [1] } });
  assert.equal(merged.name, "Rich");
  assert.equal(merged.modes.focus.depth, 0.5);
  assert.equal(merged.modes.focus.pulse_hz, 16);
  assert.equal(merged.modes.focus.future, 1);
  assert.deepEqual(merged.extra, { a: [1] });
  assert.equal(merged.modes.create.pulse_hz, 10);
  assert.equal(merged.timer.work_minutes, 50);
});

test("mergeProfile ignores values of the wrong type", () => {
  const merged = mergeProfile({ base_hz: "loud", modes: { focus: { tags: [1, 2], playlist: "abc12345" } }, blind: null, crossfade_seconds: NaN });
  assert.equal(merged.base_hz, 83.949);
  assert.deepEqual(merged.modes.focus.tags, ["focus"]);
  assert.equal(merged.modes.focus.playlist, "abc12345", "null defaults accept a value");
  assert.deepEqual(merged.blind, { enabled: true });
  assert.equal(merged.crossfade_seconds, 6);
  assert.deepEqual(mergeProfile(null), mergeProfile(undefined));
  assert.notEqual(mergeProfile(null).modes, DEFAULT_PROFILE.modes, "a copy, never the defaults themselves");
});

test("clamp keeps settings in range", () => {
  assert.equal(clamp("depth", 2), 1);
  assert.equal(clamp("depth", -1), 0);
  assert.equal(clamp("base_hz", "abc"), 1);
  assert.equal(clamp("cycles", 3), 3);
});

// --- shuffle ---------------------------------------------------------------

const lib = [
  { id: "a", title: "Rain", artist: "X", tags: ["focus", "beat"] },
  { id: "b", title: "Sun", artist: "Y", tags: ["focus", "vocals"] },
  { id: "c", title: "Snow", artist: "Z", tags: ["focus"] },
  { id: "d", title: "Wind", artist: "X", tags: ["meditate"] },
];

test("eligible: needs every tag, no excluded tag, no thumbs-down", () => {
  const ids = (xs) => xs.map((t) => t.id);
  assert.deepEqual(ids(eligible(lib, { tags: ["focus"], exclude: ["vocals"] })), ["a", "c"]);
  assert.deepEqual(ids(eligible(lib, { tags: ["focus"], exclude: [] }, { c: -1 })), ["a", "b"]);
  assert.deepEqual(ids(eligible(lib, { tags: ["focus", "beat"] })), ["a"]);
  assert.deepEqual(ids(eligible(lib, {})), ["a", "b", "c", "d"]);
});

test("Shuffler: no repeats within a round, no back-to-back across rounds", () => {
  const random = mulberry32(7);
  const s = new Shuffler(lib, {}, random);
  let previous = null;
  for (let round = 0; round < 50; round += 1) {
    const seen = new Set();
    for (let i = 0; i < lib.length; i += 1) {
      const t = s.next();
      assert.ok(!seen.has(t.id), "repeat within a round");
      assert.notEqual(t.id, previous, "same track twice in a row");
      seen.add(t.id);
      previous = t.id;
    }
  }
  assert.equal(new Shuffler([], {}).next(), undefined);
  const one = new Shuffler([lib[0]], {});
  assert.equal(one.next().id, "a");
  assert.equal(one.next().id, "a", "a pool of one repeats rather than stopping");
});

test("Shuffler: a thumbs-up triples the chance of being drawn first", () => {
  const random = mulberry32(42);
  const pool = lib.slice(0, 2);
  let firstA = 0;
  const n = 20000;
  for (let i = 0; i < n; i += 1) if (new Shuffler(pool, { a: 1 }, random).next().id === "a") firstA += 1;
  close(firstA / n, 0.75, 0.015);
});

test("weightedPick and InOrder", () => {
  assert.equal(weightedPick(["x", "y"], [0, 0], Math.random), undefined);
  assert.equal(weightedPick(["x", "y"], [1, 1], () => 0.99), "y");
  const o = new InOrder(lib.slice(0, 2));
  assert.deepEqual([o.next().id, o.next().id, o.next().id], ["a", "b", "a"]);
  assert.equal(new InOrder([]).next(), undefined);
});

test("library filter, tag counts, playlist resolution", () => {
  assert.deepEqual(filterLibrary(lib, { query: "x" }).map((t) => t.id), ["a", "d"]);
  assert.deepEqual(filterLibrary(lib, { query: "rain focus" }).map((t) => t.id), ["a"]);
  assert.deepEqual(filterLibrary(lib, { tags: ["focus", "vocals"] }).map((t) => t.id), ["b"]);
  assert.deepEqual(tagCounts(lib)[0], ["focus", 3]);
  const byId = new Map(lib.map((t) => [t.id, t]));
  assert.deepEqual(resolvePlaylist(["c", "gone", "a"], byId).map((t) => t.id), ["c", "a"]);
});

// --- rng -------------------------------------------------------------------

test("assignArm is a fair coin and randomId fits the worker's id rule", () => {
  const random = mulberry32(1);
  let on = 0;
  for (let i = 0; i < 10000; i += 1) if (assignArm(random) === "on") on += 1;
  close(on / 10000, 0.5, 0.02);
  assert.match(assignArm(), /^(on|off)$/);
  assert.match(randomId(), /^[A-Za-z0-9_-]{8,40}$/);
});

// --- stats -----------------------------------------------------------------

test("dailyMinutes: a row per day, oldest first, sums per mode", () => {
  const utc = (d) => d.toISOString().slice(0, 10);
  const now = new Date("2026-10-14T12:00:00Z");
  const sessions = [
    { mode: "focus", started_at: "2026-10-14T08:00:00Z", minutes: 50 },
    { mode: "focus", started_at: "2026-10-14T10:00:00Z", minutes: 25 },
    { mode: "meditate", started_at: "2026-10-13T21:00:00Z", minutes: 20 },
    { mode: "create", started_at: "2026-08-01T10:00:00Z", minutes: 99 },
  ];
  const days = dailyMinutes(sessions, { days: 3, now, dayKey: utc });
  assert.deepEqual(days.map((d) => d.date), ["2026-10-12", "2026-10-13", "2026-10-14"]);
  assert.deepEqual(days[2], { date: "2026-10-14", focus: 75, create: 0, meditate: 0 });
  assert.equal(days[1].meditate, 20);
});

test("blindSummary says nothing below ten rated sessions an arm", () => {
  const s = [];
  for (let i = 0; i < 9; i += 1) s.push({ mode: "focus", arm: "on", rating: 5 }, { mode: "focus", arm: "off", rating: 1 });
  s.push({ mode: "focus", arm: "on", rating: null }, { mode: "focus", arm: null, rating: 5 });
  const r = blindSummary(s, "focus", { random: mulberry32(3) });
  assert.equal(r.enough, false);
  assert.equal(r.verdict, "too-few");
  assert.equal(r.ci, null);
  assert.match(r.message, /Too few/);
  assert.equal(r.on.n, 9);
});

test("blindSummary: a real difference, no difference, and a reproducible interval", () => {
  const make = (on, off) => [
    ...on.map((rating) => ({ mode: "focus", arm: "on", rating })),
    ...off.map((rating) => ({ mode: "focus", arm: "off", rating })),
  ];
  const better = blindSummary(make([4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5], [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]), "focus", { random: mulberry32(9) });
  assert.equal(better.verdict, "better");
  close(better.diff, 2);
  assert.ok(better.ci[0] > 0 && better.ci[1] > better.ci[0]);
  const same = blindSummary(make([3, 4, 2, 5, 3, 4, 2, 5, 3, 4], [4, 3, 5, 2, 4, 3, 5, 2, 4, 3]), "focus", { random: mulberry32(9) });
  assert.equal(same.verdict, "no-difference");
  assert.ok(same.ci[0] < 0 && same.ci[1] > 0);
  const worse = blindSummary(make(Array(10).fill(1), Array(10).fill(4)), "focus", { random: mulberry32(9) });
  assert.equal(worse.verdict, "worse");
  assert.deepEqual(bootstrapDiff([1, 2, 3], [2, 3, 4], mulberry32(5), 500), bootstrapDiff([1, 2, 3], [2, 3, 4], mulberry32(5), 500));
  assert.equal(blindSummary(make([5], [1]), "create", { random: mulberry32(1) }).on.n, 0, "other modes don't count");
});

// --- schedule --------------------------------------------------------------

test("sessionPlan: one 50-minute block", () => {
  const p = sessionPlan({ work_minutes: 50, break_minutes: 10, cycles: 1 });
  assert.equal(p.total, 3000);
  assert.deepEqual(p.segments, [{ kind: "work", start: 0, end: 3000 }]);
  assert.deepEqual(p.chimes, [3000]);
  assert.deepEqual(p.master, [[0, 0], [10, 1], [2970, 1], [3000, 0]]);
  assert.deepEqual(p.workGain, [[0, 1]]);
});

test("sessionPlan: work/break cycles silence the music through each break", () => {
  const p = sessionPlan({ work_minutes: 25, break_minutes: 5, cycles: 3 });
  assert.equal(p.total, (25 * 3 + 5 * 2) * 60);
  assert.deepEqual(p.segments.map((s) => s.kind), ["work", "break", "work", "break", "work"]);
  assert.deepEqual(p.chimes, [1500, 1800, 3300, 3600, 5100]);
  assert.deepEqual(p.workGain.slice(1, 5), [[1500, 1], [1505, 0], [1795, 0], [1800, 1]]);
  for (let i = 1; i < p.workGain.length; i += 1) assert.ok(p.workGain[i][0] >= p.workGain[i - 1][0], "times never go backwards");
  assert.equal(segmentAt(p, 1600).kind, "break");
  assert.equal(segmentAt(p, 0).kind, "work");
  assert.equal(segmentAt(p, p.total), null);
});

test("sessionPlan: open-ended and very short sessions", () => {
  const open = sessionPlan({ work_minutes: 0, break_minutes: 10, cycles: 4 });
  assert.equal(open.total, Infinity);
  assert.deepEqual(open.chimes, []);
  const short = sessionPlan({ work_minutes: 0.5, break_minutes: 0, cycles: 1 });
  assert.equal(short.total, 30);
  const [, [fi], [fo]] = short.master;
  assert.ok(fi <= fo, "fades scale down rather than overlap");
  const tinyBreak = sessionPlan({ work_minutes: 1, break_minutes: 0.1, cycles: 2 });
  for (let i = 1; i < tinyBreak.workGain.length; i += 1) assert.ok(tinyBreak.workGain[i][0] >= tinyBreak.workGain[i - 1][0]);
});

test("glide: exponential, ends exact; ear curves straddle the carrier", () => {
  const g = glideCurve(10, 5, 12, 720);
  close(g[0], 10, 1e-6);
  close(g[720], 5, 1e-6);
  close(g[360], Math.sqrt(50), 1e-5);
  const [l, r] = earCurves(335.796, 10, 5, 12, 720);
  close(r[0] - l[0], 10, 1e-4);
  close(r[720] - l[720], 5, 1e-4);
  close((r[100] + l[100]) / 2, 335.796, 1e-4);
});

test("crossfade curves are equal-power", () => {
  const { out, into } = crossfadeCurves(32);
  for (let i = 0; i <= 32; i += 1) close(out[i] ** 2 + into[i] ** 2, 1, 1e-6);
  close(out[0], 1, 1e-7);
  close(into[32], 1, 1e-7);
});

test("clock formatting", () => {
  assert.equal(clock(0), "0:00");
  assert.equal(clock(61.4), "1:01");
  assert.equal(clock(3725), "1:02:05");
  assert.equal(clock(Infinity), "--:--");
});

// --- noise -----------------------------------------------------------------

test("seamlessLoop: the wrap continues the signal exactly", () => {
  const x = Float32Array.from({ length: 150 }, (_, i) => Math.sin(i / 7));
  const y = seamlessLoop(x, 100, 30);
  close(y[0], x[100], 1e-6);
  close(y[99], x[99], 1e-6);
  close(y[29], x[129] * Math.cos((29 / 30) * (Math.PI / 2)) + x[29] * Math.sin((29 / 30) * (Math.PI / 2)), 1e-6);
  assert.throws(() => seamlessLoop(x, 140, 30), RangeError);
});

test("noise loop: the step across the wrap is ordinary", () => {
  const sr = 8000;
  const y = noiseLoop(3, sr, mulberry32(11));
  assert.equal(y.length, 3 * sr);
  let maxStep = 0;
  for (let i = 1; i < y.length; i += 1) maxStep = Math.max(maxStep, Math.abs(y[i] - y[i - 1]));
  assert.ok(Math.abs(y[0] - y[y.length - 1]) <= maxStep, "wrap is no bigger than the largest step inside");
  const p = pinkNoise(sr, sr, mulberry32(2), 0.5);
  close(Math.max(...p.map(Math.abs)), 0.5, 1e-6);
});
