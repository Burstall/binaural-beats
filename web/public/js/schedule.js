// Session timing, worked out once at the start and handed to the audio clock.
// Timers are throttled while the phone is locked; the audio clock is not, so
// every fade, break and chime is an automation point, not a setTimeout.

export const FADE_IN_SECONDS = 10;
export const FADE_OUT_SECONDS = 30;
export const BREAK_RAMP_SECONDS = 5;

/**
 * The plan for one session, in seconds from its start.
 *
 * `timer` is the profile's {work_minutes, break_minutes, cycles}. A
 * `work_minutes` of 0 means open-ended: no breaks, no end, no chimes.
 *
 * Returns {total, segments, chimes, workGain, master}: `segments` are the
 * work and break blocks; `chimes` the times a bell rings (each block's end);
 * `workGain` and `master` are [time, value] points to ramp linearly between —
 * `workGain` silences the music and binaural layer through each break,
 * `master` fades the whole session in and out.
 */
export function sessionPlan(timer, { fadeIn = FADE_IN_SECONDS, fadeOut = FADE_OUT_SECONDS, ramp = BREAK_RAMP_SECONDS } = {}) {
  const work = Math.max(0, Number(timer.work_minutes) || 0) * 60;
  if (work === 0) {
    return {
      total: Infinity,
      segments: [{ kind: "work", start: 0, end: Infinity }],
      chimes: [],
      workGain: [[0, 1]],
      master: [[0, 0], [fadeIn, 1]],
    };
  }
  const brk = Math.max(0, Number(timer.break_minutes) || 0) * 60;
  const cycles = Math.max(1, Math.floor(Number(timer.cycles) || 1));
  const segments = [];
  const chimes = [];
  const workGain = [[0, 1]];
  let t = 0;
  for (let c = 0; c < cycles; c += 1) {
    segments.push({ kind: "work", start: t, end: t + work });
    t += work;
    chimes.push(t);
    if (c < cycles - 1 && brk > 0) {
      const r = Math.min(ramp, brk / 2);
      segments.push({ kind: "break", start: t, end: t + brk });
      workGain.push([t, 1], [t + r, 0], [t + brk - r, 0], [t + brk, 1]);
      t += brk;
      chimes.push(t);
    }
  }
  const total = t;
  const scale = Math.min(1, total / (fadeIn + fadeOut));
  const fi = fadeIn * scale;
  const fo = fadeOut * scale;
  const master = [[0, 0], [fi, 1], [total - fo, 1], [total, 0]];
  return { total, segments, chimes, workGain, master };
}

/** The block in progress at `t` seconds into the session, or null after the end. */
export function segmentAt(plan, t) {
  for (const s of plan.segments) if (t >= s.start && t < s.end) return s;
  return null;
}

/**
 * The beat over a glide: exponential from `from` to `to` over `minutes`,
 * sampled at `steps` + 1 evenly spaced points. Exponential because halving
 * the rate sounds like the same size of step wherever it happens.
 */
export function glideCurve(from, to, minutes, steps) {
  const out = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i += 1) out[i] = from * (to / from) ** (i / steps);
  return out;
}

/**
 * Each ear's frequency over a glide. Web Audio's oscillator integrates its
 * frequency into phase itself, so writing these as automation curves plays
 * the beat asked for at every moment — the bug Stage 8 fixed in Python cannot
 * happen here.
 */
export function earCurves(carrier, from, to, minutes, steps) {
  const beat = glideCurve(from, to, minutes, steps);
  const left = new Float32Array(beat.length);
  const right = new Float32Array(beat.length);
  for (let i = 0; i < beat.length; i += 1) {
    left[i] = carrier - beat[i] / 2;
    right[i] = carrier + beat[i] / 2;
  }
  return [left, right];
}

/** Equal-power crossfade curves: the sum of squares stays 1 throughout. */
export function crossfadeCurves(steps = 64) {
  const out = new Float32Array(steps + 1);
  const into = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i += 1) {
    const x = (i / steps) * (Math.PI / 2);
    out[i] = Math.cos(x);
    into[i] = Math.sin(x);
  }
  return { out, into };
}

/** Minutes and seconds, for clocks. */
export function clock(seconds) {
  if (!Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
