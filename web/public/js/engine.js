// The sound. One AudioContext for the life of the page, built inside the
// first tap because iOS allows nothing else.
//
//   deck A, deck B: <audio> → source → norm → fade ─┬───────────────→ music ─┐
//                                                  └→ band → dip(lfo) ─┘     │
//   binaural: two oscillators → merger(L, R) → binaural ────────────────────┤→ work → master ─┐
//   meditate bed: <audio> → source → bed ───────────────────────────────────┘                 │
//   background: looping buffer → background ─────────────────────────────────────→ master ────┤
//   chimes ──────────────────────────────────────────────────────────────────────────────────→ limiter → out
//
// The pulse is violet's: out = x − depth·m(t)·band(x), with m(t) the raised
// cosine (1 − cos 2πft)/2 — full level on every beat of the track.

import { noiseLoop } from "./noise.js";
import { crossfadeCurves, earCurves } from "./schedule.js";
import { binauralGain, dbToGain, earFrequencies, nextBeatTime, pulseRate } from "./tuning.js";

const BAND_LO = 200;
const BAND_HI = 1000;
const GLIDE_STEPS_PER_MINUTE = 60;

function silentWavUrl() {
  // 0.1 s of 8 kHz mono silence: enough for iOS to count a play() as real.
  const n = 800;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const s = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  s(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); s(8, "WAVE"); s(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 16000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, "data"); v.setUint32(40, n * 2, true);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

class Deck {
  constructor(engine, name) {
    const ctx = engine.ctx;
    this.engine = engine;
    this.name = name;
    this.el = new Audio();
    this.el.preload = "auto";
    this.el.setAttribute("playsinline", "");
    this.source = ctx.createMediaElementSource(this.el);
    this.norm = new GainNode(ctx, { gain: 1 });
    this.fade = new GainNode(ctx, { gain: 0 });
    this.hp = new BiquadFilterNode(ctx, { type: "highpass", frequency: BAND_LO, Q: Math.SQRT1_2 });
    this.lp = new BiquadFilterNode(ctx, { type: "lowpass", frequency: BAND_HI, Q: Math.SQRT1_2 });
    this.dip = new GainNode(ctx, { gain: 0 });
    this.source.connect(this.norm).connect(this.fade);
    this.fade.connect(engine.music);
    this.fade.connect(this.hp).connect(this.lp).connect(this.dip).connect(engine.music);
    this.lfo = null;
    this.lfoAmount = null;
    this.track = null;
    this.pulseHz = null;
  }

  load(track) {
    this.track = track;
    this.pulseHz = null;
    this.el.loop = false;
    this.el.src = track.url;
    this.el.load();
    this.norm.gain.value = dbToGain(track.gain_db ?? 0);
  }

  get remaining() {
    const d = this.el.duration;
    return Number.isFinite(d) ? d - this.el.currentTime : Infinity;
  }

  stopPulse() {
    const now = this.engine.ctx.currentTime;
    if (this.lfo) {
      try {
        this.lfo.stop();
      } catch {
        /* never started */
      }
      this.lfo.disconnect();
      this.lfoAmount.disconnect();
    }
    this.lfo = null;
    this.lfoAmount = null;
    this.dip.gain.cancelScheduledValues(now);
    this.dip.gain.setValueAtTime(0, now);
  }

  /**
   * Starts the pulse on this deck's track: the mode's rate locked to the
   * track's tempo, the oscillator started on one of its beats. Call once the
   * element is actually playing, so its clock means something.
   */
  startPulse(targetHz, depth) {
    this.stopPulse();
    if (!(depth > 0) || !this.track) return null;
    const ctx = this.engine.ctx;
    const now = ctx.currentTime;
    const lookahead = 0.05;
    const f = pulseRate(targetHz, this.track);
    const t0 = nextBeatTime(now + lookahead, this.el.currentTime + lookahead, this.track);
    this.dip.gain.setValueAtTime(-depth / 2, now);
    this.lfo = new OscillatorNode(ctx, { frequency: f, periodicWave: this.engine.cosine });
    this.lfoAmount = new GainNode(ctx, { gain: depth / 2 });
    this.lfo.connect(this.lfoAmount).connect(this.dip.gain);
    this.lfo.start(t0);
    this.pulseHz = f;
    return f;
  }

  stop() {
    this.stopPulse();
    this.el.pause();
    this.track = null;
  }
}

export class Engine {
  constructor() {
    this.ctx = null;
    this.ready = false;
    this.binaural = null;
    this.background = { kind: "off", source: null };
    this.buffers = new Map();
    this.onstatechange = null;
    this.generation = 0;
  }

  /** Builds the graph and unlocks every media element. Call inside a tap. */
  unlock() {
    if (navigator.audioSession) {
      try {
        navigator.audioSession.type = "playback";
      } catch {
        /* older WebKit */
      }
    }
    if (this.ready) {
      if (this.ctx.state !== "running") this.ctx.resume().catch(() => {});
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx({ latencyHint: "playback" });
    this.ctx = ctx;
    ctx.onstatechange = () => this.onstatechange?.(ctx.state);
    this.cosine = ctx.createPeriodicWave(new Float32Array([0, 1]), new Float32Array([0, 0]));

    this.limiter = new DynamicsCompressorNode(ctx, { threshold: -3, knee: 0, ratio: 20, attack: 0.002, release: 0.25 });
    this.limiter.connect(ctx.destination);
    this.master = new GainNode(ctx, { gain: 0 });
    this.master.connect(this.limiter);
    this.work = new GainNode(ctx, { gain: 1 });
    this.work.connect(this.master);
    this.music = new GainNode(ctx, { gain: 1 });
    this.music.connect(this.work);
    this.binauralGain = new GainNode(ctx, { gain: 0 });
    this.binauralGain.connect(this.work);
    this.bedGain = new GainNode(ctx, { gain: 1 });
    this.bedGain.connect(this.work);
    this.backgroundGain = new GainNode(ctx, { gain: 0 });
    this.backgroundGain.connect(this.master);

    this.decks = [new Deck(this, "A"), new Deck(this, "B")];
    this.bed = new Audio();
    this.bed.preload = "auto";
    this.bed.setAttribute("playsinline", "");
    ctx.createMediaElementSource(this.bed).connect(this.bedGain);

    // iOS lets an element play later without a tap only if it has played
    // once inside one. Play a breath of silence on each, now.
    const silence = silentWavUrl();
    for (const el of [...this.decks.map((d) => d.el), this.bed]) {
      el.src = silence;
      const p = el.play();
      if (p) {
        p.then(() => {
          if (el.src === silence) el.pause();
        }).catch(() => {});
      }
    }
    ctx.resume().catch(() => {});
    this.ready = true;
  }

  get now() {
    return this.ctx ? this.ctx.currentTime : 0;
  }

  /** Writes [time, value] points (seconds from t0) as linear ramps. */
  automate(param, points, t0) {
    param.cancelScheduledValues(0);
    const [first, ...rest] = points;
    param.setValueAtTime(first[1], t0 + first[0]);
    for (const [t, v] of rest) {
      if (Number.isFinite(t)) param.linearRampToValueAtTime(v, t0 + t);
    }
  }

  applyPlan(plan, t0) {
    this.automate(this.master.gain, plan.master, t0);
    this.automate(this.work.gain, plan.workGain, t0);
    for (const c of plan.chimes) this.chime(t0 + c);
  }

  /** A soft bell: three inharmonic partials, struck and left to ring. */
  chime(at, level = 0.18) {
    const ctx = this.ctx;
    const partials = [[523.25, 1], [523.25 * 2.76, 0.45], [523.25 * 5.4, 0.2]];
    for (const [freq, amp] of partials) {
      const osc = new OscillatorNode(ctx, { frequency: freq });
      const g = new GainNode(ctx, { gain: 0 });
      g.gain.setValueAtTime(0, at);
      g.gain.linearRampToValueAtTime(level * amp, at + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, at + 4.5 / Math.sqrt(amp + 0.5));
      osc.connect(g).connect(this.limiter);
      osc.start(at);
      osc.stop(at + 5);
    }
  }

  /** Equal-power crossfade from one deck to the other over `seconds`. */
  crossfade(from, to, seconds, at = this.ctx.currentTime + 0.02) {
    const steps = Math.max(8, Math.round(seconds * 20));
    const { out, into } = crossfadeCurves(steps);
    const dur = Math.max(0.05, seconds);
    for (const [deck, curve] of [[from, out], [to, into]]) {
      deck.fade.gain.cancelScheduledValues(0);
      deck.fade.gain.setValueAtTime(curve[0], at);
      deck.fade.gain.setValueCurveAtTime(curve, at, dur);
    }
    return at + dur;
  }

  /** Brings a deck straight up, with no other deck to fade from. */
  open(deck) {
    const now = this.ctx.currentTime;
    deck.fade.gain.cancelScheduledValues(0);
    deck.fade.gain.setValueAtTime(1, now);
  }

  // --- binaural ------------------------------------------------------------

  stopBinaural() {
    if (!this.binaural) return;
    for (const osc of this.binaural.oscs) {
      try {
        osc.stop();
      } catch {
        /* not started */
      }
      osc.disconnect();
    }
    this.binaural.merger.disconnect();
    this.binaural = null;
  }

  /** A steady binaural pair: left ear carrier − beat/2, right carrier + beat/2. */
  startBinaural(carrier, beat, level, at) {
    this.stopBinaural();
    const ctx = this.ctx;
    const [l, r] = earFrequencies(carrier, beat);
    const left = new OscillatorNode(ctx, { frequency: l });
    const right = new OscillatorNode(ctx, { frequency: r });
    const merger = new ChannelMergerNode(ctx, { numberOfInputs: 2 });
    left.connect(merger, 0, 0);
    right.connect(merger, 0, 1);
    merger.connect(this.binauralGain);
    this.binauralGain.gain.cancelScheduledValues(0);
    this.binauralGain.gain.setValueAtTime(binauralGain(level), at);
    left.start(at);
    right.start(at);
    this.binaural = { oscs: [left, right], merger, carrier, beat };
  }

  /** The meditation glide: the beat falls exponentially, then holds. */
  startBinauralGlide(carrier, from, to, minutes, level, at) {
    this.startBinaural(carrier, from, level, at);
    if (!(minutes > 0) || from === to) return;
    const steps = Math.max(2, Math.round(minutes * GLIDE_STEPS_PER_MINUTE));
    const [lc, rc] = earCurves(carrier, from, to, minutes, steps);
    const [left, right] = this.binaural.oscs;
    left.frequency.setValueCurveAtTime(lc, at, minutes * 60);
    right.frequency.setValueCurveAtTime(rc, at, minutes * 60);
  }

  setBinauralLevel(level) {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.binauralGain.gain.cancelScheduledValues(now);
    this.binauralGain.gain.setTargetAtTime(binauralGain(level), now, 0.15);
  }

  // --- background --------------------------------------------------------

  async backgroundBuffer(kind) {
    if (this.buffers.has(kind)) return this.buffers.get(kind);
    let buffer;
    if (kind === "ocean") {
      const res = await fetch("/media/beds/ocean-loop.flac", { credentials: "same-origin" });
      if (!res.ok) throw new Error(`ocean: HTTP ${res.status}`);
      buffer = await this.ctx.decodeAudioData(await res.arrayBuffer());
    } else {
      const sr = this.ctx.sampleRate;
      const seconds = 20;
      buffer = this.ctx.createBuffer(2, seconds * sr, sr);
      for (let ch = 0; ch < 2; ch += 1) buffer.copyToChannel(noiseLoop(seconds, sr, Math.random), ch);
    }
    this.buffers.set(kind, buffer);
    return buffer;
  }

  /** Switches the background layer; level 0..1 maps to gain level². */
  async setBackground(kind, level) {
    const now = this.ctx.currentTime;
    const gain = Math.max(0, Math.min(1, level)) ** 2;
    if (kind === this.background.kind && this.background.source) {
      this.backgroundGain.gain.setTargetAtTime(gain, now, 0.2);
      return;
    }
    const old = this.background.source;
    if (old) {
      this.backgroundGain.gain.setTargetAtTime(0, now, 0.15);
      setTimeout(() => {
        try {
          old.stop();
        } catch {
          /* already stopped */
        }
        old.disconnect();
      }, 800);
    }
    this.background = { kind, source: null };
    if (kind === "off") return;
    let buffer;
    try {
      buffer = await this.backgroundBuffer(kind);
    } catch (err) {
      if (kind === "ocean") {
        console.warn("ocean loop unavailable, using noise", err);
        buffer = await this.backgroundBuffer("noise");
      } else throw err;
    }
    if (this.background.kind !== kind) return;
    const source = new AudioBufferSourceNode(this.ctx, { buffer, loop: true });
    source.connect(this.backgroundGain);
    const t = this.ctx.currentTime + (old ? 0.8 : 0);
    source.start(t);
    this.backgroundGain.gain.setTargetAtTime(gain, t, 0.4);
    this.background.source = source;
  }

  // --- whole session ----------------------------------------------------

  pause() {
    for (const d of this.decks) if (d.track) d.el.pause();
    if (!this.bed.paused) this.bed.pause();
    return this.ctx.suspend();
  }

  async resume(activeDecks, bedPlaying) {
    await this.ctx.resume();
    for (const d of activeDecks) await d.el.play().catch(() => {});
    if (bedPlaying) await this.bed.play().catch(() => {});
  }

  /** Marks the start of a session, so a previous stop can't touch it. */
  begin() {
    this.generation += 1;
  }

  stopBackground() {
    const old = this.background.source;
    this.background = { kind: "off", source: null };
    if (!old) return;
    try {
      old.stop();
    } catch {
      /* already stopped */
    }
    old.disconnect();
  }

  /** Fades everything out quickly and stops it. */
  stopAll(fadeSeconds = 0.6) {
    if (!this.ctx) return;
    const generation = ++this.generation;
    const now = this.ctx.currentTime;
    for (const param of [this.master.gain, this.work.gain]) {
      param.cancelScheduledValues(0);
      param.setValueAtTime(param.value, now);
    }
    this.master.gain.linearRampToValueAtTime(0, now + fadeSeconds);
    setTimeout(() => {
      if (this.generation !== generation) return;
      for (const d of this.decks) d.stop();
      this.bed.pause();
      this.stopBinaural();
      this.stopBackground();
    }, fadeSeconds * 1000 + 100);
  }
}
