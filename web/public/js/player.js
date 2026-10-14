// A listening session, from the tap that starts it to the rating that ends it.
//
// start() must be called straight from a tap handler: everything up to the
// first play() runs synchronously, because iOS only lets audio start inside
// the gesture itself.

import { Engine } from "./engine.js";
import { MODE_NAMES } from "./profile.js";
import { assignArm, randomId } from "./rng.js";
import { segmentAt, sessionPlan } from "./schedule.js";
import { InOrder, Shuffler, eligible, resolvePlaylist } from "./shuffle.js";
import { carrierFor } from "./tuning.js";
import { emit, saveSession, state } from "./store.js";

const PRELOAD_SECONDS = 30;
const SKIP_FADE_SECONDS = 1.5;
const SAVE_EVERY_SECONDS = 60;
export const RATE_AFTER_MINUTES = 10;

export class NoTracks extends Error {}

export class Player {
  constructor() {
    this.engine = new Engine();
    this.session = null;
    this.onfinish = null;
    this.engine.onstatechange = (s) => {
      this.contextState = s;
      emit("player");
    };
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && this.session) {
        if (this.engine.ctx.state !== "running" && !this.session.paused) this.engine.ctx.resume().catch(() => {});
        this.tick();
      }
    });
  }

  get running() {
    return Boolean(this.session);
  }

  /** What the views show. Never reveals the blind arm. */
  status() {
    const s = this.session;
    if (!s) return null;
    const elapsed = this.elapsed();
    const segment = segmentAt(s.plan, elapsed);
    const deck = s.active;
    const track = deck?.track ?? null;
    return {
      mode: s.mode,
      blind: s.record.arm !== null,
      paused: s.paused,
      elapsed,
      total: s.plan.total,
      plan: s.plan,
      segment,
      track,
      source: s.sourceLabel,
      pulseHz: s.record.arm === null && s.enhance ? deck?.pulseHz ?? null : null,
      depth: s.record.arm === null ? s.depth : null,
      carrier: s.record.arm === null && s.binauralLevel > 0 ? s.carrier : null,
      beat: s.record.arm === null && s.binauralLevel > 0 ? s.beatLabel : null,
      contextState: this.engine.ctx?.state,
    };
  }

  elapsed() {
    const s = this.session;
    return s ? Math.max(0, this.engine.now - s.t0) : 0;
  }

  picker(mode, cfg, { playlistId = null, ordered = false } = {}) {
    const id = playlistId ?? cfg.playlist;
    if (id) {
      const list = state.playlists.find((p) => p.id === id);
      if (list) {
        const tracks = resolvePlaylist(list.track_ids, state.byId).filter((t) => state.ratings[t.id] !== -1);
        return {
          picker: ordered ? new InOrder(tracks) : new Shuffler(tracks, state.ratings),
          label: `${ordered ? "playlist" : "shuffle"}: ${list.name}`,
        };
      }
    }
    const pool = eligible(state.tracks, cfg, state.ratings);
    const tags = cfg.tags.length ? cfg.tags.join(" + ") : "everything";
    const without = cfg.exclude.length ? `, no ${cfg.exclude.join(", no ")}` : "";
    return { picker: new Shuffler(pool, state.ratings), label: `shuffle: ${tags}${without}` };
  }

  start(mode, options = {}) {
    if (this.session) this.stop({ silent: true });
    const profile = state.profile;
    const cfg = profile.modes[mode];
    const music = mode !== "meditate" || cfg.music;
    let picker = null;
    let sourceLabel = "the meditation bed";
    if (music) {
      const chosen = this.picker(mode, cfg, options);
      picker = chosen.picker;
      sourceLabel = chosen.label;
      if (picker.size === 0) throw new NoTracks(`Nothing to play for ${MODE_NAMES[mode]}: ${sourceLabel} matches no tracks.`);
    }

    const engine = this.engine;
    engine.unlock();
    engine.begin();
    const arm = profile.blind.enabled ? assignArm() : null;
    const enhance = arm !== "off";
    const timer = mode === "meditate" ? { work_minutes: cfg.minutes, break_minutes: 0, cycles: 1 } : profile.timer;
    const plan = sessionPlan(timer);
    const t0 = engine.now + 0.1;
    const carrier = carrierFor(profile.base_hz);
    const binauralLevel = enhance ? cfg.binaural_level : 0;

    const record = {
      id: randomId(),
      mode,
      started_at: new Date().toISOString(),
      minutes: 0,
      arm,
      rating: null,
      note: null,
      track_ids: [],
    };
    this.session = {
      mode,
      cfg,
      record,
      plan,
      t0,
      enhance,
      depth: enhance && mode !== "meditate" ? cfg.depth : 0,
      carrier,
      binauralLevel,
      beatLabel: mode === "meditate" ? `${cfg.binaural_from_hz}→${cfg.binaural_to_hz} Hz` : `${cfg.binaural_hz} Hz`,
      picker,
      sourceLabel,
      active: null,
      next: null,
      fading: false,
      paused: false,
      lastSave: 0,
      music,
      handlers: [],
    };

    engine.applyPlan(plan, t0);

    if (music) {
      const [a] = engine.decks;
      const first = picker.next();
      this.session.active = a;
      this.load(a, first);
      engine.open(a);
      this.play(a);
    } else {
      engine.bed.loop = true;
      engine.bed.src = "/media/beds/meditate.mp3";
      engine.bed.play().catch((err) => console.warn("bed would not play", err));
    }

    if (mode === "meditate") {
      engine.startBinauralGlide(carrier, cfg.binaural_from_hz, cfg.binaural_to_hz, cfg.glide_minutes, binauralLevel, t0);
    } else {
      engine.startBinaural(carrier, cfg.binaural_hz, binauralLevel, t0);
    }
    engine.setBackground(profile.background.kind, profile.background.level).catch((e) => console.warn(e));

    this.mediaSession();
    this.save(true);
    emit("player");
  }

  load(deck, track) {
    deck.load(track);
    this.session.record.track_ids.push(track.id);
    const onTime = () => this.tick();
    const onEnded = () => this.ended(deck);
    const onPlaying = () => {
      if (this.session && deck.track === track && deck.pulseHz === null) {
        deck.startPulse(this.session.cfg.pulse_hz, this.session.depth);
        emit("player");
      }
    };
    for (const h of deck.handlers ?? []) deck.el.removeEventListener(h[0], h[1]);
    deck.handlers = [["timeupdate", onTime], ["ended", onEnded], ["playing", onPlaying]];
    for (const [e, fn] of deck.handlers) deck.el.addEventListener(e, fn);
  }

  play(deck) {
    const p = deck.el.play();
    if (p) p.catch((err) => console.warn(`deck ${deck.name} would not play`, err));
  }

  other(deck) {
    return this.engine.decks.find((d) => d !== deck);
  }

  /** Everything that has to happen on time, driven by the audio elements' own events. */
  tick() {
    const s = this.session;
    if (!s || s.paused) return;
    const elapsed = this.elapsed();
    if (elapsed >= s.plan.total) {
      this.finish("done");
      return;
    }
    if (elapsed - s.lastSave >= SAVE_EVERY_SECONDS) this.save();
    if (!s.music || !s.active || s.fading) return;
    const remaining = s.active.remaining;
    if (remaining <= PRELOAD_SECONDS && !s.next) {
      const track = s.picker.next();
      if (track) {
        s.next = this.other(s.active);
        this.load(s.next, track);
      }
    }
    if (s.next && remaining <= state.profile.crossfade_seconds) this.advance(remaining);
  }

  /** Crossfades to the preloaded deck. */
  advance(seconds) {
    const s = this.session;
    if (!s || !s.next || s.fading) return;
    const from = s.active;
    const to = s.next;
    s.fading = true;
    this.play(to);
    const done = this.engine.crossfade(from, to, Math.max(0.3, seconds));
    s.active = to;
    s.next = null;
    this.mediaSession();
    emit("player");
    const wait = Math.max(0, (done - this.engine.now) * 1000) + 200;
    setTimeout(() => {
      if (this.session !== s) return;
      from.stop();
      s.fading = false;
    }, wait);
    this.save();
  }

  ended(deck) {
    const s = this.session;
    if (!s || deck !== s.active || s.fading) return;
    if (!s.next) {
      const track = s.picker.next();
      if (!track) return;
      s.next = this.other(deck);
      this.load(s.next, track);
    }
    this.advance(0.3);
  }

  skip() {
    const s = this.session;
    if (!s || !s.music || s.fading) return;
    if (!s.next) {
      const track = s.picker.next();
      if (!track) return;
      s.next = this.other(s.active);
      this.load(s.next, track);
    }
    this.advance(SKIP_FADE_SECONDS);
  }

  async pause() {
    const s = this.session;
    if (!s || s.paused) return;
    s.paused = true;
    await this.engine.pause();
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "paused";
    emit("player");
  }

  async resume() {
    const s = this.session;
    if (!s || !s.paused) return;
    this.engine.unlock();
    s.paused = false;
    await this.engine.resume(s.active ? [s.active] : [], !s.music);
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "playing";
    emit("player");
  }

  /** Ends the session early. */
  stop({ silent = false } = {}) {
    if (!this.session) return null;
    return this.finish(silent ? "replaced" : "stopped");
  }

  finish(reason) {
    const s = this.session;
    if (!s) return null;
    s.record.minutes = Math.round(Math.min(this.elapsed(), s.plan.total) / 6) / 10;
    this.session = null;
    this.engine.stopAll(reason === "done" ? 0.2 : 0.6);
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = "none";
    saveSession(s.record).catch((e) => console.error(e));
    emit("player");
    if (reason !== "replaced") this.onfinish?.(s.record, reason);
    return s.record;
  }

  save(force = false) {
    const s = this.session;
    if (!s) return;
    const elapsed = this.elapsed();
    if (!force && elapsed - s.lastSave < 5) return;
    s.lastSave = elapsed;
    s.record.minutes = Math.round(elapsed / 6) / 10;
    saveSession(s.record).catch((e) => console.error(e));
  }

  /** Live changes from the PLAY view. */
  setBackground(kind, level) {
    if (this.engine.ready && this.session) this.engine.setBackground(kind, level).catch((e) => console.warn(e));
  }

  mediaSession() {
    if (!("mediaSession" in navigator) || !this.session) return;
    const s = this.session;
    const track = s.active?.track;
    try {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: track ? track.title : MODE_NAMES[s.mode],
        artist: track ? track.artist : "violet",
        album: `violet // ${MODE_NAMES[s.mode]}`,
        artwork: [
          { src: "/icons/icon-180.png", sizes: "180x180", type: "image/png" },
          { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
        ],
      });
      navigator.mediaSession.playbackState = s.paused ? "paused" : "playing";
      const handlers = {
        play: () => this.resume(),
        pause: () => this.pause(),
        stop: () => this.stop(),
        nexttrack: s.music ? () => this.skip() : null,
      };
      for (const [action, fn] of Object.entries(handlers)) {
        try {
          navigator.mediaSession.setActionHandler(action, fn);
        } catch {
          /* unsupported action */
        }
      }
    } catch (err) {
      console.warn("media session", err);
    }
  }
}

export const player = new Player();
