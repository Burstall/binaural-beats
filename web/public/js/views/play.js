// PLAY: choose a mode, start, and see what is playing.

import { MODES, MODE_NAMES, clamp } from "../profile.js";
import { NoTracks, player } from "../player.js";
import { clock } from "../schedule.js";
import { eligible, resolvePlaylist } from "../shuffle.js";
import { on, rate, setTags, state, updateProfile } from "../store.js";
import { clear, h, toast } from "../ui.js";

const BLURB = {
  focus: "music · 16 Hz pulse · beta binaural",
  create: "music · lighter 10 Hz pulse · alpha binaural",
  meditate: "plain bed · alpha → theta glide · no pulse",
};

let root;
let ticker = null;

function selectedMode() {
  return MODES.includes(state.profile.last_mode) ? state.profile.last_mode : "focus";
}

function sourceCount(mode) {
  const cfg = state.profile.modes[mode];
  if (mode === "meditate" && !cfg.music) return null;
  if (cfg.playlist) {
    const list = state.playlists.find((p) => p.id === cfg.playlist);
    if (list) return resolvePlaylist(list.track_ids, state.byId).length;
  }
  return eligible(state.tracks, cfg, state.ratings).length;
}

function sourceSelect(mode) {
  const cfg = state.profile.modes[mode];
  if (mode === "meditate" && !cfg.music) {
    return h("p", { class: "dim small", text: "Plays violet's meditation bed. Turn on music for Meditate in SETUP to use tracks instead." });
  }
  const select = h(
    "select",
    {
      "aria-label": "What to play",
      onchange: (e) => updateProfile((p) => {
        p.modes[mode].playlist = e.target.value || null;
      }),
    },
    h("option", { value: "", text: `Shuffle by tags: ${cfg.tags.join(" + ") || "everything"}${cfg.exclude.length ? `, no ${cfg.exclude.join(", no ")}` : ""}` }),
    state.playlists.map((p) => h("option", { value: p.id, text: `Playlist: ${p.name} (${p.track_ids.length})`, selected: cfg.playlist === p.id })),
  );
  const n = sourceCount(mode);
  return h("div", {}, select, h("p", { class: `small ${n ? "dim" : "warn"}`, text: n ? `${n} tracks to draw from` : "No tracks match — tag some in LIBRARY or pick a playlist." }));
}

function timerChips() {
  const t = state.profile.timer;
  const options = [[25, "25"], [50, "50"], [90, "90"], [0, "∞"]];
  return h(
    "div",
    { class: "row" },
    h("span", { class: "dim small", text: "WORK MIN" }),
    h(
      "div",
      { class: "seg grow" },
      options.map(([m, label]) =>
        h("button", {
          class: "small",
          "aria-pressed": String(t.work_minutes === m),
          text: label,
          onclick: () => updateProfile((p) => {
            p.timer.work_minutes = m;
          }),
        }),
      ),
    ),
    t.cycles > 1 && t.work_minutes > 0 ? h("span", { class: "dim small", text: `×${t.cycles}, ${t.break_minutes}m breaks` }) : null,
  );
}

function backgroundControls() {
  const bg = state.profile.background;
  const set = (kind, level) => {
    updateProfile((p) => {
      p.background.kind = kind;
      p.background.level = level;
    });
    player.setBackground(kind, level);
  };
  return h(
    "div",
    { class: "panel" },
    h("div", { class: "row" },
      h("span", { class: "dim small", text: "BACKGROUND" }),
      h("div", { class: "seg grow" },
        ["off", "ocean", "noise"].map((kind) =>
          h("button", { class: "small", "aria-pressed": String(bg.kind === kind), text: kind.toUpperCase(), onclick: () => set(kind, bg.level) }),
        ),
      ),
    ),
    h("input", {
      type: "range", min: 0, max: 1, step: 0.01, value: bg.level, "aria-label": "Background level",
      disabled: bg.kind === "off",
      oninput: (e) => {
        const level = clamp("level", e.target.value);
        state.profile.background.level = level;
        player.setBackground(bg.kind, level);
      },
      onchange: (e) => set(bg.kind, clamp("level", e.target.value)),
    }),
  );
}

function idle() {
  const mode = selectedMode();
  return [
    h("div", { class: "modes", role: "group", "aria-label": "Mode" },
      MODES.map((m) =>
        h("button", {
          class: "mode",
          "aria-pressed": String(m === mode),
          onclick: () => updateProfile((p) => {
            p.last_mode = m;
          }),
        }, h("b", { text: MODE_NAMES[m].toUpperCase() }), h("small", { text: BLURB[m] })),
      ),
    ),
    h("h2", { text: "SOURCE" }),
    sourceSelect(mode),
    mode !== "meditate" ? h("div", { style: "margin-top:0.6rem" }, timerChips()) : h("p", { class: "dim small", text: `${state.profile.modes.meditate.minutes} minutes` }),
    state.profile.blind.enabled ? h("p", { class: "small violet", text: "Blind test on: half of sessions quietly leave the pulse and binaural layer out." }) : null,
    h("div", { style: "margin-top:1rem" },
      h("button", { class: "primary big", id: "start", text: `START ${MODE_NAMES[mode].toUpperCase()}`, onclick: () => startSession(mode) }),
    ),
    backgroundControls(),
  ];
}

export function startSession(mode, options) {
  try {
    player.start(mode, options);
    location.hash = "#/play";
  } catch (err) {
    if (err instanceof NoTracks) toast(err.message, 5000);
    else {
      console.error(err);
      toast(`Could not start: ${err.message}`, 5000);
    }
  }
}

function nowPlaying(st) {
  const t = st.track;
  if (!t) {
    return h("div", { class: "panel now" },
      h("div", { class: "title", text: st.mode === "meditate" ? "Meditation bed" : "Loading…" }),
      h("div", { class: "meta", text: st.source }),
    );
  }
  const rating = state.ratings[t.id] ?? 0;
  const vocals = (t.tags ?? []).includes("vocals");
  return h("div", { class: "panel now" },
    h("div", { class: "title ellipsis", text: t.title }),
    h("div", { class: "artist ellipsis", text: t.artist }),
    h("div", { class: "meta ellipsis", text: `${t.license}${t.bpm ? ` · ${Math.round(t.bpm)} bpm` : ""} · ${st.source}` }),
    h("div", { class: "now-actions" },
      h("button", { class: "small", "aria-pressed": String(rating === 1), "aria-label": "Thumbs up", text: "[+]", onclick: () => rate(t.id, rating === 1 ? 0 : 1) }),
      h("button", {
        class: "small", "aria-pressed": String(rating === -1), "aria-label": "Thumbs down, and skip", text: "[-]",
        onclick: async () => {
          await rate(t.id, rating === -1 ? 0 : -1);
          if (rating !== -1) player.skip();
        },
      }),
      h("button", { class: "small", "aria-label": "Skip", text: "[>>]", onclick: () => player.skip() }),
      h("button", {
        class: "small", "aria-pressed": String(vocals), "aria-label": "Has vocals", text: "VOX",
        onclick: async () => {
          await setTags(t.id, vocals ? [] : ["vocals"], vocals ? ["vocals"] : []);
          if (!vocals && state.profile.modes[st.mode].exclude?.includes("vocals")) player.skip();
          else render();
        },
      }),
    ),
  );
}

function readout(st) {
  if (st.blind) return h("p", { class: "readout", text: "BLIND SESSION — what's on stays hidden until you rate it." });
  const parts = [];
  if (st.pulseHz) parts.push(h("span", {}, "pulse ", h("b", { text: `${st.pulseHz.toFixed(2)} Hz` }), st.track?.bpm ? ` locked to ${Math.round(st.track.bpm * 10) / 10} bpm` : "", ` · depth ${st.depth}`));
  else if (st.mode !== "meditate") parts.push("pulse off");
  if (st.carrier) parts.push(h("span", {}, " · binaural ", h("b", { text: st.beat }), ` around ${st.carrier.toFixed(1)} Hz`));
  return h("p", { class: "readout" }, parts);
}

function segmentsStrip(plan) {
  if (!Number.isFinite(plan.total) || plan.segments.length < 2) return null;
  return h("div", { class: "segments" },
    plan.segments.map((s) => h("i", { class: s.kind, style: `flex:${s.end - s.start}` })),
  );
}

function running(st) {
  const finite = Number.isFinite(st.total);
  const works = st.plan.segments.filter((s) => s.kind === "work");
  const seg = st.segment;
  const label = seg ? (seg.kind === "break" ? "BREAK" : `WORK ${works.indexOf(seg) + 1}/${works.length}`) : "";
  return [
    h("p", { class: "dim small", text: `${MODE_NAMES[st.mode].toUpperCase()}${st.paused ? " · PAUSED" : ""}` }),
    nowPlaying(st),
    h("div", { class: "panel" },
      h("div", { class: "clock", id: "clock" }, clock(st.elapsed), h("small", { text: finite ? ` / ${clock(st.total)}` : "" })),
      h("div", { class: "row" }, h("span", { class: seg?.kind === "break" ? "violet" : "", id: "segment", text: label })),
      finite ? h("div", { class: "bar" }, h("i", { id: "progress", class: seg?.kind === "break" ? "break" : "", style: `width:${Math.min(100, (st.elapsed / st.total) * 100)}%` })) : null,
      segmentsStrip(st.plan),
      readout(st),
      st.contextState && st.contextState !== "running" && !st.paused ? h("p", { class: "warn small", text: `audio ${st.contextState} — tap RESUME` }) : null,
    ),
    h("div", { class: "controls" },
      h("button", { class: "big", text: st.paused || (st.contextState && st.contextState !== "running") ? "RESUME" : "PAUSE", onclick: () => (st.paused || st.contextState !== "running" ? player.resume() : player.pause()) }),
      h("button", { class: "big danger", text: "STOP", onclick: () => player.stop() }),
    ),
    backgroundControls(),
  ];
}

function render() {
  if (!root) return;
  const st = player.status();
  clear(root, st ? running(st) : idle());
}

function tickClock() {
  const st = player.status();
  if (!st) return;
  const el = document.getElementById("clock");
  if (el) el.firstChild.textContent = clock(st.elapsed);
  const bar = document.getElementById("progress");
  if (bar && Number.isFinite(st.total)) bar.style.width = `${Math.min(100, (st.elapsed / st.total) * 100)}%`;
  const segEl = document.getElementById("segment");
  if (segEl && st.segment && (segEl.className === "violet") !== (st.segment.kind === "break")) render();
  player.tick();
}

export function mount(section) {
  root = section;
  for (const e of ["profile", "library", "ratings", "playlists", "player"]) on(e, render);
  render();
}

export function show(visible) {
  clearInterval(ticker);
  ticker = null;
  if (visible) {
    render();
    ticker = setInterval(tickClock, 500);
  }
}

