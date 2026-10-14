// SETUP: the profile, credits for every track, and the account.

import { DEFAULT_PROFILE, MODE_NAMES, clamp, mergeProfile } from "../profile.js";
import { carrierFor } from "../tuning.js";
import { on, state, updateProfile } from "../store.js";
import { clear, extLink, h } from "../ui.js";

let root;
let resetArmed = false;
let selfUpdate = false;

// Changes made here don't re-render this view (that would fold every open
// section shut); everything else's changes do.
function save(fn) {
  selfUpdate = true;
  try {
    updateProfile(fn);
  } finally {
    selfUpdate = false;
  }
}

function num(label, name, get, set, { step = "any", unit = "" } = {}) {
  const input = h("input", { type: "number", inputmode: "decimal", step, value: get(), "aria-label": label });
  input.onchange = () => {
    const v = clamp(name, input.value);
    input.value = v;
    save((p) => set(p, v));
  };
  return h("label", { class: "field" }, h("span", {}, `${label}${unit ? ` (${unit})` : ""}`), input);
}

function slider(label, name, get, set, fmt = (v) => v.toFixed(2)) {
  const value = h("b", { class: "value", text: fmt(get()) });
  const input = h("input", { type: "range", min: 0, max: 1, step: 0.01, value: get(), "aria-label": label });
  input.oninput = () => {
    value.textContent = fmt(Number(input.value));
  };
  input.onchange = () => save((p) => set(p, clamp(name, input.value)));
  return h("label", { class: "field" }, h("span", {}, label, value), input);
}

function check(label, get, set) {
  const input = h("input", { type: "checkbox", checked: get() });
  input.onchange = () => save((p) => set(p, input.checked));
  return h("label", { class: "check" }, input, h("span", { text: label }));
}

function levelText(v) {
  return v > 0 ? `${Math.round(-40 + 22 * v)} dB` : "off";
}

function modeSection(mode) {
  const cfg = state.profile.modes[mode];
  if (mode === "meditate") {
    return h("details", {},
      h("summary", { text: `> ${MODE_NAMES[mode].toUpperCase()}` }),
      num("Beat at the start", "binaural_from_hz", () => cfg.binaural_from_hz, (p, v) => { p.modes.meditate.binaural_from_hz = v; }, { unit: "Hz" }),
      num("Beat at the end", "binaural_to_hz", () => cfg.binaural_to_hz, (p, v) => { p.modes.meditate.binaural_to_hz = v; }, { unit: "Hz" }),
      num("Glide over", "glide_minutes", () => cfg.glide_minutes, (p, v) => { p.modes.meditate.glide_minutes = v; }, { unit: "min" }),
      num("Session length", "minutes", () => cfg.minutes, (p, v) => { p.modes.meditate.minutes = v; }, { unit: "min" }),
      slider("Binaural level", "binaural_level", () => cfg.binaural_level, (p, v) => { p.modes.meditate.binaural_level = v; }, levelText),
      check("Play tracks tagged 'meditate' instead of the bed", () => cfg.music, (p, v) => { p.modes.meditate.music = v; }),
    );
  }
  return h("details", {},
    h("summary", { text: `> ${MODE_NAMES[mode].toUpperCase()}` }),
    num("Pulse rate", "pulse_hz", () => cfg.pulse_hz, (p, v) => { p.modes[mode].pulse_hz = v; }, { unit: "Hz, locked to each track's tempo" }),
    slider("Pulse depth", "depth", () => cfg.depth, (p, v) => { p.modes[mode].depth = v; }),
    num("Binaural beat", "binaural_hz", () => cfg.binaural_hz, (p, v) => { p.modes[mode].binaural_hz = v; }, { unit: "Hz" }),
    slider("Binaural level", "binaural_level", () => cfg.binaural_level, (p, v) => { p.modes[mode].binaural_level = v; }, levelText),
    h("p", { class: "small dim", text: `Shuffle draws tracks tagged ${cfg.tags.join(" + ") || "anything"}${cfg.exclude.length ? `, never ${cfg.exclude.join(" or ")}` : ""}.` }),
  );
}

function credits() {
  const list = h("ul", { class: "list" });
  const details = h("details", { class: "credits" }, h("summary", { text: `> CREDITS (${state.tracks.length} track${state.tracks.length === 1 ? "" : "s"})` }), list);
  details.ontoggle = () => {
    if (!details.open || list.childElementCount) return;
    const sorted = state.tracks.slice().sort((a, b) => a.artist.localeCompare(b.artist) || a.title.localeCompare(b.title));
    for (const t of sorted) {
      list.append(h("li", {}, t.attribution, " — ", extLink(t.license_url, t.license), " · ", extLink(t.source_url, "source")));
    }
  };
  return details;
}

function render() {
  if (!root) return;
  const p = state.profile;
  let carrier = "?";
  try {
    carrier = `${carrierFor(p.base_hz).toFixed(3)} Hz`;
  } catch {
    carrier = "out of range";
  }
  const name = h("input", { value: p.name, maxlength: 40, "aria-label": "Name" });
  name.onchange = () => save((q) => { q.name = name.value.trim(); });
  clear(root,
    h("h2", { text: "YOU" }),
    h("label", { class: "field" }, h("span", { text: "Name" }), name),
    num("Base frequency", "base_hz", () => p.base_hz, (q, v) => { q.base_hz = v; render(); }, { unit: "Hz" }),
    h("p", { class: "small dim" }, "Binaural carrier: ", h("b", { style: "color:var(--fg)", text: carrier }), " — the base shifted by whole octaves into 250–520 Hz. The meditation bed is tuned to 83.949 Hz whatever this says."),

    h("h2", { text: "MODES" }),
    modeSection("focus"),
    modeSection("create"),
    modeSection("meditate"),

    h("h2", { text: "SESSIONS" }),
    num("Work block", "work_minutes", () => p.timer.work_minutes, (q, v) => { q.timer.work_minutes = v; }, { unit: "min, 0 = no timer" }),
    num("Break", "break_minutes", () => p.timer.break_minutes, (q, v) => { q.timer.break_minutes = v; }, { unit: "min" }),
    num("Blocks", "cycles", () => p.timer.cycles, (q, v) => { q.timer.cycles = Math.round(v); }, { step: 1 }),
    num("Crossfade", "crossfade_seconds", () => p.crossfade_seconds, (q, v) => { q.crossfade_seconds = v; }, { unit: "s" }),
    check("Blind test: flip a coin each session", () => p.blind.enabled, (q, v) => { q.blind.enabled = v; }),
    h("p", { class: "small dim", text: "With the blind test on, half your sessions play the same music with no pulse and no binaural layer, and you rate each one before being told which it was. It is the only way to find out whether any of this does anything for you." }),

    h("h2", { text: "CREDITS" }),
    h("p", { class: "small dim", text: "Every track is someone's work, shared under the licence shown." }),
    credits(),

    h("h2", { text: "ACCOUNT" }),
    h("p", { class: "small" }, "Signed in as ", h("b", { text: state.email ?? "—" })),
    h("p", { class: "small" }, h("a", { href: "/cdn-cgi/access/logout", text: "Sign out" }), h("span", { class: "dim", text: " — your settings stay on the server for next time." })),
    resetArmed
      ? h("button", { class: "small danger", text: "REALLY RESET SETTINGS", onclick: () => { resetArmed = false; save((q) => { const keep = { name: q.name }; Object.assign(q, mergeProfile(keep, DEFAULT_PROFILE)); }); render(); } })
      : h("button", { class: "small danger", text: "RESET SETTINGS", onclick: () => { resetArmed = true; render(); } }),
    h("p", { class: "small dim", style: "margin-top:1.5rem", text: "violet · no health claims · it is music with a pulse, and a way to find out" }),
  );
}

export function mount(section) {
  root = section;
  on("profile", () => {
    if (!selfUpdate) render();
  });
  on("library", () => {
    if (root && !root.hidden) render();
  });
  render();
}

export function show() {
  render();
}
