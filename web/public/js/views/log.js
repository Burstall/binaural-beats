// LOG: minutes listened, what the blind test says, and every session — with
// a rating for any that hasn't been rated yet.

import { MODES, MODE_NAMES } from "../profile.js";
import { mulberry32 } from "../rng.js";
import { RATE_AFTER_MINUTES } from "../player.js";
import { blindSummary, dailyMinutes } from "../stats.js";
import { on, saveSession, state } from "../store.js";
import { clear, h, minutesText, openSheet } from "../ui.js";

let root;

const QUESTION = {
  focus: "How well did you focus?",
  create: "How freely did ideas come?",
  meditate: "How settled did you feel?",
};

/** Asks for a 1-5 rating; reveals the blind arm only once it's given. */
export function rateSession(session) {
  return openSheet((close) => {
    const note = h("textarea", { placeholder: "a note, if you like", "aria-label": "Note", maxlength: 2000, value: session.note ?? "" });
    const body = h("div", {});
    const finish = async (rating) => {
      const record = { ...session, rating, note: note.value.trim() || null };
      await saveSession(record);
      if (record.arm) {
        clear(body,
          h("h3", { text: "Thanks." }),
          h("p", {}, "That session had the pulse and binaural layer ", h("b", { class: "violet", text: record.arm === "on" ? "ON" : "OFF" }), "."),
          h("p", { class: "small dim", text: "Your rating was given before you knew, which is the whole point. LOG adds it up." }),
          h("button", { class: "big", text: "DONE", onclick: () => close(record) }),
        );
      } else {
        close(record);
      }
    };
    clear(body,
      h("h3", { text: QUESTION[session.mode] ?? "How was it?" }),
      h("p", { class: "small dim", text: `${MODE_NAMES[session.mode]} · ${minutesText(session.minutes)} · ${new Date(session.started_at).toLocaleString()}` }),
      h("div", { class: "stars", style: "margin:0.8rem 0" },
        [1, 2, 3, 4, 5].map((n) => h("button", { text: String(n), "aria-label": `${n} of 5`, onclick: () => finish(n) })),
      ),
      h("div", { class: "axis" }, h("span", { text: "not at all" }), h("span", { text: "completely" })),
      h("div", { class: "field" }, note),
      h("button", { class: "small ghost", text: "SKIP", onclick: () => close(null) }),
    );
    return body;
  });
}

function chart() {
  const days = dailyMinutes(state.sessions, { days: 30 });
  const max = Math.max(30, ...days.map((d) => d.focus + d.create + d.meditate));
  const total = days.reduce((a, d) => a + d.focus + d.create + d.meditate, 0);
  return [
    h("h2", { text: "30 DAYS" }),
    h("p", { class: "small dim", text: `${minutesText(total)} in all` }),
    h("div", { class: "chart", role: "img", "aria-label": `Minutes per day for 30 days, ${Math.round(total)} in all` },
      days.map((d) =>
        h("div", { class: "day", title: `${d.date}: ${Math.round(d.focus + d.create + d.meditate)} min` },
          MODES.map((m) => (d[m] > 0 ? h("i", { class: m, style: `height:${(d[m] / max) * 100}%` }) : null)),
        ),
      ),
    ),
    h("div", { class: "axis" }, h("span", { text: days[0].date.slice(5) }), h("span", { text: "today" })),
    h("div", { class: "legend" }, MODES.map((m) => h("span", {}, h("i", { class: m, style: `background:var(--${m === "focus" ? "fg" : m === "create" ? "violet" : "cyan"})` }), MODE_NAMES[m]))),
  ];
}

function blind() {
  const random = mulberry32(20261014);
  return [
    h("h2", { text: "BLIND TEST" }),
    h("p", { class: "small dim", text: "Each session flips a coin: the pulse and binaural layer on, or quietly off, with the same music at the same loudness. You rate before you're told. Only rated, blinded sessions count." }),
    MODES.map((mode) => {
      const r = blindSummary(state.sessions, mode, { random });
      return h("div", { class: "panel result" },
        h("div", { class: "row" }, h("b", { class: "grow", text: MODE_NAMES[mode] }),
          h("span", { class: "nums small dim", text: `on ${r.on.n}${r.on.mean != null ? ` · ${r.on.mean.toFixed(2)}` : ""}  |  off ${r.off.n}${r.off.mean != null ? ` · ${r.off.mean.toFixed(2)}` : ""}` })),
        h("p", { class: `small ${r.verdict === "better" ? "" : r.verdict === "worse" ? "warn" : "dim"}`, text: r.message }),
      );
    }),
  ];
}

function sessionsList() {
  const recent = state.sessions.slice().sort((a, b) => b.started_at.localeCompare(a.started_at)).slice(0, 60);
  return [
    h("h2", { text: "SESSIONS" }),
    h("ul", { class: "list" },
      recent.length === 0 ? h("li", { class: "dim", style: "padding:1rem 0", text: "Nothing yet." }) : null,
      recent.map((s) => {
        const rateable = s.rating == null && s.minutes >= RATE_AFTER_MINUTES;
        const right = s.rating != null
          ? `${"*".repeat(s.rating)}${s.arm ? ` · ${s.arm}` : ""}`
          : s.arm ? "blind" : "";
        return h("li", {},
          h("div", { class: "item" },
            h("div", { class: "main" },
              h("div", { class: "t", text: `${MODE_NAMES[s.mode]} · ${minutesText(s.minutes)}` }),
              h("div", { class: "s", text: new Date(s.started_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) }),
            ),
            rateable ? h("button", { class: "small violet", text: "RATE", onclick: () => rateSession(s) }) : h("div", { class: "side", text: right }),
          ),
        );
      }),
    ),
  ];
}

function render() {
  if (!root) return;
  clear(root, chart(), blind(), sessionsList());
}

export function mount(section) {
  root = section;
  on("sessions", render);
  render();
}

export function show() {
  render();
}
