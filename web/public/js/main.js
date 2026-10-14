// Boot, routing, the sign-in gate, and the status line.

import { player, RATE_AFTER_MINUTES } from "./player.js";
import { boot, goOffline, goOnline, on, state } from "./store.js";
import { clear, h } from "./ui.js";
import * as play from "./views/play.js";
import * as library from "./views/library.js";
import * as lists from "./views/lists.js";
import * as log from "./views/log.js";
import * as setup from "./views/setup.js";

const VIEWS = { play, library, lists, log, setup };
let current = null;

function route() {
  const name = (location.hash.match(/^#\/(\w+)/) ?? [])[1];
  const view = VIEWS[name] ? name : "play";
  if (view === current) return;
  for (const [key, mod] of Object.entries(VIEWS)) {
    const section = document.getElementById(`view-${key}`);
    const visible = key === view;
    section.hidden = !visible;
    mod.show?.(visible);
  }
  for (const a of document.querySelectorAll("[data-tab]")) {
    if (a.dataset.tab === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  document.getElementById("brand-view").textContent = view.toUpperCase();
  current = view;
  window.scrollTo(0, 0);
}

function statusLine() {
  const el = document.getElementById("status");
  const parts = [];
  if (!state.online) parts.push(h("span", { class: "off", text: "OFFLINE" }));
  if (state.pending) parts.push(h("span", { class: "off", text: ` ${state.pending} unsent` }));
  if (state.online && state.email) parts.push(h("span", { text: state.profile.name || state.email.split("@")[0] }));
  clear(el, parts);
}

const BANNER = String.raw`
       _       _      _
__   _(_) ___ | | ___| |_
\ \ / / |/ _ \| |/ _ \ __|
 \ V /| | (_) | |  __/ |_
  \_/ |_|\___/|_|\___|\__|`;

function gate() {
  const el = document.getElementById("gate");
  if (state.notConfigured) {
    el.hidden = false;
    clear(el, h("div", { class: "box" },
      h("pre", { text: BANNER }),
      h("p", { text: "Sign-in isn't switched on yet." }),
      h("p", { class: "small dim", text: "This app runs behind Cloudflare Access. Once it's enabled for this address, reload." }),
      h("button", { class: "big", text: "RELOAD", onclick: () => location.reload() }),
    ));
  } else if (state.signedOut) {
    el.hidden = false;
    clear(el, h("div", { class: "box" },
      h("pre", { text: BANNER }),
      h("p", { text: "You're signed out." }),
      h("p", { class: "small dim", text: "Sign in with the email address you were invited with; a one-time code arrives there." }),
      h("button", { class: "big primary", text: "SIGN IN", onclick: () => { location.href = "/"; } }),
    ));
  } else {
    el.hidden = true;
  }
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("/sw.js").catch((err) => console.warn("service worker", err));
}

async function start() {
  for (const [key, mod] of Object.entries(VIEWS)) mod.mount(document.getElementById(`view-${key}`));
  window.addEventListener("hashchange", route);
  route();
  on("network", () => {
    statusLine();
    gate();
  });
  on("profile", statusLine);
  window.addEventListener("online", () => goOnline());
  window.addEventListener("offline", () => goOffline());
  player.onfinish = (record) => {
    if (record.minutes >= RATE_AFTER_MINUTES) log.rateSession(record);
  };
  await boot();
  statusLine();
  gate();
  registerServiceWorker();
}

start().catch((err) => {
  console.error(err);
  const el = document.getElementById("gate");
  el.hidden = false;
  clear(el, h("div", { class: "box" }, h("p", { class: "bad", text: `violet failed to start: ${err.message}` }), h("button", { class: "big", text: "RELOAD", onclick: () => location.reload() })));
});
