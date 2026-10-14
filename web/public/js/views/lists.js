// LISTS: playlists — make, rename, reorder, delete, play, shuffle, and keep
// offline.

import { MODE_NAMES } from "../profile.js";
import { resolvePlaylist } from "../shuffle.js";
import { createPlaylist, deletePlaylist, on, playlist, savePlaylist, state } from "../store.js";
import { clear, durationText, h, minutesText, toast } from "../ui.js";
import { countCached, dropOffline, keepOffline, keptLists, storageText } from "../offline.js";
import { trackSheet } from "./library.js";
import { startSession } from "./play.js";

let root;
let openId = null;
let confirmDelete = false;
let progress = null;

function overview() {
  const name = h("input", { placeholder: "new playlist name", "aria-label": "New playlist name", maxlength: 80 });
  const create = async () => {
    const n = name.value.trim();
    if (!n) return;
    openId = await createPlaylist(n);
    render();
  };
  name.onkeydown = (e) => {
    if (e.key === "Enter") create();
  };
  const kept = keptLists();
  return [
    h("h2", { text: "PLAYLISTS" }),
    h("div", { class: "row" }, h("div", { class: "grow" }, name), h("button", { text: "CREATE", onclick: create })),
    h("ul", { class: "list", style: "margin-top:0.8rem" },
      state.playlists.length === 0 ? h("li", { class: "dim", style: "padding:1rem 0", text: "No playlists yet. Make one here, or add tracks from a track's sheet in LIBRARY." }) : null,
      state.playlists.map((p) => {
        const tracks = resolvePlaylist(p.track_ids, state.byId);
        const minutes = tracks.reduce((a, t) => a + (t.duration || 0), 0) / 60;
        return h("li", {},
          h("button", { class: "item", onclick: () => { openId = p.id; confirmDelete = false; render(); } },
            h("div", { class: "main" },
              h("div", { class: "t", text: p.name }),
              h("div", { class: "s", text: `${tracks.length} tracks · ${minutesText(minutes)}${kept.has(p.id) ? " · offline" : ""}` }),
            ),
            h("div", { class: "side", text: ">" }),
          ),
        );
      }),
    ),
    h("p", { class: "small dim", id: "storage" }),
  ];
}

function detail(p) {
  const tracks = resolvePlaylist(p.track_ids, state.byId);
  const name = h("input", { value: p.name, "aria-label": "Playlist name", maxlength: 80 });
  name.onchange = () => {
    const n = name.value.trim();
    if (n && n !== p.name) savePlaylist(p.id, n, p.track_ids);
  };
  const move = (i, d) => {
    const ids = [...p.track_ids];
    const j = i + d;
    if (j < 0 || j >= ids.length) return;
    [ids[i], ids[j]] = [ids[j], ids[i]];
    savePlaylist(p.id, p.name, ids);
  };
  const removeAt = (i) => {
    const ids = p.track_ids.filter((_, k) => k !== i);
    savePlaylist(p.id, p.name, ids);
  };
  const mode = state.profile.last_mode === "meditate" ? "focus" : state.profile.last_mode;
  const kept = keptLists().has(p.id);
  const offlineLine = h("p", { class: "small dim", id: "offline-line", text: kept ? "kept offline" : "" });
  if (kept) {
    countCached(tracks).then((n) => {
      offlineLine.textContent = `kept offline: ${n} of ${tracks.length} stored`;
    });
  }
  if (progress && progress.id === p.id) offlineLine.textContent = `downloading ${progress.done}/${progress.total}…`;

  return [
    h("button", { class: "small ghost", text: "< ALL PLAYLISTS", onclick: () => { openId = null; render(); } }),
    h("h2", { text: "PLAYLIST" }),
    name,
    h("div", { class: "controls", style: "margin-top:0.6rem" },
      h("button", { class: "primary", text: "PLAY IN ORDER", disabled: !tracks.length, onclick: () => startSession(mode, { playlistId: p.id, ordered: true }) }),
      h("button", { class: "violet", text: "SHUFFLE", disabled: !tracks.length, onclick: () => startSession(mode, { playlistId: p.id, ordered: false }) }),
    ),
    h("p", { class: "small dim", text: `plays as ${MODE_NAMES[mode]} — choose the mode on PLAY` }),
    h("div", { class: "row" },
      kept
        ? h("button", {
          class: "small", text: "REMOVE OFFLINE",
          onclick: async () => {
            await dropOffline(p.id, tracks, state.playlists);
            toast("Removed from this phone.");
            render();
          },
        })
        : h("button", {
          class: "small", text: "KEEP OFFLINE", disabled: !tracks.length || Boolean(progress),
          onclick: async () => {
            progress = { id: p.id, done: 0, total: tracks.length };
            render();
            try {
              await keepOffline(p.id, tracks, (done, total) => {
                progress = { id: p.id, done, total };
                const el = document.getElementById("offline-line");
                if (el) el.textContent = `downloading ${done}/${total}…`;
              });
              toast("Kept offline.");
            } catch (err) {
              toast(`Stopped: ${err.message}`, 5000);
            } finally {
              progress = null;
              render();
            }
          },
        }),
      confirmDelete
        ? h("button", { class: "small danger", text: "REALLY DELETE", onclick: async () => { await deletePlaylist(p.id); openId = null; confirmDelete = false; render(); } })
        : h("button", { class: "small danger", text: "DELETE", onclick: () => { confirmDelete = true; render(); } }),
    ),
    offlineLine,
    h("ul", { class: "list", style: "margin-top:0.6rem" },
      tracks.length === 0 ? h("li", { class: "dim", style: "padding:1rem 0", text: "Empty. Add tracks from LIBRARY: open a track, then 'add to playlist'." }) : null,
      p.track_ids.map((id, i) => {
        const t = state.byId.get(id);
        if (!t) return null;
        return h("li", {},
          h("div", { class: "item" },
            h("button", { class: "ghost main", style: "text-align:left;padding:0", onclick: () => trackSheet(t) },
              h("div", { class: "t", text: t.title }),
              h("div", { class: "s", text: `${t.artist} · ${durationText(t.duration)}` }),
            ),
            h("button", { class: "small ghost", "aria-label": "Move up", text: "↑", disabled: i === 0, onclick: () => move(i, -1) }),
            h("button", { class: "small ghost", "aria-label": "Move down", text: "↓", disabled: i === p.track_ids.length - 1, onclick: () => move(i, 1) }),
            h("button", { class: "small ghost", "aria-label": "Remove from playlist", text: "×", onclick: () => removeAt(i) }),
          ),
        );
      }),
    ),
  ];
}

function render() {
  if (!root) return;
  const p = openId ? playlist(openId) : null;
  if (openId && !p) openId = null;
  clear(root, p ? detail(p) : overview());
  const storage = document.getElementById("storage");
  if (storage) storageText().then((t) => { storage.textContent = t; }).catch(() => {});
}

export function mount(section) {
  root = section;
  for (const e of ["playlists", "library", "profile"]) on(e, render);
  render();
}

export function show() {
  render();
}
