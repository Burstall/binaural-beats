// LIBRARY: every track, searchable by text and tags; a sheet per track for
// its credit, its tags, a thumbs up or down, and adding it to a playlist.

import { MODES } from "../profile.js";
import { filterLibrary, tagCounts } from "../shuffle.js";
import { addToPlaylist, createPlaylist, on, rate, setTags, state } from "../store.js";
import { clear, debounce, durationText, extLink, h, openSheet, toast } from "../ui.js";

const PAGE = 120;
const TAG_RULE = /^[a-z0-9][a-z0-9-]{0,23}$/;

const filters = { query: "", tags: [] };
let shown = PAGE;
let root;
let listEl;
let chipsEl;
let countEl;

function ratingMark(id) {
  const r = state.ratings[id];
  if (r === 1) return h("span", { class: "up", text: "[+]" });
  if (r === -1) return h("span", { class: "down", text: "[-]" });
  return null;
}

function trackRow(t) {
  const modeTags = (t.tags ?? []).filter((x) => MODES.includes(x));
  return h("li", {},
    h("button", { class: "item", onclick: () => trackSheet(t) },
      h("div", { class: "main" },
        h("div", { class: "t", text: t.title }),
        h("div", { class: "s", text: `${t.artist} · ${durationText(t.duration)}${modeTags.length ? ` · ${modeTags.join(" ")}` : ""}` }),
      ),
      h("div", { class: "side" }, ratingMark(t.id)),
    ),
  );
}

function renderList() {
  const matches = filterLibrary(state.tracks, filters);
  countEl.textContent = `${matches.length} of ${state.tracks.length} tracks`;
  clear(listEl, matches.slice(0, shown).map(trackRow));
  if (matches.length > shown) {
    listEl.append(h("li", {}, h("button", { class: "item", onclick: () => { shown += PAGE; renderList(); } }, h("span", { class: "dim", text: `show ${Math.min(PAGE, matches.length - shown)} more` }))));
  }
  if (state.tracks.length === 0) {
    listEl.append(h("li", { class: "dim", style: "padding:1rem 0" }, "The library is empty. Tracks appear here once `violet library publish` has run."));
  }
}

function renderChips() {
  const counts = tagCounts(state.tracks);
  const top = counts.slice(0, 24).map(([tag]) => tag);
  for (const t of filters.tags) if (!top.includes(t)) top.unshift(t);
  clear(chipsEl, top.map((tag) => {
    const onNow = filters.tags.includes(tag);
    return h("button", {
      class: "chip", "aria-pressed": String(onNow),
      text: tag,
      onclick: () => {
        filters.tags = onNow ? filters.tags.filter((x) => x !== tag) : [...filters.tags, tag];
        shown = PAGE;
        renderChips();
        renderList();
      },
    });
  }));
}

function render() {
  if (!root || !listEl) return;
  renderChips();
  renderList();
}

export function mount(section) {
  root = section;
  const search = h("input", {
    type: "search", placeholder: "search title, artist, tag…", "aria-label": "Search the library", autocomplete: "off",
    oninput: debounce((e) => {
      filters.query = e.target.value;
      shown = PAGE;
      renderList();
    }, 150),
  });
  chipsEl = h("div", { class: "chips" });
  countEl = h("p", { class: "count" });
  listEl = h("ul", { class: "list" });
  clear(root, h("h2", { text: "LIBRARY" }), search, chipsEl, countEl, listEl);
  for (const e of ["library", "ratings"]) on(e, render);
  render();
}

export function show() {
  render();
}

// --- the track sheet --------------------------------------------------------

export function trackSheet(track) {
  return openSheet((close) => {
    const body = h("div", {});
    const draw = () => {
      const t = state.byId.get(track.id) ?? track;
      const rating = state.ratings[t.id] ?? 0;
      const tagInput = h("input", { placeholder: "add a tag", "aria-label": "New tag", autocapitalize: "none", autocomplete: "off" });
      const addTag = async () => {
        const tag = tagInput.value.trim().toLowerCase().replace(/\s+/g, "-");
        if (!TAG_RULE.test(tag)) {
          toast("Tags are lower-case letters, digits and hyphens, up to 24.");
          return;
        }
        await setTags(t.id, [tag], []);
        draw();
      };
      const listSelect = h("select", { "aria-label": "Playlist" },
        h("option", { value: "", text: "add to playlist…" }),
        state.playlists.map((p) => h("option", { value: p.id, text: p.name })),
        h("option", { value: "__new", text: "+ new playlist" }),
      );
      listSelect.onchange = async () => {
        let id = listSelect.value;
        if (!id) return;
        if (id === "__new") {
          id = await createPlaylist(`${t.title}`.slice(0, 60) || "New list");
        }
        const added = await addToPlaylist(id, t.id);
        toast(added ? "Added." : "Already on that list.");
        listSelect.value = "";
      };
      const facts = [
        ["length", durationText(t.duration)],
        ["tempo", t.bpm ? `${Math.round(t.bpm * 10) / 10} bpm (beat ${t.beat_strength?.toFixed?.(2) ?? "?"})` : "no steady beat"],
        ["energy", t.energy != null ? t.energy.toFixed(2) : "?"],
        ["loudness", t.loudness != null ? `${t.loudness.toFixed(1)} LUFS → ${t.gain_db >= 0 ? "+" : ""}${t.gain_db.toFixed(1)} dB` : "?"],
      ];
      clear(body,
        h("h3", { text: t.title }),
        h("p", {}, t.artist, t.album ? h("span", { class: "dim", text: ` · ${t.album}` }) : null),
        h("div", { class: "row", style: "margin:0.6rem 0" },
          h("button", { class: "small", "aria-pressed": String(rating === 1), text: "[+] LIKE", onclick: async () => { await rate(t.id, rating === 1 ? 0 : 1); draw(); } }),
          h("button", { class: "small", "aria-pressed": String(rating === -1), text: "[-] NEVER", onclick: async () => { await rate(t.id, rating === -1 ? 0 : -1); draw(); } }),
        ),
        h("div", { class: "small dim" }, facts.map(([k, v]) => h("div", {}, `${k.padEnd(9, " ")} `, h("span", { style: "color:var(--fg)", text: v })))),
        h("h2", { text: "TAGS" }),
        h("div", { class: "chips" },
          (t.tags ?? []).map((tag) =>
            h("button", {
              class: "chip on", "aria-label": `Remove tag ${tag}`,
              onclick: async () => { await setTags(t.id, [], [tag]); draw(); },
            }, tag, h("span", { class: "x", text: "×" })),
          ),
        ),
        h("div", { class: "row" },
          h("div", { class: "grow" }, tagInput),
          h("button", { class: "small", text: "ADD", onclick: addTag }),
        ),
        h("div", { class: "chips" },
          ["focus", "create", "meditate", "vocals", "instrumental"].filter((x) => !(t.tags ?? []).includes(x)).map((x) =>
            h("button", { class: "chip", text: `+ ${x}`, onclick: async () => { await setTags(t.id, [x], []); draw(); } }),
          ),
        ),
        h("h2", { text: "PLAYLIST" }),
        listSelect,
        h("h2", { text: "CREDIT" }),
        h("p", { class: "small", text: t.attribution }),
        h("p", { class: "small" }, "licence: ", extLink(t.license_url, t.license)),
        h("p", { class: "small" }, "source: ", extLink(t.source_url)),
      );
    };
    draw();
    return body;
  });
}
