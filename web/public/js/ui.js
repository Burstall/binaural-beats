// DOM helpers. Everything that came from a track's metadata is untrusted —
// it was typed by strangers on the Internet Archive — so text always goes in
// as text, and links only as http(s).

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "value") el.value = value;
    else if (key === "checked") el.checked = Boolean(value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

export function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

export function extLink(url, text) {
  const href = safeUrl(url);
  if (!href) return h("span", { text: text ?? "" });
  return h("a", { href, target: "_blank", rel: "noopener noreferrer", text: text ?? href });
}

let toastTimer = null;
export function toast(message, ms = 3200) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, ms);
}

/** Opens the bottom sheet with `build(close)`'s content; resolves when closed. */
export function openSheet(build) {
  const dialog = document.getElementById("sheet");
  return new Promise((resolve) => {
    const close = (value) => {
      if (dialog.open) dialog.close();
      resolve(value);
    };
    clear(dialog, h("button", { class: "close small ghost", "aria-label": "Close", onclick: () => close(undefined), text: "[x]" }), build(close));
    dialog.onclose = () => resolve(undefined);
    dialog.onclick = (e) => {
      if (e.target === dialog) close(undefined);
    };
    dialog.showModal();
  });
}

export function minutesText(minutes) {
  const m = Math.round(minutes);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
}

export function durationText(seconds) {
  const s = Math.round(seconds || 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function field(label, control, value) {
  return h("label", { class: "field" }, h("span", {}, label, value !== undefined ? h("b", { class: "value", text: value }) : null), control);
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
