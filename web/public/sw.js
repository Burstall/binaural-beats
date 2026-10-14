// violet's service worker.
//
// - The app shell is cache-first under a versioned cache. Bump VERSION with
//   every deploy that changes a file in SHELL, or phones keep the old one.
// - /api/library is network-first, with the last good copy kept for offline.
// - /media/ is answered from the offline cache when a playlist was kept
//   offline — slicing for Range itself, because iOS asks for ranges — and
//   from the network otherwise.
// - Every other /api/ request, and Access's own /cdn-cgi/ paths, go straight
//   to the network untouched.

const VERSION = "violet-v1";
const SHELL_CACHE = `${VERSION}-shell`;
const DATA_CACHE = "violet-data";
const MEDIA_CACHE = "violet-media";

const SHELL = [
  "/",
  "/manifest.webmanifest",
  "/css/app.css",
  "/icons/icon-180.png",
  "/icons/icon-512.png",
  "/js/main.js",
  "/js/api.js",
  "/js/idb.js",
  "/js/store.js",
  "/js/engine.js",
  "/js/player.js",
  "/js/offline.js",
  "/js/ui.js",
  "/js/tuning.js",
  "/js/profile.js",
  "/js/shuffle.js",
  "/js/rng.js",
  "/js/stats.js",
  "/js/schedule.js",
  "/js/noise.js",
  "/js/range.js",
  "/js/views/play.js",
  "/js/views/library.js",
  "/js/views/lists.js",
  "/js/views/log.js",
  "/js/views/setup.js",
];

// BEGIN parseRange
function parseRange(header, size) {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(String(header).trim());
  if (!m || (m[1] === "" && m[2] === "")) return null;
  let start;
  let end;
  if (m[1] === "") {
    const suffix = Number(m[2]);
    if (suffix === 0 || size === 0) return "unsatisfiable";
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(m[1]);
    if (m[2] !== "" && Number(m[2]) < start) return null;
    if (start >= size) return "unsatisfiable";
    end = m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
  }
  return { start, end };
}
// END parseRange

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("violet-v") && k !== SHELL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function fromMediaCache(request) {
  const cache = await caches.open(MEDIA_CACHE);
  const url = new URL(request.url);
  const hit = await cache.match(url.origin + url.pathname);
  if (!hit) return fetch(request);
  const blob = await hit.blob();
  const type = hit.headers.get("content-type") || "application/octet-stream";
  const range = parseRange(request.headers.get("range"), blob.size);
  const headers = { "content-type": type, "accept-ranges": "bytes" };
  if (range === "unsatisfiable") {
    return new Response(null, { status: 416, headers: { ...headers, "content-range": `bytes */${blob.size}` } });
  }
  if (!range) {
    return new Response(blob, { status: 200, headers: { ...headers, "content-length": String(blob.size) } });
  }
  const part = blob.slice(range.start, range.end + 1, type);
  return new Response(part, {
    status: 206,
    headers: {
      ...headers,
      "content-length": String(part.size),
      "content-range": `bytes ${range.start}-${range.end}/${blob.size}`,
    },
  });
}

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(request, { redirect: "manual" });
    const type = res.headers.get("content-type") || "";
    if (res.ok && type.includes("application/json")) {
      await cache.put(request, res.clone());
      return res;
    }
    const cached = await cache.match(request);
    return cached || res;
  } catch {
    const cached = await cache.match(request);
    return cached || new Response(JSON.stringify({ error: "offline" }), { status: 503, headers: { "content-type": "application/json" } });
  }
}

async function navigate(request) {
  try {
    return await fetch(request);
  } catch {
    const cache = await caches.open(SHELL_CACHE);
    return (await cache.match("/")) || Response.error();
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  const hit = await cache.match(request, { ignoreSearch: true });
  if (hit) return hit;
  return fetch(request);
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/cdn-cgi/")) return;
  if (url.pathname.startsWith("/media/")) {
    event.respondWith(fromMediaCache(request));
  } else if (url.pathname === "/api/library") {
    event.respondWith(networkFirst(request, DATA_CACHE));
  } else if (url.pathname.startsWith("/api/")) {
    return;
  } else if (request.mode === "navigate") {
    event.respondWith(navigate(request));
  } else {
    event.respondWith(cacheFirst(request));
  }
});
