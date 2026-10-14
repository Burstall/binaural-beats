// One HTTP byte range, parsed against a body of `size` bytes. The worker and
// the service worker both serve audio with this; iOS will not play media from
// a server that ignores ranges.
//
// Returns null when the whole body should be served (no header, a malformed
// one, a multi-range request, or last < first, which RFC 9110 says to
// ignore), "unsatisfiable" for a 416, or {start, end} inclusive.
export function parseRange(header, size) {
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
