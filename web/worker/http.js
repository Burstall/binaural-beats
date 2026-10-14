// Small HTTP helpers shared by the API and the media proxy.

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function json(data, status = 200, extra = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extra,
    },
  });
}

export function errorResponse(err) {
  if (err instanceof HttpError) return json({ error: err.message }, err.status);
  console.error(err);
  return json({ error: "internal error" }, 500);
}

/** The request body as JSON, refusing anything over `limit` bytes. */
export async function readJson(request, limit = 65536) {
  const declared = Number(request.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) throw new HttpError(413, `body over ${limit} bytes`);
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.length > limit) throw new HttpError(413, `body over ${limit} bytes`);
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new HttpError(400, "body is not UTF-8");
  }
  try {
    return { value: JSON.parse(text), size: bytes.length };
  } catch {
    throw new HttpError(400, "body is not JSON");
  }
}
