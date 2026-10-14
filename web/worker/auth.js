// Who is asking. Cloudflare Access sits in front of the whole hostname and
// puts a signed JWT on every request it lets through; this verifies that JWT
// and takes the person's email from it. The plain
// Cf-Access-Authenticated-User-Email header is never trusted on its own.

import { HttpError } from "./http.js";

const KEY_TTL_MS = 60 * 60 * 1000;
const REFETCH_FLOOR_MS = 60 * 1000;
const SKEW_SECONDS = 60;

// team origin -> {at, keys: Map(kid -> CryptoKey)}
const keyCache = new Map();

export function resetKeyCache() {
  keyCache.clear();
}

/** "team.cloudflareaccess.com", with or without scheme or slash, as an origin. */
export function teamOrigin(domain) {
  const bare = String(domain).trim().replace(/\/+$/, "").replace(/^https?:\/\//, "");
  return `https://${bare}`;
}

function base64urlBytes(s) {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "=");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function base64urlJson(s) {
  return JSON.parse(new TextDecoder().decode(base64urlBytes(s)));
}

async function loadKeys(origin, fetchFn) {
  const res = await fetchFn(`${origin}/cdn-cgi/access/certs`);
  if (!res.ok) throw new Error(`access certs: HTTP ${res.status}`);
  const body = await res.json();
  const keys = new Map();
  for (const jwk of body.keys ?? []) {
    if (jwk.kty !== "RSA" || !jwk.kid) continue;
    const { kty, n, e } = jwk;
    const key = await crypto.subtle.importKey(
      "jwk",
      { kty, n, e, alg: "RS256", ext: true },
      { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
      false,
      ["verify"],
    );
    keys.set(jwk.kid, key);
  }
  return keys;
}

async function keyFor(origin, kid, fetchFn, now) {
  let entry = keyCache.get(origin);
  const stale = !entry || now - entry.at > KEY_TTL_MS;
  // An unknown kid usually means the keys rotated: look again, but not more
  // than once a minute however many bad tokens arrive.
  const rotated = entry && !entry.keys.has(kid) && now - entry.at > REFETCH_FLOOR_MS;
  if (stale || rotated) {
    entry = { at: now, keys: await loadKeys(origin, fetchFn) };
    keyCache.set(origin, entry);
  }
  return entry.keys.get(kid);
}

/**
 * The verified payload of an Access JWT, or a thrown Error saying why not.
 * Checks: RS256, a known signing key, the signature, `iss` equal to the team
 * origin, `aud` containing ours, `exp` in the future and `nbf` (if any) not
 * in it, and an email claim.
 */
export async function verifyAccessJwt(token, { teamDomain, aud, now = Date.now(), fetchFn = fetch }) {
  const parts = String(token).split(".");
  if (parts.length !== 3) throw new Error("not a JWT");
  const [h, p, s] = parts;
  let header;
  let payload;
  try {
    header = base64urlJson(h);
    payload = base64urlJson(p);
  } catch {
    throw new Error("malformed JWT");
  }
  if (header.alg !== "RS256") throw new Error(`unexpected alg ${header.alg}`);
  const origin = teamOrigin(teamDomain);
  const key = await keyFor(origin, header.kid, fetchFn, now);
  if (!key) throw new Error("unknown signing key");
  const ok = await crypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    key,
    base64urlBytes(s),
    new TextEncoder().encode(`${h}.${p}`),
  );
  if (!ok) throw new Error("bad signature");
  if (payload.iss !== origin) throw new Error("wrong issuer");
  const auds = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  if (!auds.includes(aud)) throw new Error("wrong audience");
  const nowSeconds = now / 1000;
  if (!(typeof payload.exp === "number" && payload.exp > nowSeconds)) throw new Error("expired");
  if (typeof payload.nbf === "number" && payload.nbf > nowSeconds + SKEW_SECONDS) throw new Error("not yet valid");
  if (typeof payload.email !== "string" || !payload.email.includes("@")) throw new Error("no email claim");
  return payload;
}

function isLocal(url) {
  return url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "[::1]";
}

/** The signed-in person's email, lower-cased, or an HttpError. */
export async function identify(request, env, { fetchFn = fetch, now = Date.now() } = {}) {
  const url = new URL(request.url);
  if (env.DEV_EMAIL && isLocal(url)) return String(env.DEV_EMAIL).toLowerCase();
  if (!env.ACCESS_TEAM_DOMAIN || !env.ACCESS_AUD) throw new HttpError(503, "access not configured");
  const token = request.headers.get("cf-access-jwt-assertion");
  if (!token) throw new HttpError(401, "not signed in");
  try {
    const payload = await verifyAccessJwt(token, {
      teamDomain: env.ACCESS_TEAM_DOMAIN,
      aud: env.ACCESS_AUD,
      now,
      fetchFn,
    });
    return payload.email.toLowerCase();
  } catch (err) {
    console.warn(`access jwt rejected: ${err.message}`);
    throw new HttpError(401, "not signed in");
  }
}
