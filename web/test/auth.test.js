import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { identify, resetKeyCache, teamOrigin, verifyAccessJwt } from "../worker/auth.js";
import { handle } from "../worker/index.js";
import { makeEnv, req } from "./helpers.js";

const TEAM = "https://violet.cloudflareaccess.com";
const AUD = "aud-tag-123";
const NOW = Date.UTC(2026, 9, 14, 12, 0, 0);

const enc = (obj) => Buffer.from(JSON.stringify(obj)).toString("base64url");

async function keypair() {
  const { privateKey, publicKey } = await crypto.subtle.generateKey(
    { name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
    true,
    ["sign", "verify"],
  );
  const jwk = await crypto.subtle.exportKey("jwk", publicKey);
  return { privateKey, jwk };
}

const keys = { k1: await keypair(), k2: await keypair() };

async function sign(payload, { kid = "k1", alg = "RS256", key = keys[kid]?.privateKey ?? keys.k1.privateKey } = {}) {
  const h = enc({ alg, kid, typ: "JWT" });
  const p = enc(payload);
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(`${h}.${p}`));
  return `${h}.${p}.${Buffer.from(sig).toString("base64url")}`;
}

function claims(extra = {}) {
  return { iss: TEAM, aud: [AUD], email: "Rich@Example.com", exp: NOW / 1000 + 600, iat: NOW / 1000 - 10, ...extra };
}

function certsFetch(kids = ["k1"]) {
  const calls = [];
  const fn = async (url) => {
    calls.push(url);
    return new Response(JSON.stringify({ keys: kids.map((kid) => ({ ...keys[kid].jwk, kid, alg: "RS256", use: "sig" })) }));
  };
  fn.calls = calls;
  return fn;
}

const opts = (fetchFn, extra = {}) => ({ teamDomain: TEAM, aud: AUD, now: NOW, fetchFn, ...extra });

beforeEach(() => resetKeyCache());

test("teamOrigin normalises the forms people paste", () => {
  for (const d of ["violet.cloudflareaccess.com", "https://violet.cloudflareaccess.com/", " http://violet.cloudflareaccess.com "]) {
    assert.equal(teamOrigin(d), TEAM);
  }
});

test("a good token verifies, and its certs are fetched once and cached", async () => {
  const f = certsFetch();
  const token = await sign(claims());
  assert.equal((await verifyAccessJwt(token, opts(f))).email, "Rich@Example.com");
  await verifyAccessJwt(token, opts(f));
  assert.deepEqual(f.calls, [`${TEAM}/cdn-cgi/access/certs`]);
});

test("a string aud is accepted as well as an array", async () => {
  assert.ok(await verifyAccessJwt(await sign(claims({ aud: AUD })), opts(certsFetch())));
});

test("every way a token can be wrong is refused", async () => {
  const f = certsFetch();
  const bad = {
    "wrong issuer": await sign(claims({ iss: "https://evil.cloudflareaccess.com" })),
    "wrong audience": await sign(claims({ aud: ["other"] })),
    expired: await sign(claims({ exp: NOW / 1000 - 1 })),
    "not yet valid": await sign(claims({ nbf: NOW / 1000 + 3600 })),
    "no email claim": await sign(claims({ email: undefined })),
    "unexpected alg": await sign(claims(), { alg: "HS256" }),
    "unknown signing key": await sign(claims(), { kid: "nope" }),
  };
  for (const [why, token] of Object.entries(bad)) {
    await assert.rejects(verifyAccessJwt(token, opts(f)), new RegExp(why), why);
  }
  const good = await sign(claims());
  const [h, p] = good.split(".");
  const forged = `${h}.${enc(claims({ email: "attacker@example.com" }))}.${good.split(".")[2]}`;
  await assert.rejects(verifyAccessJwt(forged, opts(f)), /bad signature/);
  const otherKey = await sign(claims(), { kid: "k1", key: keys.k2.privateKey });
  await assert.rejects(verifyAccessJwt(otherKey, opts(f)), /bad signature/);
  await assert.rejects(verifyAccessJwt("a.b", opts(f)), /not a JWT/);
  await assert.rejects(verifyAccessJwt(`${h}.!!!.${p}`, opts(f)), /malformed/);
});

test("rotated keys are picked up, but an unknown kid can't make it refetch more than once a minute", async () => {
  let kids = ["k1"];
  const calls = [];
  const f = async (url) => {
    calls.push(url);
    return new Response(JSON.stringify({ keys: kids.map((kid) => ({ ...keys[kid].jwk, kid })) }));
  };
  await verifyAccessJwt(await sign(claims()), opts(f));
  kids = ["k2"];
  const rotated = await sign(claims(), { kid: "k2" });
  await assert.rejects(verifyAccessJwt(rotated, opts(f, { now: NOW + 1000 })), /unknown signing key/);
  assert.equal(calls.length, 1, "within the floor: no refetch");
  assert.ok(await verifyAccessJwt(rotated, opts(f, { now: NOW + 61000 })));
  assert.equal(calls.length, 2);
});

test("identify: lower-cases the email, refuses missing and bad tokens with 401", async () => {
  const env = makeEnv({ DEV_EMAIL: "", ACCESS_TEAM_DOMAIN: "violet.cloudflareaccess.com", ACCESS_AUD: AUD });
  const deps = { fetchFn: certsFetch(), now: NOW };
  const token = await sign(claims());
  const host = "https://violet.example.workers.dev";
  const email = await identify(req("/api/me", { host, headers: { "cf-access-jwt-assertion": token } }), env, deps);
  assert.equal(email, "rich@example.com");
  await assert.rejects(identify(req("/api/me", { host }), env, deps), (e) => e.status === 401);
  await assert.rejects(identify(req("/api/me", { host, headers: { "cf-access-jwt-assertion": "x.y.z" } }), env, deps), (e) => e.status === 401);
  const spoof = req("/api/me", { host, headers: { "cf-access-authenticated-user-email": "rich@example.com" } });
  await assert.rejects(identify(spoof, env, deps), (e) => e.status === 401, "the plain email header is not trusted");
});

test("end to end: a signed request reaches the API as its owner", async () => {
  const env = makeEnv({ DEV_EMAIL: "", ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD });
  const token = await sign(claims());
  const res = await handle(
    req("/api/me", { host: "https://violet.example.workers.dev", headers: { "cf-access-jwt-assertion": token } }),
    env,
    { fetchFn: certsFetch(), now: NOW },
  );
  assert.equal(res.status, 200);
  assert.equal((await res.json()).email, "rich@example.com");
});
