import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { WEB } from "./helpers.js";

const PUBLIC = join(WEB, "public");

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    return statSync(full).isDirectory() ? walk(full) : [full];
  });
}

function shell() {
  const src = readFileSync(join(PUBLIC, "sw.js"), "utf8");
  const m = /const SHELL = \[([\s\S]*?)\];/.exec(src);
  return [...m[1].matchAll(/"([^"]+)"/g)].map((x) => x[1]);
}

test("the service worker caches every file the app is made of, and nothing missing", () => {
  const listed = new Set(shell());
  const onDisk = walk(PUBLIC)
    .map((f) => `/${relative(PUBLIC, f).split(sep).join("/")}`)
    .filter((p) => p !== "/sw.js" && p !== "/index.html");
  for (const p of onDisk) assert.ok(listed.has(p), `${p} is not in SHELL — offline it would be missing`);
  for (const p of listed) {
    if (p === "/") continue;
    assert.ok(existsSync(join(PUBLIC, p)), `${p} is in SHELL but not on disk — install would fail`);
  }
});

test("the shell never lists /index.html: it redirects to /, and a redirect can't be cached for navigation", () => {
  assert.ok(!shell().includes("/index.html"));
  assert.ok(shell().includes("/"));
});
