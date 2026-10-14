// Random numbers. Seeded for tests and the bootstrap's reproducibility;
// crypto-backed for the blind assignment, where nobody should be able to
// predict the arm.

/** A small, fast seeded generator (mulberry32): returns floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A float in [0, 1) from the platform's cryptographic generator. */
export function cryptoRandom(cryptoImpl = globalThis.crypto) {
  const word = new Uint32Array(1);
  cryptoImpl.getRandomValues(word);
  return word[0] / 4294967296;
}

/** The blinded arm for one session: "on" or "off", one half each. */
export function assignArm(random = cryptoRandom) {
  return random() < 0.5 ? "on" : "off";
}

/** A random id for playlists and sessions, matching the worker's rule. */
export function randomId(length = 16, cryptoImpl = globalThis.crypto) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const bytes = new Uint8Array(length);
  cryptoImpl.getRandomValues(bytes);
  let out = "";
  for (const b of bytes) out += alphabet[b % alphabet.length];
  return out;
}
