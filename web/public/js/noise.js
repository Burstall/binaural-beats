// The "noise" background: soft pink noise as a seamless loop, made in the
// browser so it needs no download.

/**
 * Closes `x` into a loop of `n` samples with an equal-power fold of `fade`
 * samples: the loop's first `fade` samples blend the material that would have
 * followed sample n-1 (fading out) with the original opening (fading in). So
 * the wrap from n-1 back to 0 continues exactly as the signal itself would
 * have. Needs x.length >= n + fade.
 */
export function seamlessLoop(x, n, fade) {
  if (x.length < n + fade) throw new RangeError("need n + fade samples to fold a loop");
  const out = new Float32Array(n);
  out.set(x.subarray(0, n));
  for (let i = 0; i < fade; i += 1) {
    const a = (i / fade) * (Math.PI / 2);
    out[i] = x[n + i] * Math.cos(a) + x[i] * Math.sin(a);
  }
  return out;
}

/**
 * Pink noise (Paul Kellet's refined filter) softened with a one-pole
 * low-pass, `length` samples, scaled to a peak near `peak`. The filters run
 * for a second first so the loop doesn't open on their warm-up.
 */
export function pinkNoise(length, sampleRate, random, peak = 0.5) {
  const warm = sampleRate;
  const out = new Float32Array(length);
  let b0 = 0; let b1 = 0; let b2 = 0; let b3 = 0; let b4 = 0; let b5 = 0; let b6 = 0;
  let lp = 0;
  const k = 1 - Math.exp((-2 * Math.PI * 2500) / sampleRate);
  let max = 0;
  for (let i = -warm; i < length; i += 1) {
    const white = random() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.969 * b2 + white * 0.153852;
    b3 = 0.8665 * b3 + white * 0.3104856;
    b4 = 0.55 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.016898;
    const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
    b6 = white * 0.115926;
    lp += k * (pink - lp);
    if (i >= 0) {
      out[i] = lp;
      max = Math.max(max, Math.abs(lp));
    }
  }
  const scale = max > 0 ? peak / max : 0;
  for (let i = 0; i < length; i += 1) out[i] *= scale;
  return out;
}

/** One channel of loopable noise: `seconds` long with a two-second fold. */
export function noiseLoop(seconds, sampleRate, random) {
  const n = Math.round(seconds * sampleRate);
  const fade = Math.round(2 * sampleRate);
  return seamlessLoop(pinkNoise(n + fade, sampleRate, random), n, fade);
}
