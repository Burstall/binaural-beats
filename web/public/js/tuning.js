// Ports of violet's tuning and modulation rules. The Python package is the
// reference: web/test/tuning.test.js checks these against values it printed.

export const CARRIER_LO = 250.0;
export const CARRIER_HI = 520.0;
const OCTAVE_EPS = 1e-12;

// Subdivisions of a beat the pulse may lock to: duple and triple, as in
// violet.dsp.modulation.SUBDIVISIONS.
export const SUBDIVISIONS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

/** Whole-octave shift landing `freq` in [lo, hi]; the smallest shift wins. */
export function autoOctaves(freq, lo = CARRIER_LO, hi = CARRIER_HI) {
  if (!(freq > 0) || !Number.isFinite(freq)) {
    throw new RangeError(`frequency must be positive and finite, got ${freq}`);
  }
  const lowest = Math.ceil(Math.log2(lo / freq) - OCTAVE_EPS);
  const highest = Math.floor(Math.log2(hi / freq) + OCTAVE_EPS);
  if (lowest > highest) {
    throw new RangeError(`no whole-octave shift of ${freq} Hz lands in ${lo}-${hi} Hz`);
  }
  return Math.min(Math.max(0, lowest), highest);
}

/** The base frequency shifted by whole octaves into the binaural window. */
export function carrierFor(freq, lo = CARRIER_LO, hi = CARRIER_HI) {
  return freq * 2 ** autoOctaves(freq, lo, hi);
}

/**
 * The rate nearest `targetHz` that divides a beat at `bpm` evenly — nearest
 * by ratio, not by hertz. 16 Hz at 84 bpm is 16.8: twelve pulses a beat.
 */
export function tempoLocked(targetHz, bpm) {
  if (!(targetHz > 0)) throw new RangeError(`modulation rate must be positive, got ${targetHz}`);
  if (!(bpm > 0)) throw new RangeError(`tempo must be positive, got ${bpm}`);
  const beatHz = bpm / 60;
  let best = SUBDIVISIONS[0];
  let bestDistance = Infinity;
  for (const k of SUBDIVISIONS) {
    const distance = Math.abs(Math.log((k * beatHz) / targetHz));
    if (distance < bestDistance) {
      best = k;
      bestDistance = distance;
    }
  }
  return best * beatHz;
}

/**
 * The pulse rate for a track: locked to its tempo when it has a steady beat,
 * the mode's own rate otherwise.
 */
export function pulseRate(targetHz, track, minStrength = 0.2) {
  if (track && track.bpm > 0 && (track.beat_strength ?? 1) >= minStrength) {
    return tempoLocked(targetHz, track.bpm);
  }
  return targetHz;
}

/**
 * The first moment at or after `now` (context seconds) when the pulse should
 * be at full level: a beat of the track. `mediaTime` is where the track is at
 * `now`. Without a known beat the pulse simply starts now.
 */
export function nextBeatTime(now, mediaTime, track) {
  if (!track || !(track.bpm > 0)) return now;
  const period = 60 / track.bpm;
  const offset = track.beat_offset ?? 0;
  const beatsSince = (mediaTime - offset) / period;
  const next = Math.ceil(beatsSince - 1e-9);
  return now + (offset + next * period - mediaTime);
}

export function dbToGain(db) {
  return 10 ** (db / 20);
}

/**
 * The binaural slider: 0 is silent, otherwise 0..1 maps linearly in decibels
 * onto -40..-18 dB per ear.
 */
export function binauralGain(level) {
  if (!(level > 0)) return 0;
  const clamped = Math.min(1, level);
  return dbToGain(-40 + 22 * clamped);
}

/** Each ear's frequency for a beat around a carrier: left low, right high. */
export function earFrequencies(carrier, beat) {
  return [carrier - beat / 2, carrier + beat / 2];
}
