#!/usr/bin/env bash
# Render the full preset set into ./renders, converting to FLAC if ffmpeg
# is available. Nothing here is required — it just reproduces the session
# set in one go.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p renders

R="uv run --quiet reference"

echo "plain tones..."
$R/binaural_e_violet.py --beat theta    --minutes 20 --out renders/theta_4Hz_20min.wav
$R/binaural_e_violet.py --beat alpha    --minutes 15 --out renders/alpha_10Hz_15min.wav
$R/binaural_e_violet.py --beat delta    --minutes 30 --out renders/delta_2Hz_30min.wav
$R/binaural_e_violet.py --beat schumann --minutes 20 --out renders/schumann_7.83Hz_20min.wav

echo "layered..."
$R/binaural_layered.py --beat theta --minutes 15 --out renders/theta_layered_15min.wav

echo "ocean + chords..."
$R/ocean_chords_e.py --beat theta --minutes 30 --seed 5 --out renders/ocean_chords_theta_30min.wav

if command -v ffmpeg >/dev/null 2>&1; then
  echo "converting to flac..."
  for f in renders/*.wav; do
    ffmpeg -y -loglevel error -i "$f" -c:a flac -compression_level 8 "${f%.wav}.flac"
    rm "$f"
  done
else
  echo "ffmpeg not found — leaving WAVs in place."
fi

echo "done: $(ls renders | wc -l) files in ./renders"
