#!/usr/bin/env bash
# Render the whole preset set into ./renders. Nothing here is required — it
# just reproduces the session set in one go.
#
# FLAC comes straight out of libsndfile, so there is no ffmpeg step any more.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p renders

render() { uv run --quiet violet render "$@"; }

echo "plain tones..."
render tones --beat theta    --minutes 20 --quiet --out renders/theta_4Hz_20min.flac
render tones --beat alpha    --minutes 15 --quiet --out renders/alpha_10Hz_15min.flac
render tones --beat delta    --minutes 30 --quiet --out renders/delta_2Hz_30min.flac
render tones --beat schumann --minutes 20 --quiet --out renders/schumann_20min.flac

echo "layered..."
render layered --beat theta --minutes 15 --quiet --out renders/theta_layered_15min.flac

echo "ocean + chords..."
render ocean --beat theta --minutes 30 --seed 5 --quiet --out renders/ocean_theta_30min.flac
render sleep --minutes 45 --quiet --out renders/sleep_delta_45min.flac

echo "loops..."
render ocean-loop --quiet --out renders/ocean_loop_5min.flac
render tones-loop --quiet --out renders/tones_loop_10min.flac

echo "done: $(ls renders | wc -l) files in ./renders"
