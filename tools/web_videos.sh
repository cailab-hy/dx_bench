#!/usr/bin/env bash
# Web copies of one episode's videos: <run dir with observer/head/right_wrist/left_wrist.mp4> <out dir> <name>
#   -> <name>_observer.mp4 (1280 x 720) and <name>_robotcams.mp4 (head | right | left, 1200 x 300), H.264, 15 fps
set -eu
src=$1; out=$2; name=$3
mkdir -p "$out"
ffmpeg -loglevel error -y -i "$src/observer.mp4" -vf scale=1280:-2 -c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p \
  -movflags +faststart "$out/${name}_observer.mp4"
ffmpeg -loglevel error -y -i "$src/head.mp4" -i "$src/right_wrist.mp4" -i "$src/left_wrist.mp4" \
  -filter_complex "[0]scale=400:300[a];[1]scale=400:300[b];[2]scale=400:300[c];[a][b][c]hstack=3" \
  -c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p -movflags +faststart "$out/${name}_robotcams.mp4"
echo "[web_videos] $name"
