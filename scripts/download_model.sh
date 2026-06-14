#!/usr/bin/env bash
# Download the MediaPipe models used by the skeleton debugger.
#   (no args)  -> pose "full" + hand landmarker (the defaults)
#   lite|heavy -> that pose variant
#   hands      -> hand landmarker only
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/models

fetch_pose () {
  local v="$1"
  echo "Downloading pose_landmarker_${v}.task …"
  curl -fsSL "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/latest/pose_landmarker_${v}.task" \
    -o "data/models/pose_landmarker_${v}.task"
}
fetch_hands () {
  echo "Downloading hand_landmarker.task …"
  curl -fsSL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    -o "data/models/hand_landmarker.task"
}

case "${1:-all}" in
  all)         fetch_pose full; fetch_hands ;;
  lite|heavy)  fetch_pose "$1" ;;
  full)        fetch_pose full ;;
  hands)       fetch_hands ;;
  *) echo "usage: $0 [all|full|lite|heavy|hands]" >&2; exit 1 ;;
esac
echo "Done. Models in data/models/"
