#!/usr/bin/env bash
# Download the MediaPipe Pose Landmarker model used by Milestone 1.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/models
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Downloading pose_landmarker_lite.task …"
curl -fsSL "$URL" -o data/models/pose_landmarker_lite.task
echo "Saved to data/models/pose_landmarker_lite.task"
