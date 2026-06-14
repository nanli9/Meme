#!/usr/bin/env bash
# Download the MediaPipe Pose Landmarker model(s) used by Milestone 1.
# Default: "full" (higher accuracy). Pass "lite" or "heavy" to fetch another variant.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/models
VARIANT="${1:-full}"
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${VARIANT}/float16/latest/pose_landmarker_${VARIANT}.task"
echo "Downloading pose_landmarker_${VARIANT}.task …"
curl -fsSL "$URL" -o "data/models/pose_landmarker_${VARIANT}.task"
echo "Saved to data/models/pose_landmarker_${VARIANT}.task"
