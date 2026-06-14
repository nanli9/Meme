"""MediaPipe Hand Landmarker wrapper (fingers add-on).

NOTE: This goes beyond CLAUDE.md's documented 13-joint body skeleton. It was added on
explicit request to get finger detail. Body pose remains the primary signal; hands are
appended after the body joints (see ``capture/skeleton.py``).

Each hand has 21 landmarks. MediaPipe hand landmarks carry no per-point ``visibility``
field, so we use the hand's detection confidence as the visibility proxy for all 21
points — keeping the uniform ``[x, y, z, visibility]`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 21 hand landmarks, in MediaPipe's canonical order.
HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
NUM_HAND_LANDMARKS = len(HAND_LANDMARK_NAMES)  # 21

# Bones within one hand (indices into the 21-landmark array).
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm base
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DEFAULT_HAND_MODEL_PATH = _MODELS_DIR / "hand_landmarker.task"


def _hand_to_array(landmarks, confidence: float) -> np.ndarray:
    out = np.zeros((NUM_HAND_LANDMARKS, 4), dtype=np.float32)
    for i in range(min(NUM_HAND_LANDMARKS, len(landmarks))):
        lm = landmarks[i]
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
        out[i, 2] = float(lm.z)
        out[i, 3] = confidence
    return out


def hands_from_result(result) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Split a HandLandmarkerResult into ``(left_hand, right_hand)`` arrays of `[21, 4]`.

    Either side is ``None`` when that hand was not detected. Handedness is taken from
    MediaPipe's classification (image-space; may appear swapped under selfie mirroring).
    """
    left = right = None
    if result is None or not getattr(result, "hand_landmarks", None):
        return None, None

    for i, hand in enumerate(result.hand_landmarks):
        label, score = "Right", 1.0
        if getattr(result, "handedness", None) and i < len(result.handedness) and result.handedness[i]:
            cat = result.handedness[i][0]
            label = cat.category_name
            score = float(cat.score)
        arr = _hand_to_array(hand, score)
        if label == "Left":
            left = arr
        else:
            right = arr
    return left, right


class HandEstimator:
    """Per-frame hand extraction (up to 2 hands) in VIDEO running mode."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_HAND_MODEL_PATH,
        *,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. Download it with:\n"
                "  bash scripts/download_model.sh hands"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return hands_from_result(result)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
