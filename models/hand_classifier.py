"""Learned hand-gesture classifier (Milestone 6, brought forward for hands).

A small MLP over normalized MediaPipe hand landmarks, trained on HaGRID
(`scripts/train_hand_classifier.py`). Replaces the brittle finger *rules* for the
classes HaGRID covers; body/motion gestures stay rule-based.

Feature vector (per hand frame): the 21 landmarks made invariant to position, scale,
in-plane rotation and handedness, then flattened to 63 dims. This is what lets the model
generalize across where/how the hand is held — including using the relative z depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from capture.hands_mediapipe import NUM_HAND_LANDMARKS

# Classes the model predicts. "other" absorbs every HaGRID class that isn't one of ours
# (fist, ok, rock, ...) and maps to "no finger gesture".
GESTURE_CLASSES = ("thumbs_up", "pointing", "open_palm", "peace", "other")
# Map a predicted class to a score key in motion_rules (None => contributes nothing).
CLASS_TO_GESTURE = {
    "thumbs_up": "thumbs_up", "pointing": "pointing",
    "open_palm": "open_palm", "peace": "peace", "other": None,
}

FEATURE_DIM = NUM_HAND_LANDMARKS * 3  # 63
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "hand_gesture_clf.joblib"

_WRIST, _MIDDLE_MCP, _THUMB_TIP, _PINKY_MCP = 0, 9, 4, 17


def extract_hand_features(hand: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a `[21, >=3]` hand into a 63-dim feature vector, or None if degenerate.

    Steps: translate to wrist origin -> scale by wrist->middle-MCP length -> rotate so the
    middle finger points "up" -> mirror so the thumb is always on the same side (handedness
    invariance). The z channel is kept (relative depth) and scaled the same way.
    """
    pts = np.asarray(hand, dtype=np.float64)[:, :3].copy()
    if pts.shape[0] < NUM_HAND_LANDMARKS or not np.isfinite(pts).all():
        return None

    pts -= pts[_WRIST]
    scale = np.linalg.norm(pts[_MIDDLE_MCP])
    if scale < 1e-6:
        return None
    pts /= scale

    # Rotate in the image plane so wrist->middle_mcp points up (-y is up in image coords).
    vx, vy = pts[_MIDDLE_MCP, 0], pts[_MIDDLE_MCP, 1]
    ang = np.arctan2(vy, vx) + np.pi / 2.0  # angle to rotate so the vector becomes (0,-1)
    c, s = np.cos(-ang), np.sin(-ang)
    rot = pts.copy()
    rot[:, 0] = c * pts[:, 0] - s * pts[:, 1]
    rot[:, 1] = s * pts[:, 0] + c * pts[:, 1]

    # Mirror x so the thumb is always on the same side (left/right hand invariance).
    if rot[_THUMB_TIP, 0] < rot[_PINKY_MCP, 0]:
        rot[:, 0] = -rot[:, 0]

    return rot.reshape(-1).astype(np.float32)


class HandGestureClassifier:
    """Thin wrapper around the trained sklearn pipeline."""

    def __init__(self, pipeline=None, classes: tuple[str, ...] = GESTURE_CLASSES) -> None:
        self.pipeline = pipeline
        self.classes = classes

    @property
    def is_available(self) -> bool:
        return self.pipeline is not None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "HandGestureClassifier":
        path = Path(path)
        if not path.exists():
            return cls(pipeline=None)
        import joblib
        obj = joblib.load(path)
        return cls(pipeline=obj["pipeline"], classes=tuple(obj["classes"]))

    def predict_proba_features(self, feats: np.ndarray) -> dict[str, float]:
        """Mean class probabilities over a batch of `[N, 63]` feature rows."""
        if not self.is_available or feats.size == 0:
            return {c: 0.0 for c in self.classes}
        proba = self.pipeline.predict_proba(feats).mean(axis=0)
        order = list(self.pipeline.classes_)
        return {c: float(proba[order.index(c)]) if c in order else 0.0 for c in self.classes}


_CACHE: dict[str, HandGestureClassifier] = {}


def load_default() -> HandGestureClassifier:
    """Process-cached default classifier (returns an unavailable one if not trained yet)."""
    key = str(DEFAULT_MODEL_PATH)
    if key not in _CACHE:
        _CACHE[key] = HandGestureClassifier.load(DEFAULT_MODEL_PATH)
    return _CACHE[key]


def hand_window_features(window: np.ndarray, offset: int, visibility_threshold: float) -> np.ndarray:
    """Extract `[N, 63]` features from the frames of `window` where the hand at `offset` is
    visible. `window` is `[T, J, 4]` (full 55-joint skeleton)."""
    if window.shape[1] < offset + NUM_HAND_LANDMARKS:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    rows = []
    for t in range(window.shape[0]):
        hand = window[t, offset:offset + NUM_HAND_LANDMARKS]
        if np.median(hand[:, 3]) < visibility_threshold:
            continue
        feat = extract_hand_features(hand)
        if feat is not None:
            rows.append(feat)
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, FEATURE_DIM), dtype=np.float32)
