"""Skeleton features for rule-based gestures (Milestone 2).

Turns a `[T, J, 4]` skeleton window into a small bag of interpretable, body-relative
features that `models/motion_rules.py` maps to broad reaction categories. Pure / no I/O.

All geometry is computed in the torso-normalized frame (origin = hip midpoint, scale =
shoulder width), so thresholds are in "shoulder-width units" and roughly person-size
invariant. Missing / low-visibility joints become NaN and are ignored by the nan-aware
aggregates — never a crash (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from capture.hands_mediapipe import NUM_HAND_LANDMARKS
from capture.pose_mediapipe import J
from capture.skeleton import LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET
from features.skeleton_buffer import normalize_window

# Finger -> (mcp/base, pip/mid, tip) indices within a 21-point hand.
_FINGERS: dict[str, tuple[int, int, int]] = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
_HAND_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_PINKY_TIP = 20


@dataclass
class HandFeatures:
    present: float = 0.0                     # 0..1 fraction of frames the hand is visible
    extended: dict[str, float] = field(default_factory=dict)  # finger -> fraction extended
    thumb_up: float = 0.0                    # fraction of frames thumb tip is above index MCP
    spread: float = 0.0                      # fingertip spread / hand size (median)

    def is_extended(self, finger: str, thr: float = 0.5) -> bool:
        return self.extended.get(finger, 0.0) >= thr

    @property
    def extended_count(self) -> int:
        return sum(1 for v in self.extended.values() if v >= 0.5)


@dataclass
class SkeletonFeatures:
    valid_ratio: float = 0.0
    # static (median over the window)
    wrist_above_shoulder_l: float = np.nan
    wrist_above_shoulder_r: float = np.nan
    elbow_angle_l: float = np.nan
    elbow_angle_r: float = np.nan
    wrist_lateral_l: float = np.nan
    wrist_lateral_r: float = np.nan
    inter_wrist_dx: float = np.nan
    wrists_crossed: float = 0.0              # fraction of frames the wrists are crossed
    wrist_to_nose_l: float = np.nan
    wrist_to_nose_r: float = np.nan
    hand_face_height_l: float = np.nan       # nose_y - wrist_y (>0 => wrist above nose)
    hand_face_height_r: float = np.nan
    # motion (per-frame, torso units)
    wrist_speed_l: float = 0.0
    wrist_speed_r: float = 0.0
    wrist_x_oscillations_l: float = 0.0
    wrist_x_oscillations_r: float = 0.0
    inter_wrist_convergences: float = 0.0
    # fingers
    left_hand: HandFeatures = field(default_factory=HandFeatures)
    right_hand: HandFeatures = field(default_factory=HandFeatures)

    def best_hand(self) -> HandFeatures:
        return self.left_hand if self.left_hand.present >= self.right_hand.present else self.right_hand


# --- geometry helpers ------------------------------------------------------
def _xy(window: np.ndarray, j: int, thr: float) -> np.ndarray:
    """[T,2] x,y for joint j, NaN where below visibility threshold."""
    pts = window[:, j, :2].astype(np.float64).copy()
    pts[window[:, j, 3] < thr] = np.nan
    return pts


def _nanmedian(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-frame angle (deg) at vertex b for points a-b-c; NaN where undefined."""
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba, axis=1)
    nbc = np.linalg.norm(bc, axis=1)
    denom = nba * nbc
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", ba, bc) / denom
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[denom == 0] = np.nan
    return ang


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def _count_sign_changes(dx: np.ndarray, *, mask: np.ndarray, min_mag: float) -> int:
    """Sign changes of dx over masked, significant steps (used for wave oscillation)."""
    sig = np.where(np.isfinite(dx) & mask & (np.abs(dx) > min_mag), np.sign(dx), 0.0)
    sig = sig[sig != 0.0]
    return int(np.sum(sig[1:] * sig[:-1] < 0)) if sig.size > 1 else 0


def _hand_features(window: np.ndarray, offset: int, thr: float) -> HandFeatures:
    if window.shape[1] < offset + NUM_HAND_LANDMARKS:
        return HandFeatures(extended={k: 0.0 for k in _FINGERS})

    vis = window[:, offset:offset + NUM_HAND_LANDMARKS, 3]
    present_mask = np.median(vis, axis=1) >= thr  # per-frame hand presence
    n = int(present_mask.sum())
    hf = HandFeatures(present=float(np.mean(present_mask)), extended={k: 0.0 for k in _FINGERS})
    if n == 0:
        return hf

    def hxy(local_idx: int) -> np.ndarray:
        return _xy(window, offset + local_idx, thr)

    wrist = hxy(_HAND_WRIST)
    extended_frac: dict[str, float] = {}
    for name, (mcp, pip, tip) in _FINGERS.items():
        d_tip = _dist(hxy(tip), wrist)
        d_pip = _dist(hxy(pip if name != "thumb" else mcp), wrist)
        ext = (d_tip > d_pip * 1.05) & present_mask
        valid = np.isfinite(d_tip) & np.isfinite(d_pip) & present_mask
        extended_frac[name] = float(ext.sum() / max(1, int(valid.sum())))
    hf.extended = extended_frac

    thumb_tip, index_mcp = hxy(_FINGERS["thumb"][2]), hxy(_INDEX_MCP)
    up = (thumb_tip[:, 1] < index_mcp[:, 1]) & present_mask  # y is image-down
    hf.thumb_up = float(up.sum() / max(1, n))

    hand_size = _dist(hxy(_MIDDLE_MCP), wrist)
    spread = _dist(hxy(_FINGERS["index"][2]), hxy(_PINKY_TIP)) / np.where(hand_size > 1e-6, hand_size, np.nan)
    hf.spread = _nanmedian(spread)
    return hf


# --- main entry point ------------------------------------------------------
def compute_features(
    window: np.ndarray,
    *,
    visibility_threshold: float = 0.5,
    assume_normalized: bool = False,
) -> SkeletonFeatures:
    """Compute gesture features from a `[T, J, 4]` window."""
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 3 or window.shape[0] == 0:
        return SkeletonFeatures()
    if not assume_normalized:
        window = normalize_window(window)
    thr = visibility_threshold

    nose = _xy(window, J["nose"], thr)
    ls, rs = _xy(window, J["left_shoulder"], thr), _xy(window, J["right_shoulder"], thr)
    le, re = _xy(window, J["left_elbow"], thr), _xy(window, J["right_elbow"], thr)
    lw, rw = _xy(window, J["left_wrist"], thr), _xy(window, J["right_wrist"], thr)
    sho = (ls + rs) / 2.0

    valid = np.isfinite(ls).all(axis=1) & np.isfinite(rs).all(axis=1)
    f = SkeletonFeatures(valid_ratio=float(np.mean(valid)))

    # Static
    f.wrist_above_shoulder_l = _nanmedian(sho[:, 1] - lw[:, 1])
    f.wrist_above_shoulder_r = _nanmedian(sho[:, 1] - rw[:, 1])
    f.elbow_angle_l = _nanmedian(_angle(ls, le, lw))
    f.elbow_angle_r = _nanmedian(_angle(rs, re, rw))
    f.wrist_lateral_l = _nanmedian(np.abs(lw[:, 0] - sho[:, 0]))
    f.wrist_lateral_r = _nanmedian(np.abs(rw[:, 0] - sho[:, 0]))
    f.inter_wrist_dx = _nanmedian(np.abs(lw[:, 0] - rw[:, 0]))
    f.wrist_to_nose_l = _nanmedian(_dist(lw, nose))
    f.wrist_to_nose_r = _nanmedian(_dist(rw, nose))
    f.hand_face_height_l = _nanmedian(nose[:, 1] - lw[:, 1])
    f.hand_face_height_r = _nanmedian(nose[:, 1] - rw[:, 1])

    # wrists crossed: wrist L/R x-order reversed vs shoulders, near chest, close together
    sho_order = np.sign(ls[:, 0] - rs[:, 0])
    wrist_order = np.sign(lw[:, 0] - rw[:, 0])
    low = ((sho[:, 1] - lw[:, 1]) < 0.15) & ((sho[:, 1] - rw[:, 1]) < 0.15)  # not raised
    crossed = (sho_order * wrist_order < 0) & (np.abs(lw[:, 0] - rw[:, 0]) < 0.7) & low & valid
    f.wrists_crossed = float(np.mean(np.where(valid, crossed, 0.0)))

    # Motion
    def speed(w: np.ndarray) -> float:
        d = np.linalg.norm(np.diff(w, axis=0), axis=1)
        return float(np.nanmean(d)) if np.isfinite(d).any() else 0.0

    f.wrist_speed_l, f.wrist_speed_r = speed(lw), speed(rw)
    raised_l = (sho[:, 1] - lw[:, 1]) > -0.2
    raised_r = (sho[:, 1] - rw[:, 1]) > -0.2
    f.wrist_x_oscillations_l = _count_sign_changes(np.diff(lw[:, 0]), mask=raised_l[1:], min_mag=0.03)
    f.wrist_x_oscillations_r = _count_sign_changes(np.diff(rw[:, 0]), mask=raised_r[1:], min_mag=0.03)

    inter = _dist(lw, rw)
    if np.isfinite(inter).sum() >= 3:
        is_min = np.zeros_like(inter, dtype=bool)
        is_min[1:-1] = (inter[1:-1] < inter[:-2]) & (inter[1:-1] < inter[2:]) & (inter[1:-1] < 0.5)
        f.inter_wrist_convergences = float(np.sum(is_min))

    # Fingers
    f.left_hand = _hand_features(window, LEFT_HAND_OFFSET, thr)
    f.right_hand = _hand_features(window, RIGHT_HAND_OFFSET, thr)
    return f
