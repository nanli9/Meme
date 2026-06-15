"""Tests for rule-based gestures (Milestone 2).

Synthesizes torso-normalized windows posing each category and asserts the right `top`.
All coords are already in the normalized frame, so `assume_normalized=True`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import J  # noqa: E402
from capture.skeleton import LEFT_HAND_OFFSET, NUM_FULL_JOINTS, RIGHT_HAND_OFFSET  # noqa: E402
from features.skeleton_features import compute_features  # noqa: E402
from models.motion_rules import GestureEstimate, GestureStabilizer, estimate_gesture  # noqa: E402

# Full landmark chains (base..tip) per finger, matching skeleton_features._FINGER_CHAINS.
_FINGER_CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
                  "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _set(win, idx, xy):
    arr = np.asarray(xy, dtype=float)
    if arr.ndim == 1:
        win[:, idx, 0], win[:, idx, 1] = arr[0], arr[1]
    else:
        win[:, idx, 0], win[:, idx, 1] = arr[:, 0], arr[:, 1]
    win[:, idx, 3] = 1.0


def base(T=12):
    """A window with shoulders/nose/hips set; arms unset (invisible)."""
    win = np.zeros((T, NUM_FULL_JOINTS, 4), dtype=np.float32)
    _set(win, J["left_shoulder"], (0.5, -0.7))
    _set(win, J["right_shoulder"], (-0.5, -0.7))
    _set(win, J["nose"], (0.0, -1.3))
    _set(win, J["left_hip"], (0.4, 0.0))
    _set(win, J["right_hip"], (-0.4, 0.0))
    return win


def arms_down(win):
    _set(win, J["left_elbow"], (0.55, -0.2)); _set(win, J["right_elbow"], (-0.55, -0.2))
    _set(win, J["left_wrist"], (0.5, 0.4)); _set(win, J["right_wrist"], (-0.5, 0.4))
    return win


def set_hand(win, offset, extended: set, *, wx=0.5, wy=-0.3):
    """Pose a 21-pt hand. Each finger's 4-joint chain is laid straight (extended) or
    folded back toward the palm (curled), matching the chain-straightness extension
    metric in skeleton_features."""
    _set(win, offset + 0, (wx, wy))  # wrist
    cols = {"thumb": -0.2, "index": -0.05, "middle": 0.05, "ring": 0.15, "pinky": 0.25}
    for fname, chain in _FINGER_CHAINS.items():
        col = cols[fname]
        if fname in extended:
            ys = [wy - 0.1, wy - 0.2, wy - 0.3, wy - 0.4]   # straight -> span/length ~ 1
        else:
            ys = [wy - 0.1, wy - 0.18, wy - 0.12, wy - 0.04]  # folded back -> small span
        for local, yy in zip(chain, ys):
            _set(win, offset + local, (wx + col, yy))


def top(win):
    # Test the RULE path explicitly (use_model=False) so results are deterministic whether
    # or not a trained hand-classifier model file is present locally.
    return estimate_gesture(win, assume_normalized=True, use_model=False).top


# --- static arm poses ------------------------------------------------------
def test_neutral():
    assert top(arms_down(base())) == "neutral"


def test_hype():
    w = base()
    _set(w, J["left_wrist"], (0.7, -1.5)); _set(w, J["right_wrist"], (-0.7, -1.5))
    _set(w, J["left_elbow"], (0.6, -1.1)); _set(w, J["right_elbow"], (-0.6, -1.1))
    assert top(w) == "hype"


def test_arms_wide():
    w = base()
    _set(w, J["left_elbow"], (1.1, -0.7)); _set(w, J["right_elbow"], (-1.1, -0.7))
    _set(w, J["left_wrist"], (1.7, -0.7)); _set(w, J["right_wrist"], (-1.7, -0.7))
    assert top(w) == "arms_wide"


def test_shrug():
    w = base()
    _set(w, J["left_elbow"], (0.7, -0.4)); _set(w, J["right_elbow"], (-0.7, -0.4))
    _set(w, J["left_wrist"], (0.85, -0.75)); _set(w, J["right_wrist"], (-0.85, -0.75))
    assert top(w) == "shrug"


def test_arms_crossed():
    w = base()
    _set(w, J["left_elbow"], (0.4, -0.4)); _set(w, J["right_elbow"], (-0.4, -0.4))
    _set(w, J["left_wrist"], (-0.2, -0.3)); _set(w, J["right_wrist"], (0.2, -0.3))
    assert top(w) == "arms_crossed"


# --- hand-to-face ----------------------------------------------------------
def test_facepalm():
    # Still hand at/above nose level -> facepalm.
    w = arms_down(base())
    _set(w, J["right_wrist"], (0.1, -1.4)); _set(w, J["right_elbow"], (-0.1, -0.8))
    assert top(w) == "facepalm"


def test_thinking():
    # Still hand at the chin (below nose) -> thinking.
    w = arms_down(base())
    _set(w, J["right_wrist"], (0.15, -0.95)); _set(w, J["right_elbow"], (-0.1, -0.6))
    assert top(w) == "thinking"


def test_point_at_own_head_is_thinking():
    # Index finger up AT your own head (fingertip near the nose) reads as thinking, not
    # pointing — only `pointing` is face-aware (surgical), other gestures are unaffected.
    w = arms_down(base())
    set_hand(w, RIGHT_HAND_OFFSET, {"index"}, wx=0.05, wy=-1.0)
    assert top(w) == "thinking"


# --- motion ----------------------------------------------------------------
def test_wave():
    T = 16
    w = arms_down(base(T))
    rx = np.array([0.85 if t % 2 else 0.35 for t in range(T)])
    _set(w, J["right_wrist"], np.stack([rx, np.full(T, -1.2)], axis=1))
    assert top(w) == "wave"


def test_wave_near_face_is_wave_not_thinking():
    # A hand oscillating left-right right next to the face must read as wave, not thinking
    # (thinking requires a still hand). Regression test for the motion gate.
    T = 16
    w = arms_down(base(T))
    rx = np.array([0.3 if t % 2 else -0.1 for t in range(T)])  # oscillates near the nose (x~0)
    _set(w, J["right_wrist"], np.stack([rx, np.full(T, -1.3)], axis=1))  # nose height
    assert top(w) == "wave"


def test_clap():
    T = 12
    w = base(T)
    lw = np.zeros((T, 2)); rw = np.zeros((T, 2))
    for t in range(T):
        if t % 2 == 0:
            lw[t], rw[t] = (0.0, -0.3), (0.0, -0.3)
        else:
            lw[t], rw[t] = (-0.6, -0.3), (0.6, -0.3)
    _set(w, J["left_wrist"], lw); _set(w, J["right_wrist"], rw)
    assert top(w) == "clap"


# --- finger gestures -------------------------------------------------------
def test_open_palm():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"thumb", "index", "middle", "ring", "pinky"})
    assert top(w) == "open_palm"


def test_pointing():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"index"})
    assert top(w) == "pointing"


def test_thumbs_up():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"thumb"})
    assert top(w) == "thumbs_up"


def test_fingers_reported_explicitly():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"index", "middle"})
    est = estimate_gesture(w, assume_normalized=True, use_model=False)
    assert "index" in est.fingers and "middle" in est.fingers
    assert "ring" not in est.fingers and "pinky" not in est.fingers


def test_middle_finger():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"middle"})
    assert top(w) == "middle_finger"


def test_pinky():
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"pinky"})
    assert top(w) == "pinky"


def test_peace():
    # Index + middle extended -> peace, and crucially NOT pointing (the extended middle
    # vetoes pointing via the max() suppression).
    w = arms_down(base())
    set_hand(w, LEFT_HAND_OFFSET, {"index", "middle"})
    t = top(w)
    assert t == "peace" and t != "pointing"


# --- robustness ------------------------------------------------------------
def test_missing_landmarks_no_crash():
    w = np.zeros((10, NUM_FULL_JOINTS, 4), dtype=np.float32)  # everything invisible
    est = estimate_gesture(w, assume_normalized=True)
    assert isinstance(est, GestureEstimate)
    assert est.top == "neutral"
    assert est.valid_ratio == 0.0


def test_scores_present_for_all_gestures():
    est = estimate_gesture(arms_down(base()), assume_normalized=True)
    from models.motion_rules import GESTURES
    assert set(GESTURES).issubset(est.scores.keys())


# --- stabilizer ------------------------------------------------------------
def test_stabilizer_debounces_and_throttles():
    st = GestureStabilizer(min_interval=1.0, persist=2)
    hype = GestureEstimate(top="hype", confidence=0.9)
    # First sighting shouldn't switch (needs to persist).
    assert st.update(hype, now=0.0) == "neutral"
    # Persisted but throttled (interval not elapsed since -inf? -inf elapsed, so switches).
    assert st.update(hype, now=0.05) == "hype"
    # A one-off blip doesn't immediately flip the label.
    assert st.update(GestureEstimate(top="wave", confidence=0.9), now=0.1) == "hype"
