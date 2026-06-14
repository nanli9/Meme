"""Tests for the body+hands full skeleton (fingers add-on)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.hands_mediapipe import NUM_HAND_LANDMARKS, hands_from_result  # noqa: E402
from capture.pose_mediapipe import J as BODY_J  # noqa: E402
from capture.skeleton import (  # noqa: E402
    FULL_JOINT_NAMES,
    FULL_SKELETON_EDGES,
    LEFT_HAND_OFFSET,
    NUM_FULL_JOINTS,
    RIGHT_HAND_OFFSET,
    assemble_skeleton,
    draw_full_skeleton,
)
from features.skeleton_buffer import SkeletonBuffer, normalize_window  # noqa: E402


# --- fakes mimicking mediapipe result objects ------------------------------
class FakeLM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeCat:
    def __init__(self, name, score=0.9):
        self.category_name, self.score = name, score


class FakeHandResult:
    def __init__(self, hand_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness


def make_hand(val):
    return [FakeLM(val, val, val) for _ in range(NUM_HAND_LANDMARKS)]


# --- structure -------------------------------------------------------------
def test_full_skeleton_sizes():
    assert NUM_FULL_JOINTS == 55
    assert len(FULL_JOINT_NAMES) == 55
    assert LEFT_HAND_OFFSET == 13 and RIGHT_HAND_OFFSET == 34


def test_full_edges_in_range():
    for a, b in FULL_SKELETON_EDGES:
        assert 0 <= a < NUM_FULL_JOINTS and 0 <= b < NUM_FULL_JOINTS


# --- hands parsing ---------------------------------------------------------
def test_hands_split_by_handedness():
    res = FakeHandResult(
        hand_landmarks=[make_hand(0.1), make_hand(0.9)],
        handedness=[[FakeCat("Left")], [FakeCat("Right")]],
    )
    left, right = hands_from_result(res)
    assert left is not None and right is not None
    assert left.shape == (NUM_HAND_LANDMARKS, 4)
    assert np.allclose(left[:, 0], 0.1)
    assert np.allclose(right[:, 0], 0.9)
    assert np.allclose(left[:, 3], 0.9)  # visibility = handedness score


def test_hands_none_when_empty():
    assert hands_from_result(None) == (None, None)
    assert hands_from_result(FakeHandResult([], [])) == (None, None)


# --- assembly --------------------------------------------------------------
def test_assemble_places_components():
    body = np.full((13, 4), 0.5, np.float32)
    left = np.full((NUM_HAND_LANDMARKS, 4), 0.2, np.float32)
    skel = assemble_skeleton(body, left, None)
    assert skel.shape == (NUM_FULL_JOINTS, 4)
    assert np.allclose(skel[:13], 0.5)
    assert np.allclose(skel[LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + NUM_HAND_LANDMARKS], 0.2)
    assert np.allclose(skel[RIGHT_HAND_OFFSET:], 0.0)  # missing right hand stays zero


def test_assemble_all_missing_is_zeros():
    skel = assemble_skeleton(None, None, None)
    assert skel.shape == (NUM_FULL_JOINTS, 4)
    assert np.allclose(skel, 0.0)


# --- drawing tolerance -----------------------------------------------------
def test_draw_full_skeleton_handles_missing_and_old_shapes():
    frame = np.zeros((480, 640, 3), np.uint8)
    assert draw_full_skeleton(frame, None) is frame
    draw_full_skeleton(frame, np.zeros((NUM_FULL_JOINTS, 4), np.float32))  # all missing
    draw_full_skeleton(frame, np.full((13, 4), 0.5, np.float32))  # legacy 13-joint window
    assert frame.shape == (480, 640, 3)


# --- buffer + normalization with 55 joints ---------------------------------
def _full_frame(vis=1.0):
    f = np.zeros((NUM_FULL_JOINTS, 4), np.float32)
    f[:, 3] = vis
    f[BODY_J["left_hip"]] = [0.4, 0.6, 0.0, vis]
    f[BODY_J["right_hip"]] = [0.6, 0.6, 0.0, vis]
    f[BODY_J["left_shoulder"]] = [0.4, 0.3, 0.0, vis]
    f[BODY_J["right_shoulder"]] = [0.6, 0.3, 0.0, vis]
    f[LEFT_HAND_OFFSET] = [0.5, 0.6, 0.0, vis]  # a hand point on the hip midpoint
    return f


def test_buffer_roundtrip_55_joints(tmp_path):
    buf = SkeletonBuffer(maxlen=60, session_dir=tmp_path,
                         num_joints=NUM_FULL_JOINTS, joint_names=FULL_JOINT_NAMES)
    for _ in range(60):
        buf.append(_full_frame())
    path = buf.save_window(fps=30.0, normalize=True)
    data = np.load(path, allow_pickle=True)
    assert data["landmarks"].shape == (60, NUM_FULL_JOINTS, 4)
    assert len(list(data["joint_names"])) == NUM_FULL_JOINTS


def test_normalize_hands_in_torso_frame():
    win = _full_frame()[None, ...]
    norm = normalize_window(win)
    # The hand point sat on the hip midpoint -> maps to ~origin after normalization.
    assert np.allclose(norm[0, LEFT_HAND_OFFSET, :2], 0.0, atol=1e-5)
    # Visibility preserved across all 55 joints.
    assert np.allclose(norm[0, :, 3], 1.0)
