"""Tests for pose landmark parsing (Milestone 1).

These exercise the pure logic (`landmarks_from_result`, drawing, counting) without a
camera or GPU, using a tiny fake PoseLandmarkerResult.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import (  # noqa: E402
    MP_LANDMARK_INDICES,
    NUM_CHANNELS,
    NUM_JOINTS,
    SKELETON_EDGES,
    count_visible,
    draw_skeleton,
    landmarks_from_result,
)


class FakeLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class FakeResult:
    """Mimics mediapipe's PoseLandmarkerResult: .pose_landmarks is a list of poses."""

    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks


def make_full_pose(visibility=1.0):
    # BlazePose has 33 landmarks; give each a unique coord so mapping is checkable.
    return [FakeLandmark(x=i / 33.0, y=i / 33.0, z=i / 100.0, visibility=visibility) for i in range(33)]


def test_shape_and_dtype():
    res = FakeResult([make_full_pose()])
    out = landmarks_from_result(res)
    assert out is not None
    assert out.shape == (NUM_JOINTS, NUM_CHANNELS)
    assert out.dtype == np.float32


def test_no_detection_returns_none():
    assert landmarks_from_result(None) is None
    assert landmarks_from_result(FakeResult([])) is None


def test_correct_landmark_mapping():
    res = FakeResult([make_full_pose()])
    out = landmarks_from_result(res)
    # Joint j must pull from BlazePose index MP_LANDMARK_INDICES[j].
    for j, mp_idx in enumerate(MP_LANDMARK_INDICES):
        assert np.isclose(out[j, 0], mp_idx / 33.0)
        assert np.isclose(out[j, 1], mp_idx / 33.0)


def test_low_visibility_landmarks_are_zeroed():
    pose = make_full_pose(visibility=1.0)
    # Knock the nose (BlazePose idx 0) below threshold.
    pose[0] = FakeLandmark(x=0.9, y=0.9, z=0.1, visibility=0.05)
    out = landmarks_from_result(FakeResult([pose]), visibility_threshold=0.3)
    nose = 0  # first canonical joint
    assert np.allclose(out[nose], 0.0)  # treated as missing, not crashing


def test_count_visible():
    res = FakeResult([make_full_pose(visibility=0.8)])
    out = landmarks_from_result(res)
    assert count_visible(out, 0.5) == NUM_JOINTS
    assert count_visible(None, 0.5) == 0


def test_draw_skeleton_does_not_crash_on_missing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # All missing -> should be a no-op, no exception.
    empty = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    out = draw_skeleton(frame, empty)
    assert out.shape == frame.shape
    # None landmarks -> also safe.
    assert draw_skeleton(frame, None) is frame


def test_skeleton_edges_in_range():
    for a, b in SKELETON_EDGES:
        assert 0 <= a < NUM_JOINTS and 0 <= b < NUM_JOINTS
