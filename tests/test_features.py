"""Tests for the rolling skeleton buffer + torso normalization (Milestone 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import J, NUM_CHANNELS, NUM_JOINTS  # noqa: E402
from features.skeleton_buffer import (  # noqa: E402
    WINDOW_SIZE,
    SkeletonBuffer,
    normalize_window,
)


def make_frame(visibility=1.0):
    f = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    f[:, 0] = 0.5  # x
    f[:, 1] = 0.5  # y
    f[:, 3] = visibility
    # Give torso anchors a known geometry.
    f[J["left_hip"]] = [0.4, 0.6, 0.0, visibility]
    f[J["right_hip"]] = [0.6, 0.6, 0.0, visibility]
    f[J["left_shoulder"]] = [0.4, 0.3, 0.0, visibility]
    f[J["right_shoulder"]] = [0.6, 0.3, 0.0, visibility]
    return f


def test_buffer_is_rolling_and_capped():
    buf = SkeletonBuffer(maxlen=WINDOW_SIZE)
    assert len(buf) == 0
    for _ in range(WINDOW_SIZE + 25):
        buf.append(make_frame())
    assert len(buf) == WINDOW_SIZE  # never exceeds maxlen
    assert buf.is_full


def test_append_none_becomes_missing_frame():
    buf = SkeletonBuffer(maxlen=5)
    buf.append(None)  # no detection must not crash
    assert len(buf) == 1
    win = buf.window()
    assert win.shape == (1, NUM_JOINTS, NUM_CHANNELS)
    assert np.allclose(win, 0.0)


def test_window_shape():
    buf = SkeletonBuffer(maxlen=WINDOW_SIZE)
    for _ in range(10):
        buf.append(make_frame())
    assert buf.window().shape == (10, NUM_JOINTS, NUM_CHANNELS)


def test_normalization_centers_on_hip_midpoint():
    win = make_frame()[None, ...]  # [1, 13, 4]
    norm = normalize_window(win)
    # Hip midpoint maps to the origin.
    mid = (norm[0, J["left_hip"], :2] + norm[0, J["right_hip"], :2]) / 2.0
    assert np.allclose(mid, 0.0, atol=1e-5)


def test_normalization_scales_by_shoulder_width():
    win = make_frame()[None, ...]
    norm = normalize_window(win)
    # Shoulder width (0.4..0.6 = 0.2) becomes unit distance after normalization.
    width = np.linalg.norm(norm[0, J["left_shoulder"], :3] - norm[0, J["right_shoulder"], :3])
    assert np.isclose(width, 1.0, atol=1e-5)


def test_normalization_preserves_visibility():
    win = make_frame(visibility=0.7)[None, ...]
    norm = normalize_window(win)
    assert np.allclose(norm[0, :, 3], 0.7)


def test_normalization_missing_torso_does_not_crash():
    # All-zero frame (everything missing) -> scale ~ 0, must pass through untouched.
    win = np.zeros((1, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    norm = normalize_window(win)
    assert np.all(np.isfinite(norm))
    assert np.allclose(norm, 0.0)


def test_save_and_reload_roundtrip(tmp_path):
    buf = SkeletonBuffer(maxlen=WINDOW_SIZE, session_dir=tmp_path)
    for _ in range(WINDOW_SIZE):
        buf.append(make_frame())
    path = buf.save_window(fps=29.5, normalize=True)
    assert path is not None and path.exists()

    data = np.load(path, allow_pickle=True)
    assert data["landmarks"].shape == (WINDOW_SIZE, NUM_JOINTS, NUM_CHANNELS)
    assert np.isclose(float(data["fps"]), 29.5)
    assert bool(data["normalized"]) is True
    assert float(data["timestamp"]) > 0


def test_save_empty_buffer_returns_none(tmp_path):
    buf = SkeletonBuffer(session_dir=tmp_path)
    assert buf.save_window() is None
