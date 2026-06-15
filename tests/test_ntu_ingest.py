"""Tests for NTU RGB+D skeleton ingestion (Milestone 2 eval harness).

Validates the .skeleton parser + Kinect-25 -> our-13 remap on a synthetic file, so the
NTU code path is covered without the access-gated dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import J  # noqa: E402
from evaluation.gesture_eval import NTU_CODE_TO_GESTURE  # noqa: E402
from scripts.eval_ntu import KINECT_INDEX, action_code, parse_skeleton_file  # noqa: E402


def _joint_line(k: int) -> str:
    # x y z depthX depthY colorX colorY oW oX oY oZ trackingState  (12 tokens)
    return f"0 0 0 0 0 {k * 10.0} {k * 5.0} 0 0 0 0 2"


def _synthetic_skeleton(n_frames: int = 2) -> str:
    lines = [str(n_frames)]
    for _ in range(n_frames):
        lines.append("1")                       # one body
        lines.append("0 0 0 0 0 0 0 0 0 0")     # body info (unused)
        lines.append("25")                      # joint count
        lines += [_joint_line(k) for k in range(25)]
    return "\n".join(lines) + "\n"


def test_action_code_and_mapping():
    assert action_code("S001C001P001R001A023.skeleton") == 23
    assert NTU_CODE_TO_GESTURE[23] == "wave"
    assert NTU_CODE_TO_GESTURE[10] == "clap"
    assert NTU_CODE_TO_GESTURE[40] == "arms_crossed"
    assert action_code("nope.skeleton") is None


def test_parse_shape_and_remap(tmp_path):
    f = tmp_path / "S001C001P001R001A023.skeleton"
    f.write_text(_synthetic_skeleton(n_frames=3))

    win = parse_skeleton_file(f, max_frames=60)
    assert win.shape == (3, 13, 4)

    # nose <- Kinect Head (index 3): colorX=30, colorY=15, normalized by 1920x1080.
    nose = J["nose"]
    assert np.isclose(win[0, nose, 0], 30.0 / 1920.0)
    assert np.isclose(win[0, nose, 1], 15.0 / 1080.0)
    assert np.isclose(win[0, nose, 3], 1.0)  # trackingState 2 -> visible

    # right_wrist <- Kinect WristRight (index 10): colorX=100.
    rw = J["right_wrist"]
    assert np.isclose(win[0, rw, 0], (KINECT_INDEX["right_wrist"] * 10.0) / 1920.0)


def test_parse_subsamples_to_max_frames(tmp_path):
    f = tmp_path / "S001C001P001R001A010.skeleton"
    f.write_text(_synthetic_skeleton(n_frames=120))
    win = parse_skeleton_file(f, max_frames=60)
    assert win.shape == (60, 13, 4)


def test_parse_empty_frames(tmp_path):
    # n_frames=1 but body_count=0 -> no usable frames, returns empty (no crash).
    f = tmp_path / "S001C001P001R001A010.skeleton"
    f.write_text("1\n0\n")
    win = parse_skeleton_file(f)
    assert win.shape == (0, 13, 4)
