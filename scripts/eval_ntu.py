"""Evaluate body/motion gesture rules against NTU RGB+D 60 skeletons (Milestone 2).

NTU is access-gated (request at https://rose1.ntu.edu.sg/dataset/actionRecognition/), so
this can't auto-download. Point it at a folder of raw `.skeleton` files. NTU ships
Kinect-25 skeletons; we remap those to our 13 body joints (no hands → body-only mode) and
use the color-space (2D image) coordinates, which match our "y-down" convention.

    uv run python scripts/eval_ntu.py --data-dir /path/to/nturgbd_skeletons

Relevant NTU action codes are mapped in evaluation/gesture_eval.NTU_CODE_TO_GESTURE
(e.g. A023 hand waving -> wave, A010 clapping -> clap, A040 cross hands -> arms_crossed).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import J  # noqa: E402
from evaluation.gesture_eval import NTU_CODE_TO_GESTURE, Confusion  # noqa: E402
from models.motion_rules import estimate_gesture  # noqa: E402

# Our 13 body joints -> Kinect-25 joint index (BlazePose names -> Kinect names).
KINECT_INDEX = {
    "nose": 3,            # Head
    "left_shoulder": 4, "right_shoulder": 8,
    "left_elbow": 5, "right_elbow": 9,
    "left_wrist": 6, "right_wrist": 10,
    "left_hip": 12, "right_hip": 16,
    "left_knee": 13, "right_knee": 17,
    "left_ankle": 14, "right_ankle": 18,
}
_COLOR_W, _COLOR_H = 1920.0, 1080.0
_TRACK_VIS = {2: 1.0, 1: 0.5, 0: 0.0}
_ACODE = re.compile(r"A(\d{3})")


def action_code(filename: str) -> int | None:
    m = _ACODE.search(Path(filename).name)
    return int(m.group(1)) if m else None


def parse_skeleton_file(path: str | Path, *, max_frames: int = 60) -> np.ndarray:
    """Parse a raw NTU `.skeleton` file into our `[T, 13, 4]` body window (color coords)."""
    tokens = Path(path).read_text().split("\n")
    it = iter(t for t in tokens if t.strip() != "")
    n_frames = int(next(it))

    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        body_count = int(next(it))
        kept = None
        for b in range(body_count):
            next(it)  # body info line (id, hand states, lean, tracking) — unused
            n_joints = int(next(it))
            joints = np.array([[float(x) for x in next(it).split()] for _ in range(n_joints)])
            if b == 0:
                kept = joints  # first tracked body only
        if kept is not None and kept.shape[0] >= 25:
            frames.append(kept)
    if not frames:
        return np.zeros((0, 13, 4), dtype=np.float32)

    # Uniformly subsample to at most max_frames.
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
    out = np.zeros((len(idx), 13, 4), dtype=np.float32)
    for t, fi in enumerate(idx):
        kin = frames[fi]
        for name, k in KINECT_INDEX.items():
            colorx, colory, track = kin[k, 5], kin[k, 6], int(kin[k, 11])
            j = J[name]
            out[t, j, 0] = colorx / _COLOR_W
            out[t, j, 1] = colory / _COLOR_H
            out[t, j, 3] = _TRACK_VIS.get(track, 0.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate body gestures on NTU RGB+D skeletons.")
    p.add_argument("--data-dir", required=True, help="Folder of NTU .skeleton files.")
    p.add_argument("--limit-per-class", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=60)
    args = p.parse_args()

    files = sorted(Path(args.data_dir).glob("*.skeleton"))
    if not files:
        raise SystemExit(
            f"No .skeleton files in {args.data_dir}.\n"
            "Request NTU RGB+D 60 at https://rose1.ntu.edu.sg/dataset/actionRecognition/ "
            "and unzip the skeleton set there."
        )

    quota: dict[str, int] = defaultdict(lambda: args.limit_per_class)
    conf = Confusion()
    for f in files:
        code = action_code(f.name)
        gesture = NTU_CODE_TO_GESTURE.get(code) if code else None
        if gesture is None or quota[gesture] <= 0:
            continue
        window = parse_skeleton_file(f, max_frames=args.max_frames)
        if window.shape[0] == 0:
            continue
        quota[gesture] -= 1
        pred = estimate_gesture(window).top  # body-only (J=13)
        conf.add(gesture, pred)

    print("\n=== NTU RGB+D body-gesture evaluation ===")
    print(conf.report())


if __name__ == "__main__":
    main()
