"""Live skeleton -> meme demo (Milestone 4, the first real demo).

Webcam + skeleton on the left, the top-5 recommended memes on the right. The body gesture
becomes an intent query, retrieval + the fusion ranker pick memes, and the panel refreshes
on the CLAUDE.md cadence (at most once / ~1.5s, no per-frame flicker). Holding the same
gesture rotates through different memes (diversity + recent-duplicate penalty).

    uv run python scripts/recommend_demo.py
    uv run python scripts/recommend_demo.py --device 1 --no-hands

Keys:  q / ESC = quit.  Requires the meme DB built first:
    uv run python -m meme_db.build_index --limit 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import DEFAULT_MODEL_PATH, count_visible  # noqa: E402
from capture.skeleton import (  # noqa: E402
    FULL_JOINT_NAMES,
    NUM_FULL_JOINTS,
    FullPoseEstimator,
    draw_full_skeleton,
)
from capture.webcam import Webcam, WebcamError  # noqa: E402
from features.skeleton_buffer import WINDOW_SIZE, SkeletonBuffer  # noqa: E402
from models.fusion_ranker import Recommender, gesture_to_intent  # noqa: E402
from models.motion_rules import GestureStabilizer, estimate_gesture  # noqa: E402

REFRESH = 1.5          # seconds between recommendation refreshes (CLAUDE.md: <=1/1-2s)
PANEL_W = 380
THUMB_H = 104


def _text(img, s, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _load_thumb(path: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
    if path not in cache:
        img = cv2.imread(path)
        if img is not None:
            w = int(img.shape[1] * THUMB_H / img.shape[0])
            cache[path] = cv2.resize(img, (max(1, w), THUMB_H))
        else:
            cache[path] = None
    return cache[path]


def _build_panel(results, gesture_label, height, cache) -> np.ndarray:
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    _text(panel, f"Gesture: {gesture_label}", (10, 26), (0, 255, 255), 0.7)
    if not results:
        _text(panel, "make a gesture...", (10, 60), (180, 180, 180))
        return panel
    y = 44
    for rank, r in enumerate(results, 1):
        th = _load_thumb(r.meme.image_path, cache)
        if th is not None:
            tw = min(th.shape[1], PANEL_W - 20)
            panel[y:y + THUMB_H, 10:10 + tw] = th[:, :tw]
        _text(panel, f"{rank}. {r.score:.2f} {('/'.join(r.meme.tags) or '-')[:26]}",
              (10, y + THUMB_H + 16), scale=0.5)
        y += THUMB_H + 26
        if y + THUMB_H > height:
            break
    return panel


def main() -> None:
    p = argparse.ArgumentParser(description="Live skeleton -> meme demo (Milestone 4).")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--vis", type=float, default=0.5)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    print("Loading recommender (OpenCLIP + index + DB) ...")
    rec = Recommender()

    estimator = FullPoseEstimator(visibility_threshold=args.vis, model_path=args.model)
    buffer = SkeletonBuffer(maxlen=WINDOW_SIZE, num_joints=NUM_FULL_JOINTS,
                            joint_names=FULL_JOINT_NAMES)
    stabilizer = GestureStabilizer(min_interval=REFRESH, persist=2)
    thumb_cache: dict[str, np.ndarray] = {}
    results: list = []
    gesture_label = "neutral"
    last_refresh = 0.0

    try:
        cam = Webcam(args.device, width=args.width, height=args.height).open()
    except WebcamError as e:
        raise SystemExit(f"\n{e}\n")

    win = "skeleton -> meme demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Demo running. Hold a gesture ~2s. Keys: q / ESC = quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            landmarks = estimator.process(frame, int(time.time() * 1000))
            buffer.append(landmarks)
            draw_full_skeleton(frame, landmarks, visibility_threshold=args.vis)

            now = time.time()
            if len(buffer) >= WINDOW_SIZE // 2 and (now - last_refresh) >= REFRESH:
                last_refresh = now
                est = estimate_gesture(buffer.window(), visibility_threshold=args.vis)
                gesture_label = stabilizer.update(est, now)
                # Recommend for the stabilized, non-neutral gesture (rotates on repeat).
                if gesture_label != "neutral":
                    est.top = gesture_label
                    results = rec.recommend(est, k=args.k)

            _text(frame, f"FPS: {cam.fps:4.1f}   visible {count_visible(landmarks, args.vis)}"
                         f"/{NUM_FULL_JOINTS}   buffer {len(buffer)}/{WINDOW_SIZE}", (12, 28))
            _text(frame, "q = quit", (12, frame.shape[0] - 16))

            panel = _build_panel(results, gesture_label, frame.shape[0], thumb_cache)
            canvas = np.hstack([frame, panel])
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.release()
        estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
