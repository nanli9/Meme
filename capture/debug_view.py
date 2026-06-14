"""Native OpenCV skeleton debugger (Milestone 1, fast path).

Runs at full camera/inference speed (no browser overhead). Shows the live feed with
the 13-joint overlay and a HUD (FPS, visible landmarks, buffer length).

    uv run python -m capture.debug_view
    uv run python -m capture.debug_view --device 1 --normalize

Keys:  s = save the current 60-frame window   q / ESC = quit
"""

from __future__ import annotations

import argparse
import time

import cv2

from capture.pose_mediapipe import DEFAULT_MODEL_PATH, count_visible, draw_skeleton
from capture.skeleton import (
    FULL_JOINT_NAMES,
    NUM_FULL_JOINTS,
    FullPoseEstimator,
    draw_full_skeleton,
)
from capture.pose_mediapipe import NUM_JOINTS, PoseEstimator
from capture.webcam import Webcam, WebcamError
from features.skeleton_buffer import WINDOW_SIZE, SkeletonBuffer


def _hud(frame, lines: list[str], *, color=(255, 255, 255)) -> None:
    y = 28
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        y += 30


def main() -> None:
    p = argparse.ArgumentParser(description="Native OpenCV skeleton debugger (Milestone 1).")
    p.add_argument("--device", type=int, default=0, help="Webcam device index.")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Pose .task model path.")
    p.add_argument("--vis", type=float, default=0.5, help="Visibility threshold for drawing/counting.")
    p.add_argument("--normalize", action="store_true", help="Torso-normalize saved windows.")
    p.add_argument("--no-mirror", action="store_true", help="Disable selfie-view mirroring.")
    p.add_argument("--no-hands", action="store_true", help="Body only (skip finger landmarks).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    use_hands = not args.no_hands
    if use_hands:
        estimator = FullPoseEstimator(visibility_threshold=args.vis, model_path=args.model)
        num_joints, joint_names = NUM_FULL_JOINTS, FULL_JOINT_NAMES
        draw = draw_full_skeleton
    else:
        estimator = PoseEstimator(model_path=args.model, visibility_threshold=args.vis)
        num_joints, joint_names = NUM_JOINTS, tuple()  # default body names in buffer
        draw = draw_skeleton

    buffer = (
        SkeletonBuffer(maxlen=WINDOW_SIZE, num_joints=num_joints, joint_names=joint_names)
        if use_hands else SkeletonBuffer(maxlen=WINDOW_SIZE)
    )
    flash_until = 0.0
    flash_msg = ""

    try:
        cam = Webcam(args.device, width=args.width, height=args.height).open()
    except WebcamError as e:
        raise SystemExit(f"\n{e}\n")

    print("Skeleton debugger running. Keys: s = save window, q / ESC = quit.")
    win = "skeleton debugger"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)  # selfie view; process the mirrored frame for consistency

            landmarks = estimator.process(frame, int(time.time() * 1000))
            buffer.append(landmarks)
            draw(frame, landmarks, visibility_threshold=args.vis)

            visible = count_visible(landmarks, args.vis)
            _hud(frame, [
                f"FPS: {cam.fps:4.1f}",
                f"Visible landmarks: {visible} / {num_joints}",
                f"Buffer: {len(buffer)} / {WINDOW_SIZE}",
                "s = save window   q = quit",
            ])
            if time.time() < flash_until:
                _hud_flash(frame, flash_msg)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("s"):
                path = buffer.save_window(fps=cam.fps, normalize=args.normalize)
                if path is None:
                    flash_msg = "Buffer empty - nothing to save"
                else:
                    flash_msg = f"Saved -> {path.name}"
                    print(f"Saved {len(buffer)}-frame window -> {path}")
                flash_until = time.time() + 1.5

            # Quit if the window was closed via the title-bar button.
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.release()
        estimator.close()
        cv2.destroyAllWindows()


def _hud_flash(frame, msg: str) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, msg, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, msg, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
