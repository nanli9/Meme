"""Replay a saved skeleton window offline (Milestone 1, acceptance criterion 7).

Loads a ``.npz`` window written by the debugger and re-draws the skeleton frame by
frame on a blank canvas at the saved FPS. Works headless too (``--no-display``),
in which case it just validates and prints a summary.

Examples::

    uv run python scripts/replay_session.py                 # replay newest session
    uv run python scripts/replay_session.py path/to/x.npz   # replay a specific file
    uv run python scripts/replay_session.py --no-display     # validate only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.pose_mediapipe import NUM_CHANNELS, NUM_JOINTS  # noqa: E402
from capture.skeleton import draw_full_skeleton  # noqa: E402

SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def find_latest_session() -> Path:
    files = sorted(SESSION_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(
            f"No sessions found in {SESSION_DIR}. Run the debugger and click 'Save Window' first."
        )
    return files[-1]


def load_window(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    landmarks = data["landmarks"]
    if landmarks.ndim != 3 or landmarks.shape[2] != NUM_CHANNELS or landmarks.shape[1] < NUM_JOINTS:
        raise ValueError(f"Unexpected landmarks shape {landmarks.shape} in {path}")
    return {
        "timestamp": float(data["timestamp"]),
        "fps": float(data["fps"]),
        "landmarks": landmarks.astype(np.float32),
        "normalized": bool(data["normalized"]) if "normalized" in data else False,
        "joint_names": list(data["joint_names"]) if "joint_names" in data else None,
    }


def summarize(path: Path, win: dict) -> None:
    lm = win["landmarks"]
    T = lm.shape[0]
    # Mean count of visible joints per frame.
    visible_per_frame = (lm[:, :, 3] >= 0.5).sum(axis=1)
    print(f"File:        {path}")
    print(f"Saved at:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(win['timestamp']))}")
    print(f"Frames (T):  {T}")
    print(f"Shape:       {lm.shape}  (T, J, C)")
    print(f"Saved FPS:   {win['fps']:.1f}")
    print(f"Normalized:  {win['normalized']}")
    print(f"Visible joints/frame: min={visible_per_frame.min()} "
          f"mean={visible_per_frame.mean():.1f} max={visible_per_frame.max()}")


def replay(win: dict, *, canvas: int = 720, speed: float = 1.0) -> None:
    import cv2

    lm = win["landmarks"]
    normalized = win["normalized"]
    fps = win["fps"] if win["fps"] > 0 else 30.0
    delay = max(1, int(1000.0 / (fps * speed)))

    print("Press 'q' to quit, any other key to step a frame faster.")
    for t in range(lm.shape[0]):
        frame = np.zeros((canvas, canvas, 3), dtype=np.uint8)
        coords = lm[t].copy()
        if normalized:
            # Map torso-normalized coords (~[-1.5, 1.5]) back into [0,1] for display.
            coords[:, 0] = coords[:, 0] * 0.25 + 0.5
            coords[:, 1] = coords[:, 1] * 0.25 + 0.5
        draw_full_skeleton(frame, coords, visibility_threshold=0.5)
        cv2.putText(frame, f"frame {t+1}/{lm.shape[0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("skeleton replay", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved skeleton window.")
    parser.add_argument("path", nargs="?", help="Path to a .npz session (default: newest).")
    parser.add_argument("--no-display", action="store_true", help="Validate + summarize only.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    args = parser.parse_args()

    path = Path(args.path) if args.path else find_latest_session()
    win = load_window(path)
    summarize(path, win)

    if args.no_display:
        return
    try:
        replay(win, speed=args.speed)
    except Exception as e:  # headless / no GUI backend
        print(f"\n(Could not open a display window: {e})")
        print("Re-run with --no-display to validate without a GUI.")


if __name__ == "__main__":
    main()
