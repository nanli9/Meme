"""Run the rule-based gesture estimator over a saved window (Milestone 2).

Headless verification / threshold tuning without a camera: load a `.npz` session,
classify it, and print the top gesture + ranked scores.

    uv run python scripts/label_session.py                 # newest session
    uv run python scripts/label_session.py path/to/x.npz   # a specific file
    uv run python scripts/label_session.py --all           # every saved session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.motion_rules import estimate_gesture  # noqa: E402

SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def label_file(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    window = data["landmarks"].astype(np.float32)
    normalized = bool(data["normalized"]) if "normalized" in data else False

    est = estimate_gesture(window, assume_normalized=normalized)
    ranked = sorted(est.scores.items(), key=lambda kv: kv[1], reverse=True)

    nonzero = "  ".join(f"{name}={score:.2f}" for name, score in ranked if score > 0.01)
    print(f"\n{path.name}  [T={window.shape[0]}, J={window.shape[1]}, normalized={normalized}]")
    print(f"  -> {est.top}   (confidence {est.confidence:.2f}, valid_ratio {est.valid_ratio:.2f})")
    print("  scores: " + (nonzero if nonzero else "(all ~0)"))


def main() -> None:
    p = argparse.ArgumentParser(description="Label saved skeleton windows with gestures.")
    p.add_argument("path", nargs="?", help="A .npz session (default: newest).")
    p.add_argument("--all", action="store_true", help="Label every session in data/sessions/.")
    args = p.parse_args()

    if args.all:
        files = sorted(SESSION_DIR.glob("*.npz"))
        if not files:
            raise SystemExit(f"No sessions in {SESSION_DIR}.")
        for f in files:
            label_file(f)
        return

    if args.path:
        path = Path(args.path)
    else:
        files = sorted(SESSION_DIR.glob("*.npz"))
        if not files:
            raise SystemExit(f"No sessions in {SESSION_DIR}. Save one from the debugger first.")
        path = files[-1]
    label_file(path)


if __name__ == "__main__":
    main()
