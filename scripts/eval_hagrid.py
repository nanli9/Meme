"""Evaluate the finger-gesture rules against HaGRID (Milestone 2 tuning).

Reads images straight from the cached HaGRID sample zip, runs each through our pose+hands
pipeline, classifies it, and compares to the dataset label — a labeled check for
`thumbs_up` / `pointing` / `open_palm` without manual posing.

    uv run python scripts/eval_hagrid.py                       # 150 imgs/class
    uv run python scripts/eval_hagrid.py --limit-per-class 50  # quicker

Dataset: cj-mills/hagrid-sample-30k-384p (one ~1GB zip; downloaded once, then cached).
Images are at `.../hagrid_30k/train_val_<class>/<uuid>.jpg`.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.skeleton import FullPoseEstimator  # noqa: E402
from evaluation.gesture_eval import HAGRID_TO_GESTURE, Confusion  # noqa: E402
from models.motion_rules import estimate_gesture  # noqa: E402

REPO = "cj-mills/hagrid-sample-30k-384p"
ZIP_NAME = "hagrid-sample-30k-384p.zip"
_CLASS_TAG = "train_val_"


def _class_of(path: str) -> str | None:
    for part in path.split("/"):
        if part.startswith(_CLASS_TAG):
            return part[len(_CLASS_TAG):]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate finger gestures on HaGRID.")
    p.add_argument("--limit-per-class", type=int, default=150)
    p.add_argument("--vis", type=float, default=0.5)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download  # local import: only needed here

    print(f"Locating {REPO} (downloads once, ~1GB, then cached)…")
    zip_path = hf_hub_download(REPO, ZIP_NAME, repo_type="dataset")

    relevant = set(HAGRID_TO_GESTURE)
    quota = {c: args.limit_per_class for c in relevant}

    est = FullPoseEstimator(visibility_threshold=args.vis)
    conf = Confusion()
    no_hand = 0
    ts = 0

    with zipfile.ZipFile(zip_path) as zf:
        images = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        # group by class so we sample evenly even though the zip is ordered by class
        by_class: dict[str, list[str]] = defaultdict(list)
        for n in images:
            c = _class_of(n)
            if c in relevant:
                by_class[c].append(n)

        try:
            for c in sorted(by_class):
                for name in by_class[c][: quota[c]]:
                    ts += 1
                    with zf.open(name) as fh:
                        rgb = np.asarray(Image.open(io.BytesIO(fh.read())).convert("RGB"))
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    skel = est.process(bgr, ts)
                    if not skel[13:, 3].any():
                        no_hand += 1
                    pred = estimate_gesture(skel[None], visibility_threshold=args.vis).top
                    conf.add(HAGRID_TO_GESTURE[c], pred)
                print(f"  {c:>5} -> done ({min(len(by_class[c]), quota[c])} imgs)")
        finally:
            est.close()

    print("\n=== HaGRID finger-gesture evaluation ===")
    print(conf.report())
    if no_hand:
        print(f"\n(note: {no_hand}/{conf.total} images had no hand detected by MediaPipe)")


if __name__ == "__main__":
    main()
