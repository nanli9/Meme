"""Train the learned hand-gesture classifier on HaGRID (Milestone 6 for hands).

Runs MediaPipe Hands over a balanced HaGRID subset, normalizes the landmarks into 63-dim
features, trains a small MLP, reports held-out accuracy, and saves the model to
data/models/hand_gesture_clf.joblib (git-ignored, regenerable).

    uv run python scripts/train_hand_classifier.py                 # ~400 imgs/source class
    uv run python scripts/train_hand_classifier.py --per-class 200 # quicker
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

from capture.hands_mediapipe import HandEstimator  # noqa: E402
from models.hand_classifier import DEFAULT_MODEL_PATH, extract_hand_features  # noqa: E402

REPO, ZIP_NAME, _CLASS_TAG = "cj-mills/hagrid-sample-30k-384p", "hagrid-sample-30k-384p.zip", "train_val_"

# HaGRID source class -> training label.
TARGET_SOURCES = {"like": "thumbs_up", "one": "pointing", "palm": "open_palm",
                  "stop": "open_palm", "peace": "peace", "peace_inverted": "peace"}
OTHER_SOURCES = ["fist", "ok", "rock", "four", "mute", "call"]  # -> "other"


def _class_of(path: str) -> str | None:
    for part in path.split("/"):
        if part.startswith(_CLASS_TAG):
            return part[len(_CLASS_TAG):]
    return None


def _largest_hand(left, right):
    """Pick the more prominent (larger bounding-box) detected hand — the gesturing one."""
    best, best_area = None, -1.0
    for h in (left, right):
        if h is None:
            continue
        xy = h[:, :2]
        area = (xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min())
        if area > best_area:
            best, best_area = h, area
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Train the HaGRID hand-gesture classifier.")
    p.add_argument("--per-class", type=int, default=400, help="Images per target source class.")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import joblib

    label_of = dict(TARGET_SOURCES)
    for c in OTHER_SOURCES:
        label_of[c] = "other"
    quota = {c: args.per_class for c in TARGET_SOURCES}
    quota.update({c: max(60, args.per_class // 4) for c in OTHER_SOURCES})

    print(f"Locating {REPO} (cached after first download)…")
    zip_path = hf_hub_download(REPO, ZIP_NAME, repo_type="dataset")

    est = HandEstimator()
    X, y = [], []
    ts = 0
    with zipfile.ZipFile(zip_path) as zf:
        by_class: dict[str, list[str]] = defaultdict(list)
        for n in zf.namelist():
            if n.lower().endswith((".jpg", ".jpeg", ".png")):
                c = _class_of(n)
                if c in label_of:
                    by_class[c].append(n)
        try:
            for c in sorted(by_class):
                used = 0
                for name in by_class[c][: quota[c]]:
                    ts += 1
                    with zf.open(name) as fh:
                        rgb = np.asarray(Image.open(io.BytesIO(fh.read())).convert("RGB"))
                    left, right = est.process(np.ascontiguousarray(rgb[:, :, ::-1]), ts)
                    hand = _largest_hand(left, right)
                    if hand is None:
                        continue
                    feat = extract_hand_features(hand)
                    if feat is not None:
                        X.append(feat); y.append(label_of[c]); used += 1
                print(f"  {c:>14} -> {label_of[c]:<10} {used} samples")
        finally:
            est.close()

    X, y = np.array(X), np.array(y)
    print(f"\nTotal samples: {len(X)}  classes: {sorted(set(y))}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_frac, random_state=0, stratify=y)

    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=0))
    print("Training MLP…")
    clf.fit(Xtr, ytr)
    print("\n=== Held-out report ===")
    print(classification_report(yte, clf.predict(Xte), digits=3))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "classes": sorted(set(y))}, out)
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    main()
