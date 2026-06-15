"""Recommend memes for a saved skeleton window (Milestone 4) — camera-free demo/tester.

Loads a `.npz` window, estimates the gesture, maps it to an intent, retrieves + ranks
memes, and prints the top-5 with the ranking breakdown. The full pipeline end to end
without a webcam — handy for tuning the ranking and checking that repeated runs of the
same gesture rotate memes (diversity + recent-duplicate penalty).

    uv run python scripts/recommend_session.py                 # newest session
    uv run python scripts/recommend_session.py path/to/x.npz   # a specific file
    uv run python scripts/recommend_session.py --all           # every saved session
    uv run python scripts/recommend_session.py --show          # also open the meme images

Requires the meme DB built first: `uv run python -m meme_db.build_index --limit 200`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fusion_ranker import Recommender, gesture_to_intent  # noqa: E402
from models.motion_rules import estimate_gesture  # noqa: E402

SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def recommend_file(path: Path, rec: Recommender, k: int, show: bool) -> None:
    data = np.load(path, allow_pickle=True)
    window = data["landmarks"].astype(np.float32)
    normalized = bool(data["normalized"]) if "normalized" in data else False

    est = estimate_gesture(window, assume_normalized=normalized)
    intent = gesture_to_intent(est.top)
    results = rec.recommend(est, k=k)

    print(f"\n{path.name}  ->  gesture: {est.top} (conf {est.confidence:.2f})")
    print(f"  intent query: {intent.query!r}")
    if not results:
        print("  (no recommendations — is the meme DB built?)")
        return
    for rank, r in enumerate(results, 1):
        m = r.meme
        print(f"  {rank}. score={r.score:.3f}  [clip={r.clip:.2f} tag={r.tag_match:.2f} "
              f"int={r.intensity_match:.2f} div={r.diversity:.2f} "
              f"recent={r.recent_penalty:.0f}]  tags={'/'.join(m.tags) or '-'}  "
              f"{Path(m.image_path).name}")
    if show:
        _show([r.meme.image_path for r in results], path.name)


def _show(paths: list[str], title: str) -> None:
    try:
        import cv2
    except Exception:
        print("  (install opencv to use --show)")
        return
    thumbs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h = 240
        thumbs.append(cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)))
    if not thumbs:
        return
    height = max(t.shape[0] for t in thumbs)
    strip = np.hstack([cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 8,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in thumbs])
    cv2.imshow(f"recommendations: {title} (any key to continue)", strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend memes for saved skeleton windows.")
    p.add_argument("path", nargs="?", help="A .npz session (default: newest).")
    p.add_argument("--all", action="store_true", help="Process every session.")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--show", action="store_true", help="Open the recommended meme images.")
    args = p.parse_args()

    files = sorted(SESSION_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"No sessions in {SESSION_DIR}. Save one from the debugger first.")

    print("Loading recommender (OpenCLIP + index + DB) ...")
    rec = Recommender()

    targets = files if args.all else [Path(args.path) if args.path else files[-1]]
    for f in targets:
        recommend_file(f, rec, args.k, args.show)


if __name__ == "__main__":
    main()
