"""Load a ~200-300 row Hugging Face meme subset (Milestone 3).

Bootstrap, NOT a scraper (CLAUDE.md): pull a small SUBSET of an existing labeled dataset
to prove the pipeline runs end to end. A random subset validates *plumbing, not funniness*
— curate for quality only after the pipeline works.

Default source `not-lain/meme-dataset` (image + text, ~300 rows). `--source` / `--limit`
let you swap in MemeCap / harpreetsahota later without touching the rest of the pipeline.
Images are written under `data/memes/` (gitignored); rows returned for embedding + DB.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SOURCE = "not-lain/meme-dataset"
DEFAULT_LIMIT = 200
MEMES_DIR = Path(__file__).resolve().parent.parent / "data" / "memes"

# Column-name candidates across datasets — first present one wins.
_IMAGE_KEYS = ("image", "img", "images")
_TEXT_KEYS = ("text", "caption", "title", "post_title", "label", "name", "meme_caption")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _first_present(cols, candidates) -> str | None:
    for k in candidates:
        if k in cols:
            return k
    return None


def load_subset(
    source: str = DEFAULT_SOURCE,
    limit: int = DEFAULT_LIMIT,
    split: str = "train",
    out_dir: Path | str = MEMES_DIR,
) -> list[dict]:
    """Download a subset, save images locally, and return rows:
    `{source, source_id, image_path, text, copyrighted_character, private_person}`.
    `id`/tags/intensity are filled later by build_index + auto_label."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_slug = _safe(source)

    ds = load_dataset(source, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    cols = ds.column_names
    image_key = _first_present(cols, _IMAGE_KEYS)
    text_key = _first_present(cols, _TEXT_KEYS)
    if image_key is None:
        raise ValueError(f"No image column found in {source}; columns={cols}")

    rows: list[dict] = []
    for i, item in enumerate(ds):
        image = item[image_key]
        if image is None:
            continue
        img_path = out_dir / f"{src_slug}_{i:05d}.jpg"
        image.convert("RGB").save(img_path, format="JPEG", quality=90)
        text = str(item.get(text_key, "")) if text_key else ""
        rows.append(
            {
                "source": source,
                "source_id": str(i),
                "image_path": str(img_path),
                "text": text.strip(),
                # Safety flags default False here; M4 wires the hard-veto. A real curation
                # pass (or LLM labeler) can set these — they're stored either way.
                "copyrighted_character": False,
                "private_person": False,
            }
        )
    return rows


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Download a meme subset from Hugging Face.")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    rows = load_subset(args.source, args.limit, args.split)
    print(f"Loaded {len(rows)} memes from {args.source} -> {MEMES_DIR}")
    for r in rows[:3]:
        print(f"  {r['image_path']}  text={r['text'][:60]!r}")


if __name__ == "__main__":
    _cli()
