"""Load meme subsets from Hugging Face (Milestone 3, scaled up in M4-era).

Bootstrap from existing labeled datasets, NOT a scraper (CLAUDE.md). The default is now
**MemeCap** (`Leonardo6/memecap`, ~5.8k real memes with rich title+description captions) —
a ~20x jump over the original 300-row bootstrap. The loader handles three shapes:

  * generic `image`/`text`-style columns (e.g. not-lain/meme-dataset),
  * an image stored under `local_path` (e.g. Goodnight7/meme_templates_1k),
  * MemeCap's nested `messages` (chat-style caption) + `images` (PIL list) schema.

Multiple sources can be combined (`--source a,b,c`) to grow the DB further; `--limit 0`
takes a whole dataset. Images are written under `data/memes/` (gitignored); rows are
returned for embedding + DB. Curate for quality after the pipeline works, then scale.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# A few vetted, on-theme sources (avoid the hateful-memes sets). Aliases for convenience.
KNOWN_SOURCES = {
    "memecap": "Leonardo6/memecap",                 # ~5823, rich captions, real memes
    "memetion": "AshuReddy/memetion_dataset_7k",     # ~6992, image-only
    "templates": "Goodnight7/meme_templates_1k",     # ~992, named meme templates
    "not-lain": "not-lain/meme-dataset",             # ~300, original bootstrap
}
DEFAULT_SOURCE = "memecap"
DEFAULT_LIMIT = 0   # 0 = whole dataset
MEMES_DIR = Path(__file__).resolve().parent.parent / "data" / "memes"

# Column-name candidates across datasets — first present one wins.
_IMAGE_KEYS = ("image", "img", "images", "local_path")
_TEXT_KEYS = ("text", "caption", "title", "post_title", "label", "name", "meme_caption")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _first_present(cols, candidates) -> str | None:
    for k in candidates:
        if k in cols:
            return k
    return None


def _to_pil(obj):
    """Coerce a dataset image field to a PIL image (PIL object, or a path string)."""
    from PIL import Image
    if obj is None:
        return None
    if hasattr(obj, "convert"):
        return obj
    if isinstance(obj, str) and Path(obj).exists():
        try:
            return Image.open(obj)
        except Exception:
            return None
    return None


def _memecap_text(messages) -> str:
    """Pull a short caption from MemeCap's chat-style `messages` list.
    Prefer the quoted title; else the first chunk of the first message's content."""
    if not messages:
        return ""
    content = ""
    for m in messages:
        if isinstance(m, dict) and m.get("content"):
            content = str(m["content"])
            break
    title = re.search(r'title\s+"([^"]+)"', content)
    if title:
        return title.group(1).strip()
    return content[:160].strip()


def load_subset(
    source: str = DEFAULT_SOURCE,
    limit: int = DEFAULT_LIMIT,
    split: str = "train",
    out_dir: Path | str = MEMES_DIR,
) -> list[dict]:
    """Download one source, save images locally, and return rows
    `{source, source_id, image_path, text, copyrighted_character, private_person}`.
    `id`/tags/intensity are filled later by build_index + auto_label."""
    from datasets import load_dataset

    source = KNOWN_SOURCES.get(source, source)   # allow short aliases
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_slug = _safe(source)

    ds = load_dataset(source, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    cols = ds.column_names
    is_memecap = "messages" in cols and "images" in cols
    image_key = _first_present(cols, _IMAGE_KEYS)
    text_key = _first_present(cols, _TEXT_KEYS)
    if not is_memecap and image_key is None:
        raise ValueError(f"No image column found in {source}; columns={cols}")

    rows: list[dict] = []
    for i, item in enumerate(ds):
        if is_memecap:
            imgs = item.get("images") or []
            image = _to_pil(imgs[0]) if imgs else None
            text = _memecap_text(item.get("messages"))
        else:
            image = _to_pil(item.get(image_key))
            text = str(item.get(text_key, "")) if text_key else ""
        if image is None:
            continue
        img_path = out_dir / f"{src_slug}_{i:05d}.jpg"
        try:
            image.convert("RGB").save(img_path, format="JPEG", quality=90)
        except Exception:
            continue  # skip unreadable images rather than crash the whole build
        rows.append(
            {
                "source": source,
                "source_id": str(i),
                "image_path": str(img_path),
                "text": text.strip(),
                # Safety flags default False; M4 wires the hard-veto. A curation pass (or
                # LLM labeler) can set these — they're stored either way.
                "copyrighted_character": False,
                "private_person": False,
            }
        )
    return rows


def load_sources(
    sources: list[str] | str,
    limit: int = DEFAULT_LIMIT,
    split: str = "train",
    out_dir: Path | str = MEMES_DIR,
) -> list[dict]:
    """Load and concatenate several sources (each capped at `limit`; 0 = whole dataset)."""
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    rows: list[dict] = []
    for src in sources:
        n_before = len(rows)
        rows.extend(load_subset(src, limit, split, out_dir))
        print(f"  + {len(rows) - n_before} from {KNOWN_SOURCES.get(src, src)}")
    return rows


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Download meme subset(s) from Hugging Face.")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="dataset id or alias; comma-separate to combine "
                         f"(aliases: {', '.join(KNOWN_SOURCES)})")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="per-source cap (0 = all)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    rows = load_sources(args.source, args.limit, args.split)
    print(f"Loaded {len(rows)} memes -> {MEMES_DIR}")
    for r in rows[:3]:
        print(f"  {r['image_path']}  text={r['text'][:60]!r}")


if __name__ == "__main__":
    _cli()
