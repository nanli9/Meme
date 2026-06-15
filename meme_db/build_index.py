"""Build the meme DB + vector index end to end (Milestone 3).

Orchestrates the whole bootstrap:
    load HF subset -> save images -> OpenCLIP image embeddings -> CLIP zero-shot tags
    -> SQLite metadata (data/labels/memes.db) + vector matrix (data/embeddings/memes.npy)

ids are assigned densely `0..N-1` and shared between the DB row and the vector row, so
`retrieve.py` maps a search hit straight back to metadata. Re-run to rebuild from scratch
(the schema is reset).

    uv run python -m meme_db.build_index --limit 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meme_db import schema
from meme_db.auto_label_memes import auto_label_with_embedder
from meme_db.embed_memes import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    DEFAULT_VECTORS_PATH,
    MemeEmbedder,
    VectorIndex,
)
from meme_db.load_dataset import DEFAULT_LIMIT, DEFAULT_SOURCE, KNOWN_SOURCES, load_sources


def build(
    source: str = DEFAULT_SOURCE,
    limit: int = DEFAULT_LIMIT,
    db_path: Path | str = schema.DEFAULT_DB_PATH,
    vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
) -> int:
    """Run the full pipeline and return the number of memes indexed."""
    print(f"[1/5] Loading meme source(s) {source} (per-source limit={limit or 'all'}) ...")
    rows = load_sources(source, limit)
    print(f"      {len(rows)} memes downloaded.")
    if not rows:
        raise SystemExit("No memes loaded — nothing to index.")

    print(f"[2/5] Loading OpenCLIP ({model_name} / {pretrained}) ...")
    embedder = MemeEmbedder(model_name, pretrained)

    print("[3/5] Embedding meme images ...")
    image_vecs = embedder.embed_image_paths([r["image_path"] for r in rows])
    print(f"      embeddings: {image_vecs.shape} on {embedder.device}")

    print("[4/5] Auto-labeling (CLIP zero-shot reaction tags) ...")
    labels = auto_label_with_embedder(image_vecs, embedder)

    memes = [
        schema.Meme(
            id=i,
            source=r["source"],
            source_id=r["source_id"],
            image_path=r["image_path"],
            text=r["text"],
            tags=lab["tags"],
            intensity=lab["intensity"],
            copyrighted_character=r["copyrighted_character"],
            private_person=r["private_person"],
        )
        for i, (r, lab) in enumerate(zip(rows, labels))
    ]

    print("[5/5] Writing SQLite metadata + vector index ...")
    conn = schema.connect(db_path)
    schema.create_schema(conn, reset=True)
    n = schema.insert_memes(conn, memes)
    conn.close()
    index = VectorIndex(image_vecs)
    index.save(vectors_path)

    print(
        f"Done: {n} memes -> {db_path}\n"
        f"      vectors {image_vecs.shape} -> {vectors_path}  (search backend: {index.backend})"
    )
    sample = ", ".join(f"{m.id}:{'/'.join(m.tags) or '-'}" for m in memes[:5])
    print(f"      sample tags -> {sample}")
    return n


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Build the meme DB + vector index.")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="dataset id or alias; comma-separate to combine "
                         f"(aliases: {', '.join(KNOWN_SOURCES)})")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="per-source cap (0 = all)")
    ap.add_argument("--db", default=str(schema.DEFAULT_DB_PATH))
    ap.add_argument("--vectors", default=str(DEFAULT_VECTORS_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    args = ap.parse_args()
    build(args.source, args.limit, args.db, args.vectors, args.model, args.pretrained)


if __name__ == "__main__":
    _cli()
