"""Meme retrieval over the vector index (Milestone 3).

Loads the vector matrix (`VectorIndex`, FAISS or NumPy) + the SQLite metadata and answers
top-k cosine queries. Two entry points:

- `retrieve(text, k)` — embed a *text* query (an expressive-intent phrase) and search. This
  is what M4 will call with a gesture-derived intent string.
- `retrieve_by_vector(vec, k)` — search a precomputed query vector directly (for when M4
  fuses skeleton + face into an intent vector in CLIP space).

This is pure similarity + metadata join. The full ranking formula (tag_match, intensity,
diversity, recent-duplicate penalty, safety veto) lands in M4 on top of these candidates.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from meme_db import schema
from meme_db.embed_memes import DEFAULT_VECTORS_PATH, MemeEmbedder, VectorIndex


class RetrievedMeme(BaseModel):
    score: float
    meme: schema.Meme


@lru_cache(maxsize=4)
def _load_index(vectors_path: str) -> VectorIndex:
    return VectorIndex.load(vectors_path)


@lru_cache(maxsize=1)
def _load_embedder() -> MemeEmbedder:
    return MemeEmbedder()


def retrieve_by_vector(
    query_vec: np.ndarray,
    k: int = 5,
    *,
    db_path: Path | str = schema.DEFAULT_DB_PATH,
    vectors_path: Path | str = DEFAULT_VECTORS_PATH,
) -> list[RetrievedMeme]:
    """Search a precomputed `[D]` (or `[1, D]`) query vector; return top-k with metadata."""
    index = _load_index(str(vectors_path))
    q = np.asarray(query_vec, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    scores, ids = index.search(q, k)
    scores, ids = scores[0], ids[0]
    conn = schema.connect(db_path)
    try:
        by_id = schema.fetch_by_ids(conn, [int(i) for i in ids])
    finally:
        conn.close()
    results = []
    for s, i in zip(scores, ids):
        meme = by_id.get(int(i))
        if meme is not None:
            results.append(RetrievedMeme(score=float(s), meme=meme))
    return results


def retrieve(
    text: str,
    k: int = 5,
    *,
    db_path: Path | str = schema.DEFAULT_DB_PATH,
    vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    embedder: MemeEmbedder | None = None,
) -> list[RetrievedMeme]:
    """Embed a text query into CLIP space and return the top-k memes with metadata."""
    embedder = embedder or _load_embedder()
    q = embedder.embed_text(text)
    return retrieve_by_vector(q, k, db_path=db_path, vectors_path=vectors_path)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Retrieve memes for a text query.")
    ap.add_argument("query", nargs="+", help="expressive-intent phrase, e.g. 'facepalm reaction'")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    results = retrieve(" ".join(args.query), args.k)
    if not results:
        print("No results — did you run `python -m meme_db.build_index` first?")
        return
    print(f"Top {len(results)} for {' '.join(args.query)!r}:")
    for rank, r in enumerate(results, 1):
        m = r.meme
        flags = "".join(c for c, on in (("C", m.copyrighted_character), ("P", m.private_person)) if on)
        print(
            f"  {rank}. [{r.score:.3f}] id={m.id} tags={'/'.join(m.tags) or '-'} "
            f"int={m.intensity:.2f} {flags} {Path(m.image_path).name}"
        )
        if m.text:
            print(f"        text: {m.text[:80]}")


if __name__ == "__main__":
    _cli()
