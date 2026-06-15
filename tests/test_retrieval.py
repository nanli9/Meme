"""Meme retrieval / DB tests (Milestone 3).

No network and no torch: we exercise the index, schema, auto-labeling math, and the
retrieve join with synthetic vectors and a tiny fake embedder. The real OpenCLIP path is
covered by the live `build_index` + `retrieve` CLI runs, not unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meme_db import schema  # noqa: E402
from meme_db.auto_label_memes import auto_label  # noqa: E402
from meme_db.embed_memes import VectorIndex, _l2_normalize  # noqa: E402
from meme_db.retrieve import retrieve, retrieve_by_vector  # noqa: E402


# --- VectorIndex -----------------------------------------------------------
def test_index_search_finds_nearest():
    vecs = np.eye(4, dtype=np.float32)            # 4 orthonormal rows
    idx = VectorIndex(vecs)
    scores, ids = idx.search(vecs[2][None, :], k=1)
    assert ids[0, 0] == 2
    assert scores[0, 0] > 0.99


def test_index_ranks_descending_and_caps_k():
    vecs = _l2_normalize(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32))
    idx = VectorIndex(vecs)
    scores, ids = idx.search(np.array([[1.0, 0.0]], np.float32), k=10)  # k > N
    assert ids.shape == (1, 3)                      # capped at N
    assert list(ids[0]) == [0, 1, 2]               # nearest -> farthest
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]


def test_index_save_load_roundtrip(tmp_path):
    vecs = _l2_normalize(np.random.RandomState(0).randn(6, 5).astype(np.float32))
    p = tmp_path / "v.npy"
    VectorIndex(vecs).save(p)
    loaded = VectorIndex.load(p)
    assert len(loaded) == 6
    s, i = loaded.search(vecs[3][None, :], k=1)
    assert i[0, 0] == 3


def test_empty_index_is_graceful():
    idx = VectorIndex(np.empty((0, 4), np.float32))
    assert len(idx) == 0
    scores, ids = idx.search(np.zeros((1, 4), np.float32), k=5)
    assert ids.shape == (1, 0) and scores.shape == (1, 0)


# --- schema ----------------------------------------------------------------
def test_schema_insert_fetch_roundtrip(tmp_path):
    conn = schema.connect(tmp_path / "memes.db")
    schema.create_schema(conn, reset=True)
    memes = [
        schema.Meme(id=0, source="s", source_id="a", image_path="/x/0.jpg",
                    text="hello", tags=["amused", "smug"], intensity=0.7,
                    copyrighted_character=True),
        schema.Meme(id=1, source="s", source_id="b", image_path="/x/1.jpg",
                    text="", tags=[], intensity=0.1, private_person=True),
    ]
    assert schema.insert_memes(conn, memes) == 2
    assert schema.count(conn) == 2
    got = schema.fetch_by_ids(conn, [1, 0])
    assert got[0].tags == ["amused", "smug"] and got[0].copyrighted_character is True
    assert got[1].private_person is True and got[1].tags == []
    conn.close()


# --- auto-labeling ---------------------------------------------------------
def test_auto_label_picks_aligned_tag():
    tags = ["amused", "angry", "sad"]
    tag_vecs = np.eye(3, dtype=np.float32)
    image_vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # aligns to tag 0, 1
    out = auto_label(image_vecs, tags, tag_vecs, top_k=1)
    assert out[0]["tags"] == ["amused"]
    assert out[1]["tags"] == ["angry"]
    assert 0.0 <= out[0]["intensity"] <= 1.0


# --- end-to-end retrieve (fake embedder, no torch) -------------------------
class _FakeEmbedder:
    """Maps known phrases to known one-hot vectors so retrieval is deterministic."""

    def embed_text(self, text):
        table = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}
        return np.array(table.get(text, [0, 0, 0]), dtype=np.float32)


def _tiny_db(tmp_path):
    vecs = np.eye(3, dtype=np.float32)
    vpath = tmp_path / "v.npy"
    VectorIndex(vecs).save(vpath)
    conn = schema.connect(tmp_path / "memes.db")
    schema.create_schema(conn, reset=True)
    schema.insert_memes(conn, [
        schema.Meme(id=0, text="red meme", tags=["red"]),
        schema.Meme(id=1, text="green meme", tags=["green"]),
        schema.Meme(id=2, text="blue meme", tags=["blue"]),
    ])
    conn.close()
    return tmp_path / "memes.db", vpath


def test_retrieve_by_vector_joins_metadata(tmp_path):
    db, vpath = _tiny_db(tmp_path)
    res = retrieve_by_vector(np.array([0, 1, 0], np.float32), k=2,
                             db_path=db, vectors_path=vpath)
    assert res[0].meme.id == 1 and res[0].meme.text == "green meme"
    assert len(res) == 2 and res[0].score >= res[1].score


def test_retrieve_text_with_injected_embedder(tmp_path):
    db, vpath = _tiny_db(tmp_path)
    res = retrieve("blue", k=1, db_path=db, vectors_path=vpath, embedder=_FakeEmbedder())
    assert res[0].meme.id == 2 and res[0].meme.tags == ["blue"]
