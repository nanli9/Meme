"""Fusion-ranker tests (Milestone 4). Pure ranking — no torch, no DB, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meme_db.schema import Meme  # noqa: E402
from models.fusion_ranker import (  # noqa: E402
    GESTURE_INTENT,
    Candidate,
    FusionRanker,
    Intent,
    RankWeights,
    gesture_to_intent,
)
from models.motion_rules import GESTURES  # noqa: E402


def _cand(id, *, clip, tags=(), intensity=0.0, vec=(1.0, 0.0), flags=()):
    v = np.asarray(vec, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return Candidate(
        meme=Meme(id=id, tags=list(tags), intensity=intensity,
                  copyrighted_character=("C" in flags), private_person=("P" in flags)),
        clip=clip, vec=v,
    )


# --- intent map ------------------------------------------------------------
def test_every_gesture_has_an_intent():
    for g in (*GESTURES, "neutral"):
        intent = gesture_to_intent(g)
        assert isinstance(intent, Intent) and intent.query
    # unknown gesture falls back to neutral
    assert gesture_to_intent("???") is GESTURE_INTENT["neutral"]


# --- ranking terms ---------------------------------------------------------
def test_tag_match_boosts():
    intent = Intent("x", ("facepalm",), 0.5)
    r = FusionRanker()
    cands = [_cand(0, clip=0.5, tags=["facepalm"], vec=(1, 0)),
             _cand(1, clip=0.5, tags=["amused"], vec=(0, 1))]
    out = r.rank(intent, 0.5, cands, k=2)
    assert out[0].meme.id == 0 and out[0].tag_match == 1.0


def test_recent_duplicate_is_demoted():
    intent = Intent("x", (), 0.5)
    r = FusionRanker(recent_size=8)
    cands = [_cand(0, clip=0.5, vec=(1, 0)), _cand(1, clip=0.5, vec=(0, 1))]
    first = r.rank(intent, 0.5, cands, k=1)
    assert first[0].meme.id == 0
    second = r.rank(intent, 0.5, cands, k=1)        # same gesture again
    assert second[0].meme.id == 1                   # rotated off the just-shown meme
    assert second[0].recent_penalty == 0.0


def test_diversity_prefers_distinct_vectors():
    # Pure-diversity weights: after picking one, an identical-vector dup loses to a distinct.
    w = RankWeights(clip=0.0, tag=0.0, intensity=0.0, diversity=1.0, recent_penalty=0.0)
    r = FusionRanker(w)
    cands = [_cand(0, clip=0.0, vec=(1, 0)),
             _cand(1, clip=0.0, vec=(1, 0)),     # duplicate of 0
             _cand(2, clip=0.0, vec=(0, 1))]     # distinct
    out = r.rank(Intent("x", (), 0.0), 0.0, cands, k=3)
    assert [s.meme.id for s in out] == [0, 2, 1]
    assert out[1].diversity > out[2].diversity    # distinct pick is more diverse


def test_safety_veto_excludes_flagged():
    intent = Intent("x", (), 0.5)
    r = FusionRanker()
    cands = [_cand(0, clip=0.9, flags=("C",), vec=(1, 0)),   # copyrighted -> vetoed
             _cand(1, clip=0.1, flags=("P",), vec=(0, 1)),   # private person -> vetoed
             _cand(2, clip=0.5, vec=(1, 1))]
    out = r.rank(intent, 0.5, cands, k=5)
    assert [s.meme.id for s in out] == [2]


def test_k_is_capped_to_pool():
    out = FusionRanker().rank(Intent("x", (), 0.5), 0.5,
                              [_cand(0, clip=0.5), _cand(1, clip=0.4, vec=(0, 1))], k=10)
    assert len(out) == 2


def test_all_vetoed_returns_empty():
    r = FusionRanker()
    out = r.rank(Intent("x", (), 0.5), 0.5, [_cand(0, clip=0.5, flags=("C",))], k=3)
    assert out == []
