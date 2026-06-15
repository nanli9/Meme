"""Fusion ranker: gesture -> intent -> ranked memes (Milestone 4 baseline).

This is the first end-to-end join of motion (M2) and retrieval (M3). A `GestureEstimate`
becomes an *intent* (a CLIP query phrase + expected reaction tags + expected intensity);
the intent retrieves a candidate pool; the hand-tuned ranking formula (CLAUDE.md) picks
the top-k:

    score = 0.50*CLIP_similarity + 0.25*tag_match + 0.15*intensity_match
          + 0.10*diversity_bonus - 0.20*recent_duplicate_penalty       (safety = hard veto)

Weights are hand-tuned constants for now; Milestone 7 learns them from feedback. The
`FusionRanker` is pure (candidates + vectors in, ranking out — no torch/IO) so it's
unit-testable; `Recommender` wires the real embedder + index + DB around it.

Everything here is a *weak* signal fused together — never an emotion claim, never a direct
gesture->meme-ID mapping (CLAUDE.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from meme_db import schema
from meme_db.embed_memes import DEFAULT_VECTORS_PATH, MemeEmbedder, VectorIndex
from models.motion_rules import GestureEstimate


# --- gesture -> intent -----------------------------------------------------
@dataclass(frozen=True)
class Intent:
    query: str                  # CLIP text query (a visible-reaction phrase)
    tags: tuple[str, ...]       # reaction tags we expect a good match to carry
    intensity: float            # expected expressive intensity, 0..1


# One intent per gesture (vocabulary from CLAUDE.md). Phrases describe a *visible reaction*,
# not a mental state. `tags` overlap meme_db.auto_label_memes.TAG_VOCABULARY for tag_match.
GESTURE_INTENT: dict[str, Intent] = {
    "shrug": Intent("a shrug, indifference, i dont know, whatever reaction meme",
                    ("shrug", "confused", "unimpressed"), 0.4),
    "hype": Intent("hype, excitement, celebration, lets go reaction meme",
                   ("hyped", "excited", "celebrating"), 0.9),
    "arms_crossed": Intent("arms crossed, skeptical, unimpressed, waiting reaction meme",
                           ("unimpressed", "skeptical", "annoyed"), 0.5),
    "arms_wide": Intent("arms wide open, what, really, exasperated reaction meme",
                        ("frustrated", "shocked", "surprised"), 0.7),
    "facepalm": Intent("facepalm, disbelief, frustration, disappointment reaction meme",
                       ("facepalm", "disappointed", "frustrated"), 0.7),
    "thinking": Intent("thinking, hmm, skeptical, considering reaction meme",
                       ("thinking", "skeptical", "confused"), 0.5),
    "wave": Intent("waving hello or goodbye, friendly greeting reaction meme",
                   ("amused", "wholesome", "flirty"), 0.5),
    "clap": Intent("clapping, applause, sarcastic slow clap reaction meme",
                   ("celebrating", "sarcastic", "proud"), 0.7),
    "thumbs_up": Intent("thumbs up, approval, nice, agreement reaction meme",
                        ("agreement", "proud", "wholesome"), 0.6),
    "pointing": Intent("pointing at you, calling out, thats the one reaction meme",
                       ("smug", "excited", "amused"), 0.6),
    "middle_finger": Intent("rude, dismissive, angry, screw you reaction meme",
                            ("angry", "annoyed", "disgusted"), 0.9),
    "pinky": Intent("pinky promise, fancy, dainty, posh reaction meme",
                    ("smug", "flirty", "amused"), 0.4),
    "peace": Intent("peace sign, chill, cool, victory reaction meme",
                    ("amused", "celebrating", "proud"), 0.6),
    "open_palm": Intent("open palm, stop, talk to the hand, whatever reaction meme",
                        ("unimpressed", "annoyed", "disgusted"), 0.5),
    "neutral": Intent("a relatable reaction meme", (), 0.3),
}


def gesture_to_intent(top: str) -> Intent:
    return GESTURE_INTENT.get(top, GESTURE_INTENT["neutral"])


# --- ranking ---------------------------------------------------------------
@dataclass(frozen=True)
class RankWeights:
    clip: float = 0.50
    tag: float = 0.25
    intensity: float = 0.15
    diversity: float = 0.10
    recent_penalty: float = 0.20


@dataclass
class Candidate:
    meme: schema.Meme
    clip: float                 # raw CLIP cosine to the intent query
    vec: np.ndarray             # the meme's (L2-normalized) embedding, for diversity


class ScoredMeme(BaseModel):
    score: float
    clip: float                 # normalized CLIP term (0..1 across the pool)
    tag_match: float
    intensity_match: float
    diversity: float
    recent_penalty: float
    meme: schema.Meme


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1] across the pool; all-equal -> zeros (term contributes nothing)."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class FusionRanker:
    """Hand-tuned ranking with greedy MMR diversity + a recent-duplicate memory.

    Pure: feed it `Candidate`s (meme + raw CLIP score + embedding) and it returns ranked
    `ScoredMeme`s. It remembers recently-shown meme ids so the same gesture rotates through
    different memes on repeat (CLAUDE.md: same gesture must not always return the same meme).
    """

    def __init__(self, weights: RankWeights | None = None, *, recent_size: int = 8) -> None:
        self.w = weights or RankWeights()
        self._recent: deque[int] = deque(maxlen=recent_size)

    def rank(
        self,
        intent: Intent,
        gesture_intensity: float,
        candidates: list[Candidate],
        k: int = 5,
    ) -> list[ScoredMeme]:
        # Safety veto first — flagged memes are never recommended (hard veto).
        cands = [c for c in candidates
                 if not (c.meme.copyrighted_character or c.meme.private_person)]
        if not cands:
            return []

        clip_n = _minmax(np.array([c.clip for c in cands], dtype=np.float32))
        int_n = _minmax(np.array([c.meme.intensity for c in cands], dtype=np.float32))
        intent_tags = set(intent.tags)

        # Per-candidate base score (everything except diversity, which depends on picks).
        base, terms = [], []
        for i, c in enumerate(cands):
            tag_match = (len(intent_tags & set(c.meme.tags)) / len(intent_tags)
                         if intent_tags else 0.0)
            intensity_match = 1.0 - abs(float(gesture_intensity) - float(int_n[i]))
            recent = 1.0 if c.meme.id in self._recent else 0.0
            b = (self.w.clip * float(clip_n[i])
                 + self.w.tag * tag_match
                 + self.w.intensity * intensity_match
                 - self.w.recent_penalty * recent)
            base.append(b)
            terms.append((float(clip_n[i]), tag_match, intensity_match, recent))

        # Greedy MMR: pick one at a time, rewarding distance from already-picked memes.
        picked: list[ScoredMeme] = []
        picked_vecs: list[np.ndarray] = []
        remaining = list(range(len(cands)))
        k = max(1, min(k, len(cands)))
        while remaining and len(picked) < k:
            best_i, best_score, best_div = None, float("-inf"), 1.0
            for i in remaining:
                div = (1.0 - max(float(np.dot(cands[i].vec, pv)) for pv in picked_vecs)
                       if picked_vecs else 1.0)
                score = base[i] + self.w.diversity * div
                if score > best_score:
                    best_i, best_score, best_div = i, score, div
            c = cands[best_i]
            clip_t, tag_t, int_t, rec_t = terms[best_i]
            picked.append(ScoredMeme(
                score=round(best_score, 4), clip=round(clip_t, 4), tag_match=round(tag_t, 4),
                intensity_match=round(int_t, 4), diversity=round(best_div, 4),
                recent_penalty=rec_t, meme=c.meme,
            ))
            picked_vecs.append(c.vec)
            remaining.remove(best_i)

        for sm in picked:
            self._recent.append(sm.meme.id)
        return picked


# --- end-to-end recommender ------------------------------------------------
class Recommender:
    """Wires the real embedder + vector index + DB around `FusionRanker`."""

    def __init__(
        self,
        *,
        db_path=schema.DEFAULT_DB_PATH,
        vectors_path=DEFAULT_VECTORS_PATH,
        embedder: MemeEmbedder | None = None,
        weights: RankWeights | None = None,
        pool: int = 30,
        recent_size: int = 8,
    ) -> None:
        # Load the OpenCLIP (torch) model BEFORE the faiss index: on macOS, importing faiss
        # first and torch second segfaults (both bundle libomp). torch-then-faiss is stable
        # — same order the retrieve CLI uses. The NumPy backend is unaffected either way.
        self.db_path = db_path
        self.embedder = embedder or MemeEmbedder()
        self.index = VectorIndex.load(vectors_path)
        self.ranker = FusionRanker(weights, recent_size=recent_size)
        self.pool = pool

    def recommend(self, gesture: GestureEstimate, k: int = 5) -> list[ScoredMeme]:
        intent = gesture_to_intent(gesture.top)
        qvec = np.asarray(self.embedder.embed_text(intent.query), dtype=np.float32)
        scores, ids = self.index.search(qvec[None, :], self.pool)
        ids = [int(i) for i in ids[0]]
        conn = schema.connect(self.db_path)
        try:
            by_id = schema.fetch_by_ids(conn, ids)
        finally:
            conn.close()
        candidates = [
            Candidate(meme=by_id[i], clip=float(s), vec=self.index.vectors[i])
            for s, i in zip(scores[0], ids) if i in by_id
        ]
        return self.ranker.rank(intent, float(gesture.intensity), candidates, k)
