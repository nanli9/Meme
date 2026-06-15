"""Auto-label memes with weak reaction tags via CLIP zero-shot (Milestone 3).

Each meme image embedding is compared (in the shared CLIP space) against a fixed
*reaction-tag* vocabulary; the closest tags + a confidence-based `intensity` become the
meme's weak metadata. This needs no API key (it reuses the OpenCLIP model already loaded
for embedding) and produces exactly the `tags` + `intensity` fields the M4 ranking formula
consumes (`tag_match`, `intensity_match`).

These tags are deliberately weak — everything in this system is a weak signal. An LLM
captioner/labeler is the documented upgrade path (offline only; see CLAUDE.md), and would
slot in behind the same `auto_label` interface.
"""

from __future__ import annotations

import numpy as np

# Reaction vocabulary — broad *visible reactions*, never mental-state claims (CLAUDE.md).
# Overlaps the gesture vocab (shrug/thinking/celebrating…) so motion-derived intent in M4
# has tags to match against, plus expression words a meme's face/scene might convey.
TAG_VOCABULARY: tuple[str, ...] = (
    "amused", "laughing", "smug", "sarcastic", "deadpan", "unimpressed",
    "annoyed", "frustrated", "angry", "disgusted",
    "shocked", "surprised", "mind blown", "scared",
    "confused", "thinking", "skeptical", "awkward", "embarrassed",
    "sad", "crying", "disappointed", "bored",
    "excited", "hyped", "celebrating", "proud", "wholesome",
    "agreement", "disagreement", "shrug", "facepalm", "cringe", "flirty",
)
_PROMPT = "a reaction meme expressing {}"


def tag_text_vectors(embedder, vocabulary=TAG_VOCABULARY):
    """Embed the tag prompts -> `(tags, vectors[V, D])`."""
    prompts = [_PROMPT.format(t) for t in vocabulary]
    return list(vocabulary), embedder.embed_text(prompts)


def _softmax(x: np.ndarray, temperature: float, axis: int = -1) -> np.ndarray:
    z = (x - x.max(axis=axis, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def auto_label(
    image_vecs: np.ndarray,
    tags: list[str],
    tag_vecs: np.ndarray,
    *,
    top_k: int = 3,
    rel_threshold: float = 0.6,
    temperature: float = 0.07,
) -> list[dict]:
    """Score `image_vecs[N, D]` against `tag_vecs[V, D]` (both L2-normalized) and return,
    per meme, `{"tags": [...], "intensity": float}`.

    `tags`: up to `top_k`, keeping any whose softmax prob is >= `rel_threshold` * the top
    prob (so a clearly-single-reaction meme gets one tag, an ambiguous one gets a few).
    `intensity`: the top softmax probability (0..1) — how confidently any one reaction
    dominates, used as the M4 `intensity_match` cue. `temperature` ~0.07 matches CLIP's
    contrastive scale so the small cosine gaps become a peaked distribution."""
    if image_vecs.size == 0:
        return []
    sims = np.asarray(image_vecs, np.float32) @ np.asarray(tag_vecs, np.float32).T  # [N, V]
    probs = _softmax(sims, temperature, axis=1)
    order = np.argsort(-probs, axis=1)
    out: list[dict] = []
    for i in range(probs.shape[0]):
        ranked = order[i]
        top_p = float(probs[i, ranked[0]])
        cutoff = rel_threshold * top_p
        chosen = [tags[j] for j in ranked[:top_k] if probs[i, j] >= cutoff]
        out.append({"tags": chosen, "intensity": round(top_p, 4)})
    return out


def auto_label_with_embedder(image_vecs: np.ndarray, embedder, **kwargs) -> list[dict]:
    """Convenience: build tag vectors from `embedder`, then `auto_label`."""
    tags, tag_vecs = tag_text_vectors(embedder)
    return auto_label(image_vecs, tags, tag_vecs, **kwargs)
