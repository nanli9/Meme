"""OpenCLIP meme embeddings + a FAISS-or-NumPy vector index (Milestone 3).

Two pieces, deliberately separable:

- `MemeEmbedder` wraps OpenCLIP so images (memes) and text (queries / tag prompts) land in
  the *same* embedding space — that shared space is what lets a gesture-derived intent
  query (M4) retrieve memes. torch/open_clip are imported lazily so the index and the unit
  tests run without them.
- `VectorIndex` is exact cosine search over L2-normalized vectors. It uses FAISS
  (`IndexFlatIP`) when importable and an identical NumPy matmul otherwise — CLAUDE.md flags
  that `faiss-cpu` may not install under uv on Mac, and for a few-hundred-meme bootstrap
  brute force is instant, so FAISS is an optimization, never a hard dependency.

Source of truth on disk is the `.npy` matrix; a FAISS index is rebuilt from it on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_VECTORS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "embeddings" / "memes.npy"
)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


# --- vector index ----------------------------------------------------------
def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


class VectorIndex:
    """Exact cosine (inner-product on normalized vectors) search over `[N, D]` rows."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = _l2_normalize(vectors)
        self.dim = int(self.vectors.shape[1]) if self.vectors.size else 0
        self._faiss = _try_import_faiss()
        self._index = None
        if self._faiss is not None and self.vectors.size:
            self._index = self._faiss.IndexFlatIP(self.dim)
            self._index.add(np.ascontiguousarray(self.vectors))

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def __len__(self) -> int:
        return int(self.vectors.shape[0]) if self.vectors.size else 0

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Return `(scores, ids)` each shaped `[Q, k]`, sorted by descending cosine.
        `k` is capped at the number of indexed vectors. Empty index -> empty arrays."""
        q = _l2_normalize(queries)
        n = len(self)
        if n == 0:
            return np.empty((q.shape[0], 0), np.float32), np.empty((q.shape[0], 0), np.int64)
        k = max(1, min(k, n))
        if self._index is not None:
            scores, ids = self._index.search(np.ascontiguousarray(q), k)
            return scores.astype(np.float32), ids.astype(np.int64)
        # NumPy fallback: full matmul, then top-k per query.
        sims = q @ self.vectors.T                       # [Q, N]
        ids = np.argsort(-sims, axis=1)[:, :k]          # descending
        scores = np.take_along_axis(sims, ids, axis=1)
        return scores.astype(np.float32), ids.astype(np.int64)

    def save(self, path: Path | str = DEFAULT_VECTORS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_VECTORS_PATH) -> "VectorIndex":
        return cls(np.load(Path(path)))


# --- OpenCLIP embedder -----------------------------------------------------
class MemeEmbedder:
    """OpenCLIP image+text encoder. Outputs are L2-normalized so inner product = cosine."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ) -> None:
        import open_clip  # lazy: heavy import only when actually embedding
        import torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            self.dim = int(self.embed_text("dimension probe").shape[-1])

    def embed_images(self, images: Sequence, batch_size: int = 32) -> np.ndarray:
        """Embed PIL images -> `[N, D]` normalized float32."""
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in batch])
            with torch.no_grad():
                feats = self.model.encode_image(tensors.to(self.device))
            out.append(feats.float().cpu().numpy())
        if not out:
            return np.empty((0, self.dim), np.float32)
        return _l2_normalize(np.concatenate(out, axis=0))

    def embed_image(self, image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_text(self, texts) -> np.ndarray:
        """Embed a string or list of strings -> `[N, D]` (or `[D]` for a single string)."""
        torch = self._torch
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        tokens = self.tokenizer(items)
        with torch.no_grad():
            feats = self.model.encode_text(tokens.to(self.device))
        arr = _l2_normalize(feats.float().cpu().numpy())
        return arr[0] if single else arr
