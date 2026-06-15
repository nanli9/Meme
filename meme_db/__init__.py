"""Meme retrieval DB (Milestone 3).

faiss-cpu and torch each bundle their own OpenMP runtime (libomp). On macOS, loading both
in one process — which the live pipeline must (torch to embed, faiss to search) — aborts
with "OMP: Error #15: libomp.dylib already initialized". We set the documented workaround
here, *before* any heavy import, so importing anything under `meme_db` is safe. The index
is an exact flat inner-product search, so the threaded-correctness caveat behind this flag
doesn't apply; `embed_memes.VectorIndex` also falls back to NumPy if faiss is absent.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
