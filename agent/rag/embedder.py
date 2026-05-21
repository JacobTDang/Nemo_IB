"""Singleton wrapper around the sentence-transformers MiniLM embedder.

The first call lazily downloads (~80MB) and loads the `all-MiniLM-L6-v2`
model — this is slow on cold start but fast (~1ms/sentence) thereafter. The
model instance is cached at module level so subsequent imports reuse it.

Output convention:
  - All embeddings are returned as numpy float32 arrays (compatible with
    sqlite-vec's `float[384]` column).
  - Embeddings are L2-normalized so cosine similarity equals dot product —
    this is what sqlite-vec's `vec_distance_cosine` expects under the hood
    and it keeps downstream similarity math simple.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np


_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_DIM = 384
_model: Optional[object] = None

# Thread-safe lazy load. Without this, a warmup daemon thread and the
# main thread's first embed() call can race — both see _model is None,
# both kick off independent loads, both block on the same import lock,
# and the "warmup" delivers no speedup. The lock serializes loading;
# the event lets callers wait for it without busy-spinning.
_load_lock = threading.Lock()
_loaded_event = threading.Event()


def _load_model():
    """Lazy-load the SentenceTransformer model under a lock so concurrent
    callers (warmup thread + main thread) share a single load. Called on
    first embed()."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        # Re-check inside the lock — another thread may have loaded it
        # while we were waiting.
        if _model is None:
            # Import locally so module import doesn't pay the
            # sentence_transformers cost (torch etc) unless embed() is
            # actually called.
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
            _loaded_event.set()
    return _model


def await_loaded(timeout: Optional[float] = None) -> bool:
    """Block until the model is loaded (typically by a warmup thread).
    Returns True if loaded within timeout, False otherwise. Useful when
    the caller wants to make sure first embed() doesn't pay cold-start
    cost.
    """
    return _loaded_event.wait(timeout=timeout)


def is_loaded() -> bool:
    """Non-blocking check: has the model been loaded into the singleton?"""
    return _loaded_event.is_set()


def embed(text: str) -> np.ndarray:
    """Embed a single string. Returns (384,) float32, L2-normalized.

    Empty strings are accepted but produce a zero vector — callers that
    care about meaningful similarity should filter empties upstream.
    """
    if text is None:
        text = ""
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, 384) float32, L2-normalized."""
    if not texts:
        return np.zeros((0, _MODEL_DIM), dtype=np.float32)
    model = _load_model()
    # Replace Nones with empty strings to keep the encoder happy.
    safe = [t if t is not None else "" for t in texts]
    vecs = model.encode(
        safe,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def dim() -> int:
    """Public accessor for the model's embedding dimensionality."""
    return _MODEL_DIM
