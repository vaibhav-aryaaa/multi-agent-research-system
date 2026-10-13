"""Lazy-loaded sentence-transformer embedder using all-MiniLM-L6-v2 on CPU."""

from typing import List, Optional
import numpy as np

_MODEL = None
EMBEDDING_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_model():
    """Lazily load the MiniLM embedding model on CPU once."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
    return _MODEL

def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of text strings into L2-normalized vectors.
    
    Returns an np.ndarray of shape (len(texts), 384) with float32 values.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
        device="cpu",
    )
    return np.asarray(embeddings, dtype=np.float32)
