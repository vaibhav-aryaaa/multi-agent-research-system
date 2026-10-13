"""Feature engineering module for claim-evidence sentence pairs."""

from typing import Tuple
import numpy as np

def best_evidence(
    claim_emb: np.ndarray,
    passage_embs: np.ndarray,
    k: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the top-k most similar passages by cosine similarity.
    
    Returns (indices, similarities).
    """
    if passage_embs.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    c = claim_emb.flatten()
    # If passage_embs are L2 normalized and c is normalized, dot product is cosine similarity
    norm_c = np.linalg.norm(c)
    norm_passages = np.linalg.norm(passage_embs, axis=1) + 1e-9
    sims = np.dot(passage_embs, c) / (norm_passages * (norm_c + 1e-9))

    top_k = min(k, len(sims))
    sorted_idx = np.argsort(sims)[::-1][:top_k]
    sorted_sims = sims[sorted_idx].astype(np.float32)

    return sorted_idx, sorted_sims

def pair_features(claim_emb: np.ndarray, evidence_emb: np.ndarray) -> np.ndarray:
    """Concatenate [claim, evidence, abs(claim - evidence), claim * evidence, cos_sim].
    
    Standard sentence-pair feature layout with cosine similarity as scalar feature.
    Shape: (4 * D + 1,) where D is embedding dimension (e.g., 1537 for D=384).
    """
    c = claim_emb.flatten().astype(np.float32)
    e = evidence_emb.flatten().astype(np.float32)

    diff = np.abs(c - e)
    prod = c * e

    norm_c = np.linalg.norm(c)
    norm_e = np.linalg.norm(e)
    cos_sim = np.dot(c, e) / ((norm_c * norm_e) + 1e-9)
    cos_sim_arr = np.array([cos_sim], dtype=np.float32)

    return np.concatenate([c, e, diff, prod, cos_sim_arr])

def build_claim_features(
    claim_emb: np.ndarray,
    passage_embs: np.ndarray,
    k: int = 2,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Build pair features for a claim against evidence passages using the top-k mean.
    
    Returns:
    - features: 1D np.ndarray of shape (1537,)
    - top1_sim: float scalar of top-1 similarity (coverage signal)
    - top_indices: np.ndarray of indices of top-k passages used
    """
    c = claim_emb.flatten()
    dim = len(c)

    if passage_embs.shape[0] == 0:
        e = np.zeros(dim, dtype=np.float32)
        feats = pair_features(c, e)
        return feats, 0.0, np.array([], dtype=int)

    top_indices, sims = best_evidence(c, passage_embs, k=k)
    top1_sim = float(sims[0]) if len(sims) > 0 else 0.0

    selected_embs = passage_embs[top_indices]
    mean_evidence = np.mean(selected_embs, axis=0)
    norm_ev = np.linalg.norm(mean_evidence)
    if norm_ev > 1e-9:
        mean_evidence = mean_evidence / norm_ev

    feats = pair_features(c, mean_evidence)
    return feats, top1_sim, top_indices
