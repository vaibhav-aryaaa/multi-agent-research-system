"""Inference module for DistilBERT cross-encoder faithfulness classifier.

Evaluates claim-evidence support by reading claim and evidence together.
For each claim, retrieves top-k evidence passages via MiniLM embeddings and scores
each pair with the cross-encoder, taking the maximum support probability.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_DEVICE = None

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "cross_encoder",
)


def load_cross_encoder() -> Tuple[Any, Any, Any]:
    """Lazily load the cross-encoder model and tokenizer once from local path or Hugging Face Hub.
    
    Heavy imports (torch, transformers) are kept inside this function so importing
    this module does not slow down startup or load PyTorch prematurely.
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    model_path = os.getenv("FAITHFULNESS_CE_PATH", DEFAULT_MODEL_PATH)

    # If it is a local path and doesn't exist, provide a clear instructional error
    is_local_dir = os.path.exists(model_path)
    is_hub_id = "/" in model_path and not is_local_dir

    if not is_local_dir and not is_hub_id:
        raise FileNotFoundError(
            f"Cross-encoder model not found at '{model_path}'.\n"
            "To resolve this, either:\n"
            "1. Train the model on Google Colab GPU with: 'python -m faithfulness.train_cross_encoder'\n"
            f"   and unzip/place the saved model folder at '{DEFAULT_MODEL_PATH}'.\n"
            "2. Set the FAITHFULNESS_CE_PATH environment variable to a valid local path or Hugging Face Hub repo id."
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cross-encoder model from '{model_path}': {exc}\n"
            "Ensure the directory contains config.json, model weights (model.safetensors or pytorch_model.bin), "
            "and tokenizer files, or that FAITHFULNESS_CE_PATH is correct."
        ) from exc

    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_DEVICE = device

    return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE


def score_claims(
    claims: List[str],
    passages: List[str],
    k: int = 3,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Score a list of claims against evidence passages using cross-encoder inference.

    Steps:
    1. Embed claims and passages with MiniLM embedder.
    2. For each claim, find the top-k most similar passages by cosine similarity.
    3. Score the claim against each candidate passage with the cross-encoder:
       (premise = passage, hypothesis = claim).
    4. Compute p_supported = max over the k candidate passages.
    5. If passages is empty, returns p_supported = 0.0 for every claim.

    Returns:
    List of dicts:
      {
        "claim": str,
        "p_supported": float,
        "best_passage_index": int,
        "top1_cosine": float,
      }
    """
    if not claims:
        return []

    # Case: No evidence passages available
    if not passages:
        return [
            {
                "claim": c,
                "p_supported": 0.0,
                "best_passage_index": -1,
                "top1_cosine": 0.0,
            }
            for c in claims
        ]

    # Step 1: Embed with MiniLM for cosine candidate retrieval
    from faithfulness.embedder import embed

    claim_embs = embed(claims)      # shape: (n_claims, D)
    passage_embs = embed(passages)  # shape: (n_passages, D)

    # Step 2: Retrieve top-k passages for each claim
    n_passages = len(passages)
    effective_k = min(k, n_passages)

    # Since MiniLM vectors are normalized, dot product equals cosine similarity
    sim_matrix = np.dot(claim_embs, passage_embs.T)  # shape: (n_claims, n_passages)

    claim_top_indices = []
    claim_top1_cos = []

    for i in range(len(claims)):
        row_sims = sim_matrix[i]
        top_k_idx = np.argsort(row_sims)[::-1][:effective_k]
        claim_top_indices.append(top_k_idx)
        claim_top1_cos.append(float(row_sims[top_k_idx[0]]) if len(top_k_idx) > 0 else 0.0)

    # Step 3: Prepare pairs for cross-encoder scoring
    # Sentence pair: premise = passage (evidence), hypothesis = claim
    flat_premises: List[str] = []
    flat_hypotheses: List[str] = []
    claim_pair_slices: List[Tuple[int, int]] = []

    current_idx = 0
    for i, claim in enumerate(claims):
        top_idx = claim_top_indices[i]
        start_idx = current_idx
        for p_idx in top_idx:
            flat_premises.append(passages[p_idx])
            flat_hypotheses.append(claim)
            current_idx += 1
        claim_pair_slices.append((start_idx, current_idx))

    # Step 4: Run Cross-Encoder Inference
    # Check for injected mock/stub model for testing
    if model is not None and hasattr(model, "predict_proba"):
        # Supports unit testing without torch/transformers
        try:
            raw_probs = model.predict_proba(flat_premises, flat_hypotheses)
        except TypeError:
            raw_probs = model.predict_proba(list(zip(flat_premises, flat_hypotheses)))
        
        raw_probs = np.asarray(raw_probs)
        if raw_probs.ndim == 2:
            p_scores = raw_probs[:, 1].tolist()
        else:
            p_scores = raw_probs.tolist()

    elif model is not None and not hasattr(model, "predict_proba") and tokenizer is None:
        # Stub callable
        p_scores = [float(model(p, h)) for p, h in zip(flat_premises, flat_hypotheses)]

    else:
        # Standard model path: load cross-encoder
        active_model = model
        active_tokenizer = tokenizer
        device = None

        if active_model is None or active_tokenizer is None:
            active_model, active_tokenizer, device = load_cross_encoder()
        else:
            import torch
            device = next(active_model.parameters()).device

        import torch

        # Batched tokenization with truncation only on premise side
        encoded = active_tokenizer(
            flat_premises,
            flat_hypotheses,
            padding=True,
            truncation="only_first",
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: val.to(device) for key, val in encoded.items()}

        with torch.no_grad():
            outputs = active_model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            p_scores = probs.tolist()

    # Step 5: Max pooling over top-k passages for each claim
    results: List[Dict[str, Any]] = []
    for i, claim in enumerate(claims):
        start_idx, end_idx = claim_pair_slices[i]
        claim_scores = p_scores[start_idx:end_idx]
        top_idx = claim_top_indices[i]

        if not claim_scores:
            results.append({
                "claim": claim,
                "p_supported": 0.0,
                "best_passage_index": -1,
                "top1_cosine": claim_top1_cos[i],
            })
            continue

        best_sub_idx = int(np.argmax(claim_scores))
        max_p = float(claim_scores[best_sub_idx])
        best_passage_idx = int(top_idx[best_sub_idx])

        results.append({
            "claim": claim,
            "p_supported": max_p,
            "best_passage_index": best_passage_idx,
            "top1_cosine": claim_top1_cos[i],
        })

    return results
