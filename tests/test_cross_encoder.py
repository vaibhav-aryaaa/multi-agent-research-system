"""Unit tests for DistilBERT cross-encoder faithfulness module (Phase 3B).

Runs completely offline with zero API calls, no GPU requirement, and injected
stubs to verify scoring logic and lazy import hygiene.
"""

import os
import sys
import subprocess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithfulness.cross_encoder import score_claims
from faithfulness.evaluate import optimize_tau


class FakePredictProbaModel:
    """Stub cross-encoder model that returns synthetic probabilities."""

    def __init__(self, mapping=None):
        # mapping: dict of (premise, hypothesis) -> p_supported
        self.mapping = mapping or {}
        self.call_history = []

    def predict_proba(self, premises, hypotheses):
        probs = []
        for p, h in zip(premises, hypotheses):
            self.call_history.append((p, h))
            p_sup = self.mapping.get((p, h), 0.5)
            # return [p_not_supported, p_supported]
            probs.append([1.0 - p_sup, p_sup])
        return np.array(probs, dtype=np.float32)


def test_lazy_imports():
    """Verify that importing faithfulness.cross_encoder does NOT import torch or transformers."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import faithfulness.cross_encoder; "
            "assert 'torch' not in sys.modules, f'torch was imported prematurely: {sys.modules.get(\"torch\")}'; "
            "assert 'transformers' not in sys.modules, f'transformers was imported prematurely: {sys.modules.get(\"transformers\")}'; "
            "print('Lazy import verified successfully.')"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Lazy import check failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"
    print("[PASS] Lazy import check: torch and transformers are NOT imported at module level.")


def test_score_claims_empty_passages():
    """When passages is empty, p_supported must be 0.0 for every claim."""
    claims = ["The sky is green.", "Quantum vacuum produces Casimir force."]
    passages = []
    results = score_claims(claims, passages, k=3)
    assert len(results) == 2
    for r in results:
        assert r["p_supported"] == 0.0
        assert r["best_passage_index"] == -1
        assert r["top1_cosine"] == 0.0
    print("[PASS] Unit test: score_claims returns 0.0 for empty passages.")


def test_score_claims_max_pooling_and_top_k():
    """Verify top-k candidate selection and max pooling over candidate passages."""
    claim = "Casimir force attracts neutral conducting plates."
    passages = [
        "Unrelated passage about cooking pasta in boiling salted water.",      # idx 0: low similarity
        "The Casimir force arises between two neutral plates in a vacuum.",    # idx 1: high similarity, p=0.85
        "Conducting surfaces experience quantum fluctuations.",                # idx 2: medium similarity, p=0.30
        "Another unrelated passage about gardening in summer.",                # idx 3: low similarity
    ]

    # Map candidate pairs to synthetic probabilities
    fake_probs = {
        (passages[1], claim): 0.85,
        (passages[2], claim): 0.30,
        (passages[0], claim): 0.10,
        (passages[3], claim): 0.05,
    }
    fake_model = FakePredictProbaModel(mapping=fake_probs)

    # With k=2, only top-2 most similar passages by MiniLM should be scored
    results = score_claims([claim], passages, k=2, model=fake_model)

    assert len(results) == 1
    res = results[0]
    assert res["claim"] == claim

    # Check that fake_model only scored at most 2 pairs
    assert len(fake_model.call_history) == 2, f"Expected 2 pairs scored, got: {fake_model.call_history}"

    # Verify max pooling: p_supported should be 0.85 from passage 1
    assert pytest.approx(res["p_supported"], abs=1e-4) == 0.85
    assert res["best_passage_index"] == 1
    assert res["top1_cosine"] > 0.0
    print("[PASS] Unit test: score_claims selects top-k and applies max pooling correctly.")


def test_optimize_tau():
    """Verify that optimize_tau picks threshold maximizing macro F1."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    # scores cleanly separating classes at 0.6
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    best_tau, best_f1 = optimize_tau(y_true, scores)
    assert 0.4 < best_tau < 0.7
    assert best_f1 == 1.0
    print("[PASS] Unit test: optimize_tau finds optimal decision boundary.")


def run_all():
    print("=" * 70)
    print("Running Phase 3B Cross-Encoder Offline Unit Tests")
    print("=" * 70)
    test_lazy_imports()
    test_score_claims_empty_passages()
    test_score_claims_max_pooling_and_top_k()
    test_optimize_tau()
    print("=" * 70)
    print("ALL PHASE 3B TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
