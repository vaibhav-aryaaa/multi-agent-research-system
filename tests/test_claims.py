"""Unit tests for claim extraction and evaluation module compatibility (Phase 3C).

Runs 100% offline with zero API calls or downloads.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faithfulness.claims import extract_claims, extract_claims_with_stats
from faithfulness.evaluate import evaluate_backend, load_and_validate_labeled_data


def test_table_cells_are_separate_claims():
    """Table cells must be extracted as separate segments and never glued together."""
    table_text = (
        "# Title\n\n"
        "| # | Finding | Evidence & Reasoning |\n"
        "|---|---------|----------------------|\n"
        "| 1 | All mechanical cycles violate conservation laws and require work | Forward proposed extracting energy until plates touch. A reset requires external work. |\n"
    )
    claims, stats = extract_claims_with_stats(table_text)
    # The finding cell and the first sentence of the evidence cell must NOT be glued together
    assert any(c == "All mechanical cycles violate conservation laws and require work." for c in claims), f"Missing finding claim: {claims}"
    assert any("Forward proposed extracting energy until plates touch." in c for c in claims), f"Missing evidence claim: {claims}"
    assert not any("require work Forward proposed" in c for c in claims), f"Glued table cells found: {claims}"
    print("[PASS] Unit Test: Table cells are extracted as separate claims.")


def test_bullet_item_not_glued_to_preceding_sentence():
    """A bullet item following a colon or sentence must be its own claim, not glued."""
    bullet_text = (
        "Over the past two decades several concrete proposals have been published, most notably:\n"
        "* Forward's 'Casimir-spring' engine - plates are allowed to move together, extracting potential energy.\n"
        "* Pinto's switchable-mirror cycle - the reflectivity of the cavity walls is altered periodically.\n"
    )
    claims, stats = extract_claims_with_stats(bullet_text)
    # Bullet claims must be extracted cleanly
    assert any("Forward's 'Casimir-spring' engine" in c for c in claims), f"Bullet 1 not extracted: {claims}"
    assert any("Pinto's switchable-mirror cycle" in c for c in claims), f"Bullet 2 not extracted: {claims}"
    # Must NOT be glued to "most notably:"
    assert not any("most notably: Forward" in c or "most notably Forward" in c for c in claims), f"Glued bullet found: {claims}"
    print("[PASS] Unit Test: Bullet items are not glued to preceding sentences.")


def test_sentence_with_formula_dropped():
    """Sentences with LaTeX formulas (\\[...\\], \\(...\\)) must be dropped entirely and counted."""
    latex_text = (
        "The attractive Casimir force between two plates is \\[F = -\\dfrac{\\pi^2 \\hbar c}{240 d^4} A\\] (the standard Casimir formula).\n"
        "Integrating this force gives a finite potential energy \\(U(d) = -\\dfrac{\\pi^2 \\hbar c}{720 d^3} A\\).\n"
        "This experimental observation confirms that the vacuum energy spectrum is modified by boundary conditions.\n"
    )
    claims, stats = extract_claims_with_stats(latex_text)
    # The two formula sentences must be dropped
    assert stats["dropped_latex"] >= 2, f"Expected at least 2 dropped formulas, got: {stats}"
    assert not any("standard Casimir formula" in c for c in claims), f"Formula sentence 1 leaked into claims: {claims}"
    assert not any("finite potential energy" in c for c in claims), f"Formula sentence 2 leaked into claims: {claims}"
    # The non-formula claim sentence must remain
    assert any("vacuum energy spectrum is modified" in c for c in claims), f"Factual claim dropped: {claims}"
    print("[PASS] Unit Test: Sentences with LaTeX formulas are dropped and counted in stats.")


def test_meta_sentences_and_questions_dropped():
    """Meta-sentences about the report/aim and questions must be dropped and counted."""
    meta_text = (
        "This report surveys the principal proposals and evaluates the underlying physics.\n"
        "The aim is to answer the central question: Can a Casimir cavity be used to extract perpetual mechanical energy?\n"
        "Can a mechanical reset step operate with zero external energy dissipation?\n"
        "Overall, fundamental thermodynamic analyses have repeatedly highlighted apparent violations.\n"
        "The Casimir effect produces a measurable attractive force between conducting plates placed nanometers apart.\n"
    )
    claims, stats = extract_claims_with_stats(meta_text)
    assert stats["dropped_non_claims"] >= 3, f"Expected dropped non-claims, got: {stats}"
    assert not any(c.lower().startswith("this report") for c in claims)
    assert not any(c.lower().startswith("the aim") for c in claims)
    assert not any(c.endswith("?") for c in claims)
    assert not any(c.lower().startswith("overall,") for c in claims)
    assert any("Casimir effect produces a measurable attractive force" in c for c in claims)
    print("[PASS] Unit Test: Meta sentences and questions are dropped and counted.")


def test_target_unicode_sentence_still_extracted():
    """The target Casimir energy claim with Unicode characters must still be extracted."""
    sample = (
        "The energy cost of the transition has been measured in thin-film VO₂ and graphene devices "
        "and is ≥ 10⁻¹⁸ J per µm², which exceeds the Casimir energy gain for plate separations > 10 nm."
    )
    claims = extract_claims(sample)
    assert len(claims) == 1, f"Expected 1 claim, got: {claims}"
    assert "≥ 10⁻¹⁸ J per µm²" in claims[0]
    assert "exceeds the Casimir energy gain" in claims[0]
    print("[PASS] Unit Test: Target Unicode sentence (≥ 10⁻¹⁸ J per µm²) is extracted cleanly.")


def test_evaluate_label_rules_and_split():
    """Test evaluate.py with Part D format, strict label validation, dropping blank/?, and topic-grouped split."""
    # Create synthetic DataFrame with 4 distinct topics
    # 20 samples: 10 ones, 10 zeros, plus 2 blank/?, plus 1 invalid row
    topics = ["Casimir Engine", "Perovskite Solar", "CRISPR Editing", "Solid-State Battery"]
    data = []
    for t_idx, topic in enumerate(topics):
        # 5 rows per topic (10 total ones and 10 total zeros across 4 topics)
        # Topic 0: 3 ones, 2 zeros
        # Topic 1: 2 ones, 3 zeros
        # Topic 2: 3 ones, 2 zeros
        # Topic 3: 2 ones, 3 zeros
        n_ones = 3 if t_idx % 2 == 0 else 2
        for i in range(5):
            is_one = i < n_ones
            data.append({
                "topic": topic,
                "claim": f"Claim {t_idx}_{i} stating research facts for {topic}.",
                "evidence_top1": f"Top evidence 1 passage for claim {t_idx}_{i}.",
                "evidence_top2": f"Top evidence 2 passage for claim {t_idx}_{i}.",
                "label": "1" if is_one else "0",
            })
    # Add unannotated rows (blank and ?)
    data.append({
        "topic": "Casimir Engine",
        "claim": "Unlabeled claim with question mark label.",
        "evidence_top1": "Evidence 1",
        "evidence_top2": "Evidence 2",
        "label": "?",
    })
    data.append({
        "topic": "Casimir Engine",
        "claim": "Unlabeled claim with empty string label.",
        "evidence_top1": "Evidence 1",
        "evidence_top2": "Evidence 2",
        "label": "",
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_labels.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)

        # 1. Test load_and_validate_labeled_data drops blank & ?
        df_clean = load_and_validate_labeled_data(csv_path, skip_sample_gate=True)
        assert len(df_clean) == 20, f"Expected 20 usable rows after dropping '?' and '', got: {len(df_clean)}"
        assert list(df_clean["label"].value_counts().values) == [10, 10]

        # 2. Test invalid label raises ValueError naming the row
        invalid_data = list(data) + [{
            "topic": "Casimir Engine",
            "claim": "Claim with bad label.",
            "evidence_top1": "E1",
            "evidence_top2": "E2",
            "label": "unclear",
        }]
        bad_csv = os.path.join(tmpdir, "bad_labels.csv")
        pd.DataFrame(invalid_data).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Invalid label 'unclear' at row 24"):
            load_and_validate_labeled_data(bad_csv, skip_sample_gate=True)

        # 3. Test evaluate_backend runs end-to-end with stub scorer and topic grouping
        class StubModel:
            def predict_proba(self, X):
                # Return synthetic probabilities based on feature shape
                n = X.shape[0]
                probs = np.full((n, 2), 0.5)
                probs[:, 1] = 0.7
                probs[:, 0] = 0.3
                return probs

        eval_res = evaluate_backend(
            csv_path=csv_path,
            backend="lr",
            output_path=os.path.join(tmpdir, "eval_out.json"),
            skip_sample_gate=True,
            model_stub=StubModel(),
        )
        assert eval_res["n_total"] == 20
        assert eval_res["n_dev"] == 10
        assert eval_res["n_heldout"] == 10
        assert set(eval_res["dev_topics"]).isdisjoint(set(eval_res["heldout_topics"])), "Dev and heldout topics must be disjoint!"
        assert "accuracy" in eval_res["heldout_metrics"]
        assert "always_supported" in eval_res["baselines_heldout"]
        assert "cosine_threshold" in eval_res["baselines_heldout"]
        print("[PASS] Unit Test: evaluate.py validates labels, drops ?, and splits dev/heldout by topic.")


def run_all():
    print("=" * 70)
    print("Running Phase 3C Claim Extraction & Evaluation Compatibility Tests")
    print("=" * 70)
    test_table_cells_are_separate_claims()
    test_bullet_item_not_glued_to_preceding_sentence()
    test_sentence_with_formula_dropped()
    test_meta_sentences_and_questions_dropped()
    test_target_unicode_sentence_still_extracted()
    test_evaluate_label_rules_and_split()
    print("=" * 70)
    print("ALL PHASE 3C TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
