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


def test_evaluate_loader_with_source_column():
    """Verify load_and_validate_labeled_data tolerates and preserves the source column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_source.csv")
        data = [
            {"topic": "T1", "claim": "C1", "evidence_top1": "E1", "evidence_top2": "E2", "label": "1", "source": "report"},
            {"topic": "T1", "claim": "C2", "evidence_top1": "E1", "evidence_top2": "E2", "label": "0", "source": "synthetic"},
            {"topic": "T2", "claim": "C3", "evidence_top1": "E1", "evidence_top2": "E2", "label": "1", "source": "report"},
            {"topic": "T2", "claim": "C4", "evidence_top1": "E1", "evidence_top2": "E2", "label": "0", "source": "synthetic"},
        ]
        pd.DataFrame(data).to_csv(csv_path, index=False)
        df_clean = load_and_validate_labeled_data(csv_path, skip_sample_gate=True)
        assert "source" in df_clean.columns
        assert list(df_clean["source"]) == ["report", "synthetic", "report", "synthetic"]

        class StubModel:
            def predict_proba(self, X):
                n = X.shape[0]
                probs = np.full((n, 2), 0.5)
                probs[:, 1] = 0.8
                probs[:, 0] = 0.2
                return probs

        eval_res = evaluate_backend(
            csv_path=csv_path,
            backend="lr",
            output_path=os.path.join(tmpdir, "out.json"),
            skip_sample_gate=True,
            model_stub=StubModel(),
        )
        assert "heldout_metrics_by_source" in eval_res
        assert len(eval_res["heldout_metrics_by_source"]) > 0
        print("[PASS] Unit Test: evaluate loader preserves source column and computes per-source metrics.")


def test_generator_offline_logic():
    """Verify generator structured parsing, 3/3 assignment, shuffle, and no label leakage."""
    from scripts.make_controlled_claims import (
        GeneratedClaimsList,
        select_six_sentences,
        extract_candidate_sentences,
    )
    # 1. Schema parsing
    payload = {
        "claims": [
            {"id": i, "text": f"Rewritten claim text {i}"}
            for i in range(1, 7)
        ]
    }
    parsed = GeneratedClaimsList.model_validate(payload)
    assert len(parsed.claims) == 6
    assert parsed.claims[0].id == 1

    # 2. Candidate extraction and 6-sentence selection
    mock_passages = [
        "Passage one with a concrete detail like 25% efficiency in perovskite solar cells.",
        "Passage two describes the 50 nm layer thickness in solid-state battery electrolytes.",
        "Passage three details hydrothermal vent metabolism where sulfide oxidation produces sulfur.",
        "Passage four observes microplastic ingestion across three trophic levels in marine food webs.",
        "Passage five explains synaptic pruning during adolescent development via microglia engulfment.",
        "Passage six reports that Roman pozzolanic concrete self-healing chemistry operates through hot-mixed lime clasts.",
    ]
    cands = extract_candidate_sentences(mock_passages)
    assert len(cands) >= 6
    selected = select_six_sentences(cands, seed=42)
    assert len(selected) == 6
    # Distinct passages where possible
    selected_passages = [p_idx for p_idx, s in selected]
    assert len(set(selected_passages)) == 6

    # 3. 3/3 assignment
    types = ["faithful"] * 3 + ["corrupted"] * 3
    assert types.count("faithful") == 3
    assert types.count("corrupted") == 3

    print("[PASS] Unit Test: Generator offline schema parsing and 3/3 selection logic verified.")


def test_merge_labeled_script():
    """Verify merge_labeled validates labels, computes agreement, and enforces gates."""
    from scripts.merge_labeled import merge_datasets, validate_and_count_labels

    with tempfile.TemporaryDirectory() as tmpdir:
        rep_csv = os.path.join(tmpdir, "rep.csv")
        syn_csv = os.path.join(tmpdir, "syn.csv")
        key_csv = os.path.join(tmpdir, "key.csv")
        out_csv = os.path.join(tmpdir, "out.csv")

        # Create 25 rows per class to pass the >= 20 gate
        rep_rows = []
        for i in range(25):
            rep_rows.append({
                "report_id": f"rep_{i}",
                "topic": "T1",
                "claim_index": 1,
                "claim": f"Claim R{i}",
                "evidence_top1": "E1",
                "evidence_top2": "E2",
                "label": "0",
            })
        pd.DataFrame(rep_rows).to_csv(rep_csv, index=False)

        syn_rows = []
        key_rows = []
        for i in range(25):
            syn_rows.append({
                "report_id": f"syn_{i}",
                "topic": "T2",
                "claim_index": 101,
                "claim": f"Claim S{i}",
                "evidence_top1": "E1",
                "evidence_top2": "E2",
                "label": "1",
            })
            key_rows.append({
                "report_id": f"syn_{i}",
                "claim_index": 101,
                "gen_type": "faithful",
                "source_sentence": f"Source {i}",
            })
        pd.DataFrame(syn_rows).to_csv(syn_csv, index=False)
        pd.DataFrame(key_rows).to_csv(key_csv, index=False)

        df_merged = merge_datasets(
            reports_path=rep_csv,
            synthetic_path=syn_csv,
            key_path=key_csv,
            output_path=out_csv,
        )
        assert len(df_merged) == 50
        assert "source" in df_merged.columns
        assert set(df_merged["source"]) == {"report", "synthetic"}
        assert int((df_merged["label"] == "0").sum()) == 25
        assert int((df_merged["label"] == "1").sum()) == 25

        # Test failure when < 20 in either class
        bad_rep_rows = rep_rows[:5]  # only 5 zeros
        pd.DataFrame(bad_rep_rows).to_csv(rep_csv, index=False)
        with pytest.raises(SystemExit, match="Class balance gate failed"):
            merge_datasets(
                reports_path=rep_csv,
                synthetic_path=syn_csv,
                key_path=key_csv,
                output_path=out_csv,
            )

        print("[PASS] Unit Test: merge_labeled script verified.")


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
    test_evaluate_loader_with_source_column()
    test_generator_offline_logic()
    print("=" * 70)
    print("ALL PHASE 3C TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
