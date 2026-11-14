"""Unit tests for offline analysis script and confound checks (Task I).

Tests:
- Bootstrap confidence interval calculations and seed reproducibility.
- Holm-Bonferroni p-value step-down correction and monotonicity.
- End-to-end generation of all core analysis tables, confound checks, and figures.
- Dose-response, matched comparison, and judge agreement logic.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from scripts.analyze_results import (
    holm_bonferroni_correction,
    bootstrap_ci,
    run_analysis,
)
from scripts.analyze_confounds import (
    run_dose_response,
    run_matched_comparison,
    run_judge_agreement,
    run_confound_analysis,
)


def test_holm_bonferroni_correction():
    # Test empty
    assert holm_bonferroni_correction([]) == []

    # Test single
    assert holm_bonferroni_correction([0.02]) == [0.02]

    # Test standard case: 3 tests
    # sorted: p1=0.01 (m=3 -> 0.03), p2=0.04 (m=2 -> 0.08), p3=0.05 (m=1 -> 0.05 -> monotonic cap 0.08)
    p_vals = [0.04, 0.01, 0.05]
    adj = holm_bonferroni_correction(p_vals)
    assert len(adj) == 3
    assert adj[1] == pytest.approx(0.03)  # p=0.01 * 3
    assert adj[0] == pytest.approx(0.08)  # p=0.04 * 2
    assert adj[2] == pytest.approx(0.08)  # p=0.05 * 1, but max(0.05, 0.08) = 0.08

    # Test capping at 1.0
    assert holm_bonferroni_correction([0.8, 0.9]) == [1.0, 1.0]


def test_bootstrap_ci():
    # Reproducible seed 0
    diffs = np.array([0.1, 0.2, 0.15, -0.05, 0.3, 0.25, 0.1])
    lower1, upper1 = bootstrap_ci(diffs, n_resamples=1000, seed=0)
    lower2, upper2 = bootstrap_ci(diffs, n_resamples=1000, seed=0)

    assert lower1 == pytest.approx(lower2)
    assert upper1 == pytest.approx(upper2)
    assert lower1 < np.mean(diffs) < upper1


def test_analysis_end_to_end_synthetic(tmp_path):
    eval_dir = tmp_path / "eval"
    ragas_dir = tmp_path / "ragas"
    out_dir = tmp_path / "analysis"
    eval_dir.mkdir()
    ragas_dir.mkdir()

    # Create synthetic questions Q01 and Q02
    qids = ["Q01", "Q02"]
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]

    for qid in qids:
        for s in systems:
            # Eval record
            retry_count = 1 if s in ["adaptive_llm", "adaptive_classifier"] and qid == "Q02" else 0
            critic_score = 8.5 if s != "baseline" else None
            attempts = [
                {
                    "attempt": 0,
                    "critic_score": 7.0 if retry_count > 0 else critic_score,
                    "decision": "writer" if retry_count > 0 else "end",
                    "judge_accept": (retry_count == 0),
                },
            ]
            if retry_count > 0:
                attempts.append({
                    "attempt": 1,
                    "critic_score": 8.5,
                    "decision": "end",
                    "judge_accept": True,
                })

            eval_record = {
                "qid": qid,
                "system": s,
                "topic": f"Topic for {qid}",
                "retry_count": retry_count,
                "final_score": critic_score,
                "attempts": attempts,
                "contexts": ["passage 1 content", "passage 2 content"],
                "token_usage": {"total_tokens": 1500 + (500 if retry_count > 0 else 0)},
            }
            with open(eval_dir / f"{qid}_{s}.json", "w") as f:
                json.dump(eval_record, f)

            # RAGAS record
            f_val = 0.20 if s == "fixed" else (0.25 if s == "adaptive_llm" else 0.22)
            ar_val = 0.90
            ragas_record = {
                "qid": qid,
                "system": s,
                "topic": f"Topic for {qid}",
                "scores": {
                    "faithfulness": f_val,
                    "answer_relevancy": ar_val,
                },
            }
            with open(ragas_dir / f"{qid}_{s}.json", "w") as f:
                json.dump(ragas_record, f)

    # Run full analysis (includes confound checks & figure generation)
    run_analysis(str(eval_dir), str(ragas_dir), str(out_dir))

    # Assert core artifacts exist
    expected_core_files = [
        "per_system.csv",
        "paired.csv",
        "retry_effect.csv",
        "judge_disagreement.csv",
        "cost.csv",
        "README.md",
    ]
    for filename in expected_core_files:
        filepath = out_dir / filename
        assert filepath.exists(), f"Missing core artifact: {filename}"
        assert filepath.stat().st_size > 0

    # Assert confound artifacts exist (Task I)
    expected_confound_files = [
        "dose_response.csv",
        "dose_response_correlations.csv",
        "matched_s3_s4.csv",
        "matched_s3_s4_diff_retries.csv",
        "matched_s3_s4_summary.csv",
        "judge_agreement.csv",
        "judge_agreement_summary.csv",
    ]
    for filename in expected_confound_files:
        filepath = out_dir / filename
        assert filepath.exists(), f"Missing confound artifact: {filename}"
        assert filepath.stat().st_size > 0

    # Assert all 5 figure files exist in figures/
    expected_figures = [
        "fig_faithfulness_by_system.png",
        "fig_relevancy_by_system.png",
        "fig_paired_faithfulness.png",
        "fig_cost_vs_faithfulness.png",
        "fig_retries_vs_gain.png",
    ]
    fig_dir = out_dir / "figures"
    assert fig_dir.exists(), "Figures directory was not created"
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 1000

    # Verify dose_response content
    df_dr = pd.read_csv(out_dir / "dose_response.csv")
    assert len(df_dr) == 6  # 2 questions x 3 systems (S2, S3, S4)
    assert set(df_dr.columns).issuperset({"qid", "system", "retry_count", "n_contexts", "context_characters", "faithfulness"})

    # Verify matched_s3_s4 content
    df_matched = pd.read_csv(out_dir / "matched_s3_s4.csv")
    assert len(df_matched) == 2  # Both Q01 (0 retries) and Q02 (1 retry) match in this fixture

    # Verify judge_agreement content
    df_ja = pd.read_csv(out_dir / "judge_agreement.csv")
    assert len(df_ja) == 2
    assert "agreement_category" in df_ja.columns
