"""Unit tests for offline analysis script (scripts/analyze_results.py).

Tests:
- Bootstrap confidence interval calculations and seed reproducibility.
- Holm-Bonferroni p-value step-down correction and monotonicity.
- End-to-end generation of all 5 CSV tables and README from synthetic fixtures.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from scripts.analyze_results import (
    holm_bonferroni_correction,
    bootstrap_ci,
    run_analysis,
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
                {"attempt": 0, "critic_score": 7.0 if retry_count > 0 else critic_score},
            ]
            if retry_count > 0:
                attempts.append({"attempt": 1, "critic_score": 8.5})

            eval_record = {
                "qid": qid,
                "system": s,
                "topic": f"Topic for {qid}",
                "retry_count": retry_count,
                "final_score": critic_score,
                "attempts": attempts,
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

    # Run analysis
    run_analysis(str(eval_dir), str(ragas_dir), str(out_dir))

    # Assert all expected artifacts exist
    expected_files = [
        "per_system.csv",
        "paired.csv",
        "retry_effect.csv",
        "judge_disagreement.csv",
        "cost.csv",
        "README.md",
    ]
    for filename in expected_files:
        filepath = out_dir / filename
        assert filepath.exists(), f"Missing artifact: {filename}"
        assert filepath.stat().st_size > 0

    # Verify per_system.csv content
    df_sys = pd.read_csv(out_dir / "per_system.csv")
    assert len(df_sys) == 4
    assert set(df_sys["system_code"]) == {"S1", "S2", "S3", "S4"}
    assert df_sys[df_sys["system_name"] == "fixed"]["faithfulness_mean"].iloc[0] == pytest.approx(0.20)

    # Verify paired.csv content
    df_paired = pd.read_csv(out_dir / "paired.csv")
    assert len(df_paired) == 6  # 3 pairs x 2 metrics
    assert "wilcoxon_p_holm" in df_paired.columns
    assert "ci_95_lower" in df_paired.columns

    # Verify retry_effect.csv content
    df_retry = pd.read_csv(out_dir / "retry_effect.csv")
    assert len(df_retry) == 2  # Q02 adaptive_llm and Q02 adaptive_classifier

    # Verify cost.csv content
    df_cost = pd.read_csv(out_dir / "cost.csv")
    assert len(df_cost) == 4
