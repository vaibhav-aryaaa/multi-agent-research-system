#!/usr/bin/env python3
"""Analysis and Statistical Comparison Script for Multi-Agent Researcher.

Produces outputs in results/analysis/:
1. per_system.csv: Mean, median, std, N, nulls of Faithfulness & Answer Relevancy,
   plus mean critic score, mean retry count, and total token usage.
2. paired.csv: Paired differences for S2->S3, S2->S4, S3->S4 with bootstrap 95% CI
   (10,000 resamples, seed 0), Wilcoxon signed-rank p-value, Holm-Bonferroni correction,
   and win/tie/loss counts.
3. retry_effect.csv: Effect of retries on critic score and faithfulness vs S2.
4. judge_disagreement.csv: Disagreement between LLM and classifier judges (retries vs faithfulness).
5. cost.csv: Token consumption and token efficiency per +0.1 faithfulness gain vs S2.
6. README.md: Plain-English summary of table contents and sample counts.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

SYSTEM_MAP = {
    "baseline": "S1",
    "fixed": "S2",
    "adaptive_llm": "S3",
    "adaptive_classifier": "S4",
}
REV_SYSTEM_MAP = {v: k for k, v in SYSTEM_MAP.items()}


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values.
    
    Guarantees monotonicity: p_adj[i] <= p_adj[j] for p_val[i] <= p_val[j].
    """
    m = len(p_values)
    if m == 0:
        return []
    
    # Sort indices by p-value
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    
    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        mult = m - rank
        adj = min(1.0, mult * p_val)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = cum_max
        
    return adjusted


def bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap percentile confidence interval for the mean difference."""
    if len(diffs) == 0:
        return (np.nan, np.nan)
    if len(diffs) == 1:
        return (float(diffs[0]), float(diffs[0]))
    
    rng = np.random.default_rng(seed)
    # Generate resampled means
    boot_indices = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper


def load_eval_data(eval_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load results/eval JSON files into dict keyed by (qid, system)."""
    data = {}
    pattern = os.path.join(eval_dir, "Q*_*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        if filename.endswith("_ERROR.json") or "first_pass" in filename:
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                item = json.load(f)
            qid = item.get("qid")
            system = item.get("system")
            if qid and system:
                data[(qid, system)] = item
        except Exception:
            pass
    return data


def load_ragas_data(ragas_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load results/ragas JSON files into dict keyed by (qid, system)."""
    data = {}
    pattern = os.path.join(ragas_dir, "Q*_*.json")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        if filename.endswith("_ERROR.json"):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                item = json.load(f)
            qid = item.get("qid")
            system = item.get("system")
            if qid and system:
                data[(qid, system)] = item
        except Exception:
            pass
    return data


def generate_per_system_csv(
    eval_data: Dict[str, Dict[str, Any]],
    ragas_data: Dict[str, Dict[str, Any]],
    out_path: str,
) -> pd.DataFrame:
    """Generate per_system.csv."""
    rows = []
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]

    for sys_name in systems:
        sys_code = SYSTEM_MAP[sys_name]
        f_vals = []
        r_vals = []
        f_nulls = 0
        r_nulls = 0
        critic_scores = []
        retry_counts = []
        total_tokens_list = []
        
        qids = sorted(list({q for q, s in eval_data.keys() if s == sys_name}))
        
        for qid in qids:
            # Eval data
            ed = eval_data.get((qid, sys_name), {})
            retry_counts.append(ed.get("retry_count", 0))
            
            c_score = ed.get("final_score")
            if c_score is not None:
                critic_scores.append(float(c_score))
                
            tu = ed.get("token_usage", {})
            total_tokens_list.append(tu.get("total_tokens", 0))
            
            # RAGAS data
            rd = ragas_data.get((qid, sys_name), {})
            scores = rd.get("scores", {})
            
            f_score = scores.get("faithfulness")
            if f_score is not None and not (isinstance(f_score, float) and np.isnan(f_score)):
                f_vals.append(float(f_score))
            else:
                f_nulls += 1
                
            ar_score = scores.get("answer_relevancy")
            if ar_score is not None and not (isinstance(ar_score, float) and np.isnan(ar_score)):
                r_vals.append(float(ar_score))
            else:
                r_nulls += 1
                
        row = {
            "system_code": sys_code,
            "system_name": sys_name,
            "N": len(qids),
            "faithfulness_mean": round(float(np.mean(f_vals)), 4) if f_vals else np.nan,
            "faithfulness_median": round(float(np.median(f_vals)), 4) if f_vals else np.nan,
            "faithfulness_std": round(float(np.std(f_vals, ddof=1)), 4) if len(f_vals) > 1 else np.nan,
            "faithfulness_nulls": f_nulls,
            "answer_relevancy_mean": round(float(np.mean(r_vals)), 4) if r_vals else np.nan,
            "answer_relevancy_median": round(float(np.median(r_vals)), 4) if r_vals else np.nan,
            "answer_relevancy_std": round(float(np.std(r_vals, ddof=1)), 4) if len(r_vals) > 1 else np.nan,
            "answer_relevancy_nulls": r_nulls,
            "mean_critic_score": round(float(np.mean(critic_scores)), 4) if critic_scores else np.nan,
            "mean_retry_count": round(float(np.mean(retry_counts)), 4) if retry_counts else np.nan,
            "total_tokens": int(sum(total_tokens_list)),
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def generate_paired_csv(
    ragas_data: Dict[str, Dict[str, Any]],
    out_path: str,
) -> pd.DataFrame:
    """Generate paired.csv with bootstrap 95% CI and Holm-corrected Wilcoxon tests."""
    pairs = [
        ("fixed", "adaptive_llm", "S2->S3"),
        ("fixed", "adaptive_classifier", "S2->S4"),
        ("adaptive_llm", "adaptive_classifier", "S3->S4"),
    ]
    metrics = ["faithfulness", "answer_relevancy"]
    
    all_qids = sorted(list({q for q, _ in ragas_data.keys()}))
    
    comparisons = []
    raw_p_values = []
    
    for sys_a, sys_b, pair_label in pairs:
        for metric in metrics:
            diffs = []
            nulls_dropped = 0
            wins = 0
            ties = 0
            losses = 0
            
            for qid in all_qids:
                ra = ragas_data.get((qid, sys_a), {})
                rb = ragas_data.get((qid, sys_b), {})
                
                va = ra.get("scores", {}).get(metric)
                vb = rb.get("scores", {}).get(metric)
                
                # Check null in either
                if va is None or vb is None or np.isnan(va) or np.isnan(vb):
                    nulls_dropped += 1
                    continue
                
                diff = float(vb) - float(va)
                diffs.append(diff)
                
                if diff > 1e-6:
                    wins += 1
                elif diff < -1e-6:
                    losses += 1
                else:
                    ties += 1
                    
            diffs_arr = np.array(diffs, dtype=float)
            mean_diff = float(np.mean(diffs_arr)) if len(diffs_arr) > 0 else np.nan
            ci_lower, ci_upper = bootstrap_ci(diffs_arr, n_resamples=10000, seed=0)
            
            # Wilcoxon signed-rank test dropping zero-differences
            non_zero_diffs = [d for d in diffs if abs(d) > 1e-9]
            n_zero_diff_dropped = len(diffs) - len(non_zero_diffs)
            
            if len(non_zero_diffs) < 2:
                # Not enough variation for Wilcoxon
                p_val = 1.0
            else:
                try:
                    res = stats.wilcoxon(non_zero_diffs, alternative="two-sided")
                    p_val = float(res.pvalue)
                except Exception:
                    p_val = 1.0
                    
            raw_p_values.append(p_val)
            
            comp_record = {
                "pair": pair_label,
                "metric": metric,
                "n_initial": len(all_qids),
                "n_dropped_null": nulls_dropped,
                "n_eval": len(diffs_arr),
                "mean_diff": round(mean_diff, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "n_zero_diff_dropped": n_zero_diff_dropped,
                "wilcoxon_p_raw": p_val,
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
            comparisons.append(comp_record)
            
    # Apply Holm correction across all 6 comparisons
    holm_p_values = holm_bonferroni_correction(raw_p_values)
    for comp, holm_p in zip(comparisons, holm_p_values):
        comp["wilcoxon_p_holm"] = round(holm_p, 5)
        comp["wilcoxon_p_raw"] = round(comp["wilcoxon_p_raw"], 5)
        
    df = pd.DataFrame(comparisons)
    df.to_csv(out_path, index=False)
    return df


def generate_retry_effect_csv(
    eval_data: Dict[str, Dict[str, Any]],
    ragas_data: Dict[str, Dict[str, Any]],
    out_path: str,
) -> pd.DataFrame:
    """Generate retry_effect.csv for questions where retries actually occurred."""
    rows = []
    qids = sorted(list({q for q, _ in eval_data.keys()}))
    
    for qid in qids:
        # S2 baseline for comparison
        s2_ragas = ragas_data.get((qid, "fixed"), {})
        s2_faith = s2_ragas.get("scores", {}).get("faithfulness")
        
        for sys_name in ["adaptive_llm", "adaptive_classifier"]:
            ed = eval_data.get((qid, sys_name), {})
            retry_count = ed.get("retry_count", 0)
            if retry_count > 0:
                attempts = ed.get("attempts", [])
                score_before = attempts[0].get("critic_score") if attempts else np.nan
                score_after = ed.get("final_score")
                
                score_diff = (score_after - score_before) if (score_after is not None and score_before is not None) else np.nan
                
                rd = ragas_data.get((qid, sys_name), {})
                this_faith = rd.get("scores", {}).get("faithfulness")
                
                faith_diff = (this_faith - s2_faith) if (this_faith is not None and s2_faith is not None) else np.nan
                
                rows.append({
                    "qid": qid,
                    "system": sys_name,
                    "system_code": SYSTEM_MAP[sys_name],
                    "retry_count": retry_count,
                    "critic_score_before": round(float(score_before), 3) if score_before is not None else np.nan,
                    "critic_score_after": round(float(score_after), 3) if score_after is not None else np.nan,
                    "critic_score_diff": round(float(score_diff), 3) if score_diff is not None else np.nan,
                    "faithfulness_s2": round(float(s2_faith), 4) if s2_faith is not None else np.nan,
                    "faithfulness_this": round(float(this_faith), 4) if this_faith is not None else np.nan,
                    "faithfulness_diff": round(float(faith_diff), 4) if faith_diff is not None else np.nan,
                })
                
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def generate_judge_disagreement_csv(
    eval_data: Dict[str, Dict[str, Any]],
    ragas_data: Dict[str, Dict[str, Any]],
    out_path: str,
) -> pd.DataFrame:
    """Generate judge_disagreement.csv comparing S3 vs S4 retry decisions and faithfulness."""
    rows = []
    qids = sorted(list({q for q, _ in eval_data.keys()}))
    
    for qid in qids:
        ed_s3 = eval_data.get((qid, "adaptive_llm"), {})
        ed_s4 = eval_data.get((qid, "adaptive_classifier"), {})
        
        s3_retries = ed_s3.get("retry_count", 0)
        s4_retries = ed_s4.get("retry_count", 0)
        
        rd_s3 = ragas_data.get((qid, "adaptive_llm"), {})
        rd_s4 = ragas_data.get((qid, "adaptive_classifier"), {})
        
        s3_faith = rd_s3.get("scores", {}).get("faithfulness")
        s4_faith = rd_s4.get("scores", {}).get("faithfulness")
        
        if s3_faith is None or s4_faith is None:
            higher = "null"
        elif s3_faith > s4_faith + 1e-6:
            higher = "S3 (adaptive_llm)"
        elif s4_faith > s3_faith + 1e-6:
            higher = "S4 (adaptive_classifier)"
        else:
            higher = "tie"
            
        rows.append({
            "qid": qid,
            "s3_retries": s3_retries,
            "s4_retries": s4_retries,
            "retries_differ": s3_retries != s4_retries,
            "s3_faithfulness": round(float(s3_faith), 4) if s3_faith is not None else np.nan,
            "s4_faithfulness": round(float(s4_faith), 4) if s4_faith is not None else np.nan,
            "higher_faithfulness": higher,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def generate_cost_csv(
    eval_data: Dict[str, Dict[str, Any]],
    ragas_data: Dict[str, Dict[str, Any]],
    out_path: str,
) -> pd.DataFrame:
    """Generate cost.csv: token consumption and tokens per +0.1 faithfulness gained vs S2."""
    rows = []
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]
    
    # Pre-calculate mean faithfulness and total tokens
    stats_dict = {}
    for sys_name in systems:
        qids = sorted(list({q for q, s in eval_data.keys() if s == sys_name}))
        tokens = [eval_data.get((q, sys_name), {}).get("token_usage", {}).get("total_tokens", 0) for q in qids]
        faith_vals = [
            ragas_data.get((q, sys_name), {}).get("scores", {}).get("faithfulness")
            for q in qids
        ]
        valid_faith = [float(f) for f in faith_vals if f is not None and not np.isnan(f)]
        stats_dict[sys_name] = {
            "n": len(qids),
            "total_tokens": sum(tokens),
            "mean_tokens": float(np.mean(tokens)) if tokens else 0.0,
            "mean_faith": float(np.mean(valid_faith)) if valid_faith else 0.0,
        }
        
    s2_total_tokens = stats_dict["fixed"]["total_tokens"]
    s2_mean_tokens = stats_dict["fixed"]["mean_tokens"]
    s2_mean_faith = stats_dict["fixed"]["mean_faith"]
    
    for sys_name in systems:
        st = stats_dict[sys_name]
        gain_vs_s2 = st["mean_faith"] - s2_mean_faith
        
        if sys_name == "fixed":
            efficiency = "baseline_s2"
        elif gain_vs_s2 > 1e-4:
            # Extra tokens per 0.1 gain: (extra_tokens) / (gain / 0.1)
            extra_tokens = st["total_tokens"] - s2_total_tokens
            cost_per_point_one = extra_tokens / (gain_vs_s2 / 0.1)
            efficiency = f"{round(cost_per_point_one, 1)}"
        else:
            efficiency = "N/A (gain <= 0)"
            
        rows.append({
            "system_code": SYSTEM_MAP[sys_name],
            "system_name": sys_name,
            "total_tokens": st["total_tokens"],
            "mean_tokens_per_question": round(st["mean_tokens"], 1),
            "mean_faithfulness": round(st["mean_faith"], 4),
            "faithfulness_gain_vs_s2": round(gain_vs_s2, 4),
            "tokens_per_plus_0_1_faithfulness": efficiency,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def generate_readme(
    out_dir: str,
    n_questions: int,
):
    """Generate descriptive README.md summarizing what each file shows and sample counts."""
    content = f"""# Analysis Results Summary (N = {n_questions} Questions)

This directory contains statistical evaluations and comparisons across the benchmark questions:

1. **per_system.csv**: Evaluates N={n_questions} questions per system (S1 Baseline, S2 Fixed, S3 Adaptive LLM, S4 Adaptive Classifier). Reports mean, median, standard deviation, and null counts for RAGAS Faithfulness and Answer Relevancy, alongside mean critic scores, mean retry counts, and total token usage.
2. **paired.csv**: Contains paired comparisons (S2->S3, S2->S4, S3->S4) across both metrics. Reports mean differences, 10,000-resample bootstrap 95% confidence intervals (seed 0), two-sided Wilcoxon signed-rank test p-values (dropping zero-difference pairs), step-down Holm-Bonferroni adjusted p-values across all 6 comparisons, and win/tie/loss counts.
3. **retry_effect.csv**: Analyzes questions where adaptive systems triggered retries (retry_count > 0). Shows pre-retry and post-retry critic scores alongside RAGAS Faithfulness compared against the shared attempt-0 S2 baseline.
4. **judge_disagreement.csv**: Documents per-question retry counts between S3 (LLM judge) and S4 (Cross-Encoder judge), identifying decision differences and which system achieved higher final Faithfulness.
5. **cost.csv**: Summarizes total token consumption and marginal token expenditure per +0.1 gain in RAGAS Faithfulness relative to S2 Fixed.
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def run_analysis(
    eval_dir: str = "results/eval",
    ragas_dir: str = "results/ragas",
    out_dir: str = "results/analysis",
):
    """Run all analysis pipelines and save CSV/Markdown artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    
    eval_data = load_eval_data(eval_dir)
    ragas_data = load_ragas_data(ragas_dir)
    
    qids = sorted(list({q for q, _ in eval_data.keys()}))
    n_questions = len(qids)
    
    print(f"[Analysis] Found {len(eval_data)} eval files and {len(ragas_data)} ragas files across {n_questions} questions.")
    
    generate_per_system_csv(eval_data, ragas_data, os.path.join(out_dir, "per_system.csv"))
    generate_paired_csv(ragas_data, os.path.join(out_dir, "paired.csv"))
    generate_retry_effect_csv(eval_data, ragas_data, os.path.join(out_dir, "retry_effect.csv"))
    generate_judge_disagreement_csv(eval_data, ragas_data, os.path.join(out_dir, "judge_disagreement.csv"))
    generate_cost_csv(eval_data, ragas_data, os.path.join(out_dir, "cost.csv"))
    generate_readme(out_dir, n_questions)
    
    # Run Task I Confound checks and figures
    try:
        from scripts.analyze_confounds import run_confound_analysis
    except ImportError:
        from analyze_confounds import run_confound_analysis
    run_confound_analysis(eval_dir=eval_dir, ragas_dir=ragas_dir, out_dir=out_dir)
    
    print(f"[Analysis] Successfully generated all analysis artifacts in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Evaluation and RAGAS Results")
    parser.add_argument("--eval-dir", default="results/eval", help="Path to results/eval")
    parser.add_argument("--ragas-dir", default="results/ragas", help="Path to results/ragas")
    parser.add_argument("--out-dir", default="results/analysis", help="Path to results/analysis")
    args = parser.parse_args()
    
    run_analysis(args.eval_dir, args.ragas_dir, args.out_dir)
