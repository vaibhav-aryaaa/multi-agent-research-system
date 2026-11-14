#!/usr/bin/env python3
"""Confound Analysis and Visualization Script (Task I).

Analyzes:
I1 — Dose-response: Relationship between retries, evidence volume, and faithfulness.
I2 — Matched comparison: S3 vs S4 on questions with identical retry counts, and split by retry differences.
I3 — Judge agreement: LLM judge vs Cross-Encoder classifier judge on attempt 0 decisions.
I4 — Figures: Publication-grade colorblind-safe matplotlib plots with N in titles.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats

# Configure headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colorblind-safe palette (Okabe & Ito)
CB_PALETTE = {
    "S1": "#56B4E9",  # Sky Blue (Baseline)
    "S2": "#0072B2",  # Blue (Fixed)
    "S3": "#E69F00",  # Orange (Adaptive LLM)
    "S4": "#009E73",  # Green (Adaptive Classifier)
    "neutral": "#999999",
    "dark": "#333333",
}

SYSTEM_LABELS = {
    "baseline": "S1 (Baseline)",
    "fixed": "S2 (Fixed)",
    "adaptive_llm": "S3 (Adaptive LLM)",
    "adaptive_classifier": "S4 (Adaptive Classifier)",
}


def bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap percentile confidence interval for the mean."""
    if len(diffs) == 0:
        return (np.nan, np.nan)
    if len(diffs) == 1:
        return (float(diffs[0]), float(diffs[0]))
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper


def load_all_data(eval_dir: str, ragas_dir: str):
    """Load eval and ragas data keyed by (qid, system)."""
    eval_data = {}
    for f in glob.glob(os.path.join(eval_dir, "Q*.json")):
        if "first_pass" in f or "_ERROR" in f:
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            qid = d.get("qid")
            sys_name = d.get("system")
            if qid and sys_name:
                eval_data[(qid, sys_name)] = d
        except Exception:
            pass

    ragas_data = {}
    for f in glob.glob(os.path.join(ragas_dir, "Q*.json")):
        if "_ERROR" in f:
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            qid = d.get("qid")
            sys_name = d.get("system")
            if qid and sys_name:
                ragas_data[(qid, sys_name)] = d
        except Exception:
            pass

    qids = sorted(list({q for q, _ in eval_data.keys()}))
    return eval_data, ragas_data, qids


# ---------------------------------------------------------------------------
# I1 — Dose-Response Analysis
# ---------------------------------------------------------------------------

def run_dose_response(eval_data, ragas_data, qids, out_dir):
    """Generate dose_response.csv and dose_response_correlations.csv."""
    rows = []
    target_systems = [
        ("fixed", "S2"),
        ("adaptive_llm", "S3"),
        ("adaptive_classifier", "S4"),
    ]

    for sys_name, sys_code in target_systems:
        for qid in qids:
            ed = eval_data.get((qid, sys_name), {})
            rd = ragas_data.get((qid, sys_name), {})
            ctx = ed.get("contexts", [])
            f_val = rd.get("scores", {}).get("faithfulness")
            ar_val = rd.get("scores", {}).get("answer_relevancy")

            rows.append({
                "qid": qid,
                "system": sys_name,
                "system_code": sys_code,
                "retry_count": ed.get("retry_count", 0),
                "n_contexts": len(ctx),
                "context_characters": sum(len(c) for c in ctx),
                "faithfulness": float(f_val) if f_val is not None else np.nan,
                "answer_relevancy": float(ar_val) if ar_val is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    dose_path = os.path.join(out_dir, "dose_response.csv")
    df.to_csv(dose_path, index=False)

    # Compute Spearman correlations
    corr_rows = []
    subsets = [
        ("Pooled (S2, S3, S4)", df),
        ("S2 (fixed)", df[df["system"] == "fixed"]),
        ("S3 (adaptive_llm)", df[df["system"] == "adaptive_llm"]),
        ("S4 (adaptive_classifier)", df[df["system"] == "adaptive_classifier"]),
    ]

    variables = [
        ("retry_count", "retry_count"),
        ("n_contexts", "n_contexts"),
        ("context_characters", "context_characters"),
    ]

    for subset_name, sub_df in subsets:
        n_obs = len(sub_df)
        for var_col, var_label in variables:
            x_vals = sub_df[var_col].values
            y_vals = sub_df["faithfulness"].values
            # Filter NaNs
            valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            x_clean = x_vals[valid]
            y_clean = y_vals[valid]

            if len(x_clean) < 3 or np.std(x_clean) == 0 or np.std(y_clean) == 0:
                rho = np.nan
                p_val = np.nan
                note = "constant" if np.std(x_clean) == 0 else "insufficient_data"
            else:
                res = stats.spearmanr(x_clean, y_clean)
                rho = float(res.statistic)
                p_val = float(res.pvalue)
                note = "valid"

            corr_rows.append({
                "subset": subset_name,
                "variable": var_label,
                "N": len(x_clean),
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else np.nan,
                "p_value": round(p_val, 5) if not np.isnan(p_val) else np.nan,
                "note": note,
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(out_dir, "dose_response_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    return df, corr_df


# ---------------------------------------------------------------------------
# I2 — Matched Comparison S3 vs S4
# ---------------------------------------------------------------------------

def run_matched_comparison(eval_data, ragas_data, qids, out_dir):
    """Generate matched_s3_s4.csv and matched_s3_s4_diff_retries.csv."""
    matched_rows = []
    diff_retries_rows = []

    for qid in qids:
        ed_s3 = eval_data.get((qid, "adaptive_llm"), {})
        ed_s4 = eval_data.get((qid, "adaptive_classifier"), {})
        rd_s3 = ragas_data.get((qid, "adaptive_llm"), {})
        rd_s4 = ragas_data.get((qid, "adaptive_classifier"), {})

        r3 = ed_s3.get("retry_count", 0)
        r4 = ed_s4.get("retry_count", 0)
        f3 = rd_s3.get("scores", {}).get("faithfulness")
        f4 = rd_s4.get("scores", {}).get("faithfulness")
        ar3 = rd_s3.get("scores", {}).get("answer_relevancy")
        ar4 = rd_s4.get("scores", {}).get("answer_relevancy")

        f_diff = (f4 - f3) if (f4 is not None and f3 is not None) else np.nan
        ar_diff = (ar4 - ar3) if (ar4 is not None and ar3 is not None) else np.nan

        if r3 == r4:
            matched_rows.append({
                "qid": qid,
                "matched_retry_count": r3,
                "s3_faithfulness": round(float(f3), 4) if f3 is not None else np.nan,
                "s4_faithfulness": round(float(f4), 4) if f4 is not None else np.nan,
                "faithfulness_diff_s4_minus_s3": round(float(f_diff), 4) if not np.isnan(f_diff) else np.nan,
                "s3_relevancy": round(float(ar3), 4) if ar3 is not None else np.nan,
                "s4_relevancy": round(float(ar4), 4) if ar4 is not None else np.nan,
                "relevancy_diff_s4_minus_s3": round(float(ar_diff), 4) if not np.isnan(ar_diff) else np.nan,
            })
        else:
            retried_more = "S3" if r3 > r4 else "S4"
            diff_retries_rows.append({
                "qid": qid,
                "s3_retries": r3,
                "s4_retries": r4,
                "more_retries_by": retried_more,
                "s3_faithfulness": round(float(f3), 4) if f3 is not None else np.nan,
                "s4_faithfulness": round(float(f4), 4) if f4 is not None else np.nan,
                "faithfulness_diff_s4_minus_s3": round(float(f_diff), 4) if not np.isnan(f_diff) else np.nan,
            })

    matched_cols = [
        "qid", "matched_retry_count", "s3_faithfulness", "s4_faithfulness",
        "faithfulness_diff_s4_minus_s3", "s3_relevancy", "s4_relevancy", "relevancy_diff_s4_minus_s3"
    ]
    matched_df = pd.DataFrame(matched_rows, columns=matched_cols)
    matched_path = os.path.join(out_dir, "matched_s3_s4.csv")
    matched_df.to_csv(matched_path, index=False)

    diff_cols = [
        "qid", "s3_retries", "s4_retries", "more_retries_by",
        "s3_faithfulness", "s4_faithfulness", "faithfulness_diff_s4_minus_s3"
    ]
    diff_df = pd.DataFrame(diff_retries_rows, columns=diff_cols)
    diff_path = os.path.join(out_dir, "matched_s3_s4_diff_retries.csv")
    diff_df.to_csv(diff_path, index=False)

    # Matched summary calculation
    diffs = matched_df["faithfulness_diff_s4_minus_s3"].dropna().values
    n_matched = len(diffs)
    mean_diff = float(np.mean(diffs)) if n_matched > 0 else np.nan
    ci_lower, ci_upper = bootstrap_ci(diffs, n_resamples=10000, seed=0)
    wins = int(np.sum(diffs > 1e-6))
    ties = int(np.sum(np.abs(diffs) <= 1e-6))
    losses = int(np.sum(diffs < -1e-6))

    summary_records = [{
        "split": "Matched retries (S3 == S4)",
        "N": n_matched,
        "n_lt_5": n_matched < 5,
        "mean_diff_s4_minus_s3": round(mean_diff, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "wins_s4": wins,
        "ties": ties,
        "losses_s4": losses,
    }]

    # Different retries split calculation
    for split_group, label in [("S3", "S3 retried more (S3 > S4)"), ("S4", "S4 retried more (S4 > S3)")]:
        sub = diff_df[diff_df["more_retries_by"] == split_group]
        sub_diffs = sub["faithfulness_diff_s4_minus_s3"].dropna().values
        n_sub = len(sub_diffs)
        sub_mean = float(np.mean(sub_diffs)) if n_sub > 0 else np.nan
        sub_ci_l, sub_ci_u = bootstrap_ci(sub_diffs, n_resamples=10000, seed=0) if n_sub > 1 else (np.nan, np.nan)
        summary_records.append({
            "split": label,
            "N": n_sub,
            "n_lt_5": n_sub < 5,
            "mean_diff_s4_minus_s3": round(sub_mean, 4),
            "ci_95_lower": round(sub_ci_l, 4) if not np.isnan(sub_ci_l) else np.nan,
            "ci_95_upper": round(sub_ci_u, 4) if not np.isnan(sub_ci_u) else np.nan,
            "wins_s4": int(np.sum(sub_diffs > 1e-6)) if n_sub > 0 else 0,
            "ties": int(np.sum(np.abs(sub_diffs) <= 1e-6)) if n_sub > 0 else 0,
            "losses_s4": int(np.sum(sub_diffs < -1e-6)) if n_sub > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(out_dir, "matched_s3_s4_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    return matched_df, diff_df, summary_df


# ---------------------------------------------------------------------------
# I3 — Judge Agreement Analysis
# ---------------------------------------------------------------------------

def run_judge_agreement(eval_data, ragas_data, qids, out_dir):
    """Generate judge_agreement.csv and judge_agreement_summary.csv."""
    rows = []

    for qid in qids:
        ed_s3 = eval_data.get((qid, "adaptive_llm"), {})
        ed_s4 = eval_data.get((qid, "adaptive_classifier"), {})
        rd_s2 = ragas_data.get((qid, "fixed"), {})

        att3 = ed_s3.get("attempts", [{}])[0]
        att4 = ed_s4.get("attempts", [{}])[0]

        llm_decision = att3.get("decision", "end")
        cls_decision = att4.get("decision", "end")

        llm_retried = (llm_decision != "end")
        cls_retried = (cls_decision != "end")

        fp_critic_score = att3.get("critic_score")
        fp_faithfulness = rd_s2.get("scores", {}).get("faithfulness")

        if (not llm_retried) and (not cls_retried):
            category = "both_accept"
        elif llm_retried and cls_retried:
            category = "both_retry"
        elif llm_retried and (not cls_retried):
            category = "only_llm_retries"
        else:
            category = "only_classifier_retries"

        rows.append({
            "qid": qid,
            "first_pass_critic_score": round(float(fp_critic_score), 2) if fp_critic_score is not None else np.nan,
            "first_pass_faithfulness": round(float(fp_faithfulness), 4) if fp_faithfulness is not None else np.nan,
            "llm_judge_decision": "retry" if llm_retried else "accept",
            "classifier_judge_decision": "retry" if cls_retried else "accept",
            "agreement_category": category,
        })

    df = pd.DataFrame(rows)
    agree_path = os.path.join(out_dir, "judge_agreement.csv")
    df.to_csv(agree_path, index=False)

    # Summary by category
    summary_rows = []
    for cat in ["both_accept", "both_retry", "only_llm_retries", "only_classifier_retries"]:
        sub = df[df["agreement_category"] == cat]
        n_cat = len(sub)
        faith_vals = sub["first_pass_faithfulness"].dropna().values
        score_vals = sub["first_pass_critic_score"].dropna().values

        summary_rows.append({
            "category": cat,
            "count": n_cat,
            "n_lt_5": n_cat < 5,
            "mean_first_pass_faithfulness": round(float(np.mean(faith_vals)), 4) if len(faith_vals) > 0 else np.nan,
            "mean_first_pass_critic_score": round(float(np.mean(score_vals)), 2) if len(score_vals) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "judge_agreement_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    return df, summary_df


# ---------------------------------------------------------------------------
# I4 — Figures Generation
# ---------------------------------------------------------------------------

def generate_figures(eval_data, ragas_data, qids, figures_dir):
    """Generate 5 publication-ready PNG figures with N in titles."""
    os.makedirs(figures_dir, exist_ok=True)
    n_q = len(qids)

    # 1. fig_faithfulness_by_system.png
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]
    codes = ["S1", "S2", "S3", "S4"]
    names = ["S1 (Baseline)", "S2 (Fixed)", "S3 (Adaptive LLM)", "S4 (Adaptive Classifier)"]

    f_data = {}
    ar_data = {}
    for s in systems:
        f_data[s] = [
            ragas_data.get((q, s), {}).get("scores", {}).get("faithfulness", np.nan)
            for q in qids
        ]
        ar_data[s] = [
            ragas_data.get((q, s), {}).get("scores", {}).get("answer_relevancy", np.nan)
            for q in qids
        ]

    # --- Figure 1: Faithfulness by System ---
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    x_positions = np.arange(len(systems))
    means = []
    ci_err_lower = []
    ci_err_upper = []

    for i, s in enumerate(systems):
        vals = np.array([v for v in f_data[s] if not np.isnan(v)])
        m = float(np.mean(vals))
        means.append(m)
        lower, upper = bootstrap_ci(vals, n_resamples=10000, seed=0)
        ci_err_lower.append(m - lower)
        ci_err_upper.append(upper - m)

        # Overlay individual points with jitter
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            x_positions[i] + jitter,
            vals,
            color=CB_PALETTE[codes[i]],
            alpha=0.45,
            edgecolors="none",
            s=40,
            zorder=2,
        )

    ax.errorbar(
        x_positions,
        means,
        yerr=[ci_err_lower, ci_err_upper],
        fmt="o",
        color="#111111",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=5,
        capthick=1.5,
        markersize=8,
        zorder=3,
        label="Mean with 95% Bootstrap CI",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{c}\n{SYSTEM_LABELS[s]}" for c, s in zip(codes, systems)], fontsize=9)
    ax.set_ylabel("RAGAS Faithfulness Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Faithfulness Score by System (N = {n_q} questions)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_faithfulness_by_system.png"), dpi=200)
    plt.close()

    # --- Figure 2: Answer Relevancy by System ---
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    means_ar = []
    ar_err_lower = []
    ar_err_upper = []

    for i, s in enumerate(systems):
        vals = np.array([v for v in ar_data[s] if not np.isnan(v)])
        m = float(np.mean(vals))
        means_ar.append(m)
        lower, upper = bootstrap_ci(vals, n_resamples=10000, seed=0)
        ar_err_lower.append(m - lower)
        ar_err_upper.append(upper - m)

        jitter = np.random.default_rng(i + 10).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            x_positions[i] + jitter,
            vals,
            color=CB_PALETTE[codes[i]],
            alpha=0.45,
            edgecolors="none",
            s=40,
            zorder=2,
        )

    ax.errorbar(
        x_positions,
        means_ar,
        yerr=[ar_err_lower, ar_err_upper],
        fmt="o",
        color="#111111",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=5,
        capthick=1.5,
        markersize=8,
        zorder=3,
        label="Mean with 95% Bootstrap CI",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{c}\n{SYSTEM_LABELS[s]}" for c, s in zip(codes, systems)], fontsize=9)
    ax.set_ylabel("RAGAS Answer Relevancy Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Answer Relevancy Score by System (N = {n_q} questions)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0.65, 1.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_relevancy_by_system.png"), dpi=200)
    plt.close()

    # --- Figure 3: Paired Faithfulness Slope Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=200, sharey=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    s2_vals = np.array(f_data["fixed"])
    s3_vals = np.array(f_data["adaptive_llm"])
    s4_vals = np.array(f_data["adaptive_classifier"])

    # Panel 1: S2 -> S3
    for i in range(n_q):
        y1 = s2_vals[i]
        y2 = s3_vals[i]
        col = CB_PALETTE["S3"] if y2 > y1 + 1e-6 else (CB_PALETTE["dark"] if y2 < y1 - 1e-6 else CB_PALETTE["neutral"])
        ax1.plot([0, 1], [y1, y2], color=col, alpha=0.6, linewidth=1.5, marker="o", markersize=4)

    ax1.plot([0, 1], [np.mean(s2_vals), np.mean(s3_vals)], color="#000000", linewidth=3.0, marker="s", markersize=7, label="Mean Trend")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["S2 (Fixed)", "S3 (Adaptive LLM)"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("RAGAS Faithfulness", fontsize=11, fontweight="bold")
    ax1.set_title("S2 Fixed vs. S3 Adaptive LLM", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # Panel 2: S2 -> S4
    for i in range(n_q):
        y1 = s2_vals[i]
        y2 = s4_vals[i]
        col = CB_PALETTE["S4"] if y2 > y1 + 1e-6 else (CB_PALETTE["dark"] if y2 < y1 - 1e-6 else CB_PALETTE["neutral"])
        ax2.plot([0, 1], [y1, y2], color=col, alpha=0.6, linewidth=1.5, marker="o", markersize=4)

    ax2.plot([0, 1], [np.mean(s2_vals), np.mean(s4_vals)], color="#000000", linewidth=3.0, marker="s", markersize=7, label="Mean Trend")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["S2 (Fixed)", "S4 (Adaptive Cls)"], fontsize=10, fontweight="bold")
    ax2.set_title("S2 Fixed vs. S4 Adaptive Classifier", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Per-Question Faithfulness Trajectory: S2 vs S3 and S2 vs S4 (N = {n_q} questions)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_paired_faithfulness.png"), dpi=200)
    plt.close()

    # --- Figure 4: Cost vs Faithfulness Tradeoff ---
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.3)

    tokens = [
        sum(eval_data[(q, s)]["token_usage"]["total_tokens"] for q in qids)
        for s in systems
    ]

    for i, s in enumerate(systems):
        ax.errorbar(
            tokens[i],
            means[i],
            yerr=[[ci_err_lower[i]], [ci_err_upper[i]]],
            fmt="o",
            color=CB_PALETTE[codes[i]],
            ecolor=CB_PALETTE[codes[i]],
            elinewidth=2,
            capsize=5,
            markersize=10,
            label=f"{codes[i]}: {SYSTEM_LABELS[s]}",
            zorder=3,
        )
        offset_y = 0.02 if i % 2 == 0 else -0.03
        ax.annotate(
            f"{codes[i]} ({means[i]:.3f})",
            (tokens[i], means[i]),
            textcoords="offset points",
            xytext=(10, offset_y * 100),
            fontsize=9,
            fontweight="bold",
        )

    # Plot connecting line showing Pareto / trajectory
    ax.plot(tokens, means, color="#aaaaaa", linestyle=":", linewidth=1.5, zorder=2)

    ax.set_xlabel("Total Tokens Used Across Benchmark", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean RAGAS Faithfulness", fontsize=11, fontweight="bold")
    ax.set_title(f"Cost vs. Faithfulness Tradeoff Across Systems (N = {n_q} questions)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_cost_vs_faithfulness.png"), dpi=200)
    plt.close()

    # --- Figure 5: Retries vs Gain over S2 ---
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.3)

    r3_retries = [eval_data[(q, "adaptive_llm")].get("retry_count", 0) for q in qids]
    r4_retries = [eval_data[(q, "adaptive_classifier")].get("retry_count", 0) for q in qids]
    s3_gains = [s3_vals[i] - s2_vals[i] for i in range(n_q)]
    s4_gains = [s4_vals[i] - s2_vals[i] for i in range(n_q)]

    # Add small horizontal jitter for visualization clarity
    rng3 = np.random.default_rng(42)
    rng4 = np.random.default_rng(84)
    j3 = rng3.uniform(-0.06, 0.06, size=n_q)
    j4 = rng4.uniform(-0.06, 0.06, size=n_q)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax.scatter(np.array(r3_retries) - 0.08 + j3, s3_gains, color=CB_PALETTE["S3"], alpha=0.7, s=55, label="S3 (Adaptive LLM)", edgecolors="none")
    ax.scatter(np.array(r4_retries) + 0.08 + j4, s4_gains, color=CB_PALETTE["S4"], alpha=0.7, s=55, label="S4 (Adaptive Classifier)", edgecolors="none")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Retry Count", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Faithfulness Gain over S2 ($\Delta$ Faithfulness)", fontsize=11, fontweight="bold")
    ax.set_title(f"Retry Count vs. Faithfulness Gain Over S2 (N = {n_q} questions)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_retries_vs_gain.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Update README.md
# ---------------------------------------------------------------------------

def update_readme(out_dir: str, n_questions: int):
    """Update README.md with comprehensive descriptions of all analysis tables and figures."""
    content = f"""# Analysis Results and Confound Checks Summary (N = {n_questions} Questions)

This directory contains statistical evaluations, confound checks, and visual figures across the benchmark evaluation:

## Core Comparison Files
1. **per_system.csv**: Evaluates N={n_questions} questions per system (S1 Baseline, S2 Fixed, S3 Adaptive LLM, S4 Adaptive Classifier). Reports mean, median, standard deviation, and null counts for RAGAS Faithfulness and Answer Relevancy, alongside mean critic scores, mean retry counts, and total token usage.
2. **paired.csv**: Contains paired comparisons (S2->S3, S2->S4, S3->S4) across both metrics. Reports mean differences, 10,000-resample bootstrap 95% confidence intervals (seed 0), two-sided Wilcoxon signed-rank test p-values (dropping zero-difference pairs), step-down Holm-Bonferroni adjusted p-values across all 6 comparisons, and win/tie/loss counts.
3. **retry_effect.csv**: Analyzes questions where adaptive systems triggered retries (retry_count > 0). Shows pre-retry and post-retry critic scores alongside RAGAS Faithfulness compared against the shared attempt-0 S2 baseline.
4. **judge_disagreement.csv**: Documents per-question retry counts between S3 (LLM judge) and S4 (Cross-Encoder judge), identifying decision differences and which system achieved higher final Faithfulness.
5. **cost.csv**: Summarizes total token consumption and marginal token expenditure per +0.1 gain in RAGAS Faithfulness relative to S2 Fixed.

## Confound Checks (Task I)
6. **dose_response.csv**: Records per-question metrics for S2, S3, and S4 (N={n_questions * 3} rows total), capturing retry_count, number of context passages (n_contexts), total context characters, RAGAS Faithfulness, and Answer Relevancy.
7. **dose_response_correlations.csv**: Spearman rank correlation between Faithfulness and retry_count, n_contexts, and context characters. Computed for pooled systems (N={n_questions * 3}) and per-system subsets (N={n_questions} each). Note: S2 retry_count and n_contexts are constant (rho=N/A).
8. **matched_s3_s4.csv**: Isolates questions where S3 and S4 underwent identical retry counts (N=6 questions: Q05, Q07, Q11, Q15, Q16, Q18).
9. **matched_s3_s4_diff_retries.csv**: Isolates questions where retry counts differed between S3 and S4 (N=14 questions).
10. **matched_s3_s4_summary.csv**: Aggregated comparison for matched retries vs differing retries. Reports mean faithfulness differences, bootstrap 95% CI (seed 0), win/tie/loss counts, and notes where subset sizes are n < 5 (specifically, the S3 > S4 split has n=3).
11. **judge_agreement.csv**: Documents attempt-0 judge behavior for all N={n_questions} questions, recording first-pass critic score, first-pass faithfulness, individual judge decisions (accept vs retry), and categorizing agreement.
12. **judge_agreement_summary.csv**: Group summary of the four agreement categories (both_accept, both_retry, only_llm_retries, only_classifier_retries) with counts, mean first-pass faithfulness, and mean critic score. Notes where group size is n < 5 (both_accept has n=1, only_llm_retries has n=1).

## Publication Figures (`figures/`)
- **fig_faithfulness_by_system.png**: Mean RAGAS Faithfulness per system with 95% bootstrap CI error bars and overlaid individual question points.
- **fig_relevancy_by_system.png**: Mean RAGAS Answer Relevancy per system with 95% bootstrap CI error bars and overlaid individual question points.
- **fig_paired_faithfulness.png**: Per-question slope plot tracing individual trajectories for S2 -> S3 and S2 -> S4 alongside the mean trend.
- **fig_cost_vs_faithfulness.png**: Total token consumption vs. mean RAGAS Faithfulness for all four systems with 95% bootstrap CI.
- **fig_retries_vs_gain.png**: Number of retries vs. faithfulness gain over S2, colored by system (S3 vs S4).
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_confound_analysis(
    eval_dir: str = "results/eval",
    ragas_dir: str = "results/ragas",
    out_dir: str = "results/analysis",
):
    """Execute all Task I confound checks and figure generation."""
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    eval_data, ragas_data, qids = load_all_data(eval_dir, ragas_dir)
    print(f"[Confound Analysis] Loaded {len(eval_data)} eval records and {len(ragas_data)} ragas records across {len(qids)} questions.")

    # I1: Dose-response
    run_dose_response(eval_data, ragas_data, qids, out_dir)
    print(f"[Confound Analysis] Generated dose_response.csv and dose_response_correlations.csv")

    # I2: Matched comparison
    run_matched_comparison(eval_data, ragas_data, qids, out_dir)
    print(f"[Confound Analysis] Generated matched_s3_s4.csv, matched_s3_s4_diff_retries.csv, matched_s3_s4_summary.csv")

    # I3: Judge agreement
    run_judge_agreement(eval_data, ragas_data, qids, out_dir)
    print(f"[Confound Analysis] Generated judge_agreement.csv and judge_agreement_summary.csv")

    # I4: Figures
    generate_figures(eval_data, ragas_data, qids, figures_dir)
    print(f"[Confound Analysis] Generated 5 figures in {figures_dir}/")

    # Update README
    update_readme(out_dir, len(qids))
    print(f"[Confound Analysis] Updated README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Confounds and Generate Figures (Task I)")
    parser.add_argument("--eval-dir", default="results/eval", help="Path to results/eval")
    parser.add_argument("--ragas-dir", default="results/ragas", help="Path to results/ragas")
    parser.add_argument("--out-dir", default="results/analysis", help="Path to results/analysis")
    args = parser.parse_args()

    run_confound_analysis(args.eval_dir, args.ragas_dir, args.out_dir)
