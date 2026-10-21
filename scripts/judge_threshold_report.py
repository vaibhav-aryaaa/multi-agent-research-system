#!/usr/bin/env python3
"""Judge threshold report analysis for report-source claims (Phase 3 Part C0).

For each of the 15 reports (report-source rows only):
- Computes human-labeled supported fraction.
- Computes cross-encoder predicted supported fraction at tau 0.66 under two scoring methods:
  1. Way 1 (CSV): Scored against the two evidence passages in the labeled CSV.
  2. Way 2 (Pipeline Full Contexts): Scored against top-3 passages retrieved from that
     report's full cleaned evidence (from results/ or docs/examples/ JSONs).
- Sweeps theta in [0.1, 0.2, ..., 0.9] and reports the number of accepted reports.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from scripts.collect_claims import clean_evidence_passages
from faithfulness.cross_encoder import score_claims, load_cross_encoder

DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "validation_claims_labeled.csv")
SEARCH_DIRS = [
    os.path.join(PROJECT_ROOT, "docs", "examples"),
    os.path.join(PROJECT_ROOT, "results"),
]


def find_report_contexts(report_id: str) -> Tuple[List[str], Optional[str]]:
    """Locate the result JSON for report_id and return raw contexts and file path."""
    for sdir in SEARCH_DIRS:
        for fpath in glob.glob(os.path.join(sdir, "*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rid = data.get("run_id") or os.path.splitext(os.path.basename(fpath))[0]
                rid = rid.replace("_fixed", "").replace("_adaptive_llm", "").replace("_baseline", "")
                if rid == report_id or os.path.basename(fpath).startswith(report_id):
                    return data.get("contexts", []), fpath
            except Exception:
                continue
    return [], None


def run_threshold_analysis(
    csv_path: str = DEFAULT_CSV_PATH,
    tau: float = 0.66,
    thetas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if thetas is None:
        thetas = [round(x, 1) for x in np.linspace(0.1, 0.9, 9)]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "source" not in df.columns:
        raise ValueError("CSV missing 'source' column.")

    rep_df = df[df["source"] == "report"].copy()
    if rep_df.empty:
        raise ValueError("No rows found with source='report'.")

    # Drop non-binary labels if any
    rep_df = rep_df[rep_df["label"].astype(str).str.strip().isin(["0", "1", 0, 1])].copy()
    rep_df["label"] = rep_df["label"].astype(int)

    # Ensure model is loaded once
    print("Loading Cross-Encoder model...")
    model, tokenizer, device = load_cross_encoder()

    report_ids = rep_df["report_id"].unique()
    print(f"Analyzing {len(report_ids)} reports ({len(rep_df)} total claims) at tau = {tau:.2f}...\n")

    report_results = []

    for rid in report_ids:
        r_rows = rep_df[rep_df["report_id"] == rid].copy()
        topic = r_rows["topic"].iloc[0]
        n_claims = len(r_rows)
        claims = r_rows["claim"].tolist()
        human_labels = r_rows["label"].tolist()

        human_supp_count = sum(human_labels)
        human_supp_frac = human_supp_count / n_claims if n_claims > 0 else 0.0

        # Way 1: Score each claim against its 2 evidence passages from CSV
        way1_supp_count = 0
        way1_probs = []
        for _, row in r_rows.iterrows():
            c = row["claim"]
            passages = []
            ev1 = str(row.get("evidence_top1", "")).strip()
            ev2 = str(row.get("evidence_top2", "")).strip()
            if ev1 and ev1.lower() != "nan":
                passages.append(ev1)
            if ev2 and ev2.lower() != "nan":
                passages.append(ev2)

            scored = score_claims([c], passages, k=2, model=model, tokenizer=tokenizer)
            p_supp = scored[0]["p_supported"] if scored else 0.0
            way1_probs.append(p_supp)
            if p_supp >= tau:
                way1_supp_count += 1

        way1_supp_frac = way1_supp_count / n_claims if n_claims > 0 else 0.0

        # Way 2: Score against top-3 retrieved from report's full cleaned evidence
        raw_ctx, fpath = find_report_contexts(rid)
        if not raw_ctx:
            print(f"  [WARN] No context JSON found for report {rid}")
            cleaned_passages = []
        else:
            cleaned_passages = clean_evidence_passages(raw_ctx)

        scored_way2 = score_claims(claims, cleaned_passages, k=3, model=model, tokenizer=tokenizer)
        way2_probs = [s["p_supported"] for s in scored_way2]
        way2_supp_count = sum(1 for p in way2_probs if p >= tau)
        way2_supp_frac = way2_supp_count / n_claims if n_claims > 0 else 0.0

        report_results.append({
            "report_id": rid,
            "topic": topic,
            "n_claims": n_claims,
            "human_supp_count": human_supp_count,
            "human_supp_frac": human_supp_frac,
            "way1_csv_supp_count": way1_supp_count,
            "way1_csv_supp_frac": way1_supp_frac,
            "way2_full_supp_count": way2_supp_count,
            "way2_full_supp_frac": way2_supp_frac,
            "n_full_passages": len(cleaned_passages),
            "source_file": os.path.basename(fpath) if fpath else "NOT FOUND",
        })

    # Summary acceptance across thetas
    acceptance_by_theta = []
    for theta in thetas:
        acc_human = sum(1 for r in report_results if r["human_supp_frac"] >= theta)
        acc_way1 = sum(1 for r in report_results if r["way1_csv_supp_frac"] >= theta)
        acc_way2 = sum(1 for r in report_results if r["way2_full_supp_frac"] >= theta)
        acceptance_by_theta.append({
            "theta": theta,
            "accepted_human": acc_human,
            "accepted_way1_csv": acc_way1,
            "accepted_way2_full": acc_way2,
            "total_reports": len(report_results),
        })

    return {
        "tau": tau,
        "thetas": thetas,
        "reports": report_results,
        "acceptance": acceptance_by_theta,
    }


def print_tables(data: Dict[str, Any]):
    reports = data["reports"]
    tau = data["tau"]
    acceptance = data["acceptance"]

    print("=" * 115)
    print(f"REPORT-LEVEL FAITHFULNESS SUMMARY (tau={tau:.2f})")
    print("=" * 115)
    header = (
        f"{'Report ID':<24} | {'Topic (truncated)':<35} | {'Claims':<6} | "
        f"{'Human (Frac)':<12} | {'Way 1 CSV':<12} | {'Way 2 Full Top-3':<16}"
    )
    print(header)
    print("-" * 115)

    for r in reports:
        rid = r["report_id"]
        top = (r["topic"][:32] + "...") if len(r["topic"]) > 35 else r["topic"]
        nc = r["n_claims"]
        h_str = f"{r['human_supp_count']}/{nc} ({r['human_supp_frac']:.2f})"
        w1_str = f"{r['way1_csv_supp_count']}/{nc} ({r['way1_csv_supp_frac']:.2f})"
        w2_str = f"{r['way2_full_supp_count']}/{nc} ({r['way2_full_supp_frac']:.2f})"
        print(f"{rid:<24} | {top:<35} | {nc:<6} | {h_str:<12} | {w1_str:<12} | {w2_str:<16}")

    print("=" * 115)
    print("\n" + "=" * 75)
    print(f"REPORT ACCEPTANCE VS THETA (Accept if supported_fraction >= theta)")
    print("=" * 75)
    print(f"{'Theta':<8} | {'Human Accepted':<16} | {'Way 1 (CSV) Accepted':<22} | {'Way 2 (Full Top-3) Accepted':<26}")
    print("-" * 75)
    total_reps = len(reports)
    for a in acceptance:
        th = a["theta"]
        h_acc = f"{a['accepted_human']}/{total_reps} ({a['accepted_human']/total_reps*100:.1f}%)"
        w1_acc = f"{a['accepted_way1_csv']}/{total_reps} ({a['accepted_way1_csv']/total_reps*100:.1f}%)"
        w2_acc = f"{a['accepted_way2_full']}/{total_reps} ({a['accepted_way2_full']/total_reps*100:.1f}%)"
        print(f"{th:<8.1f} | {h_acc:<16} | {w1_acc:<22} | {w2_acc:<26}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Judge Threshold Report")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to labeled claims CSV")
    parser.add_argument("--tau", type=float, default=0.66, help="Decision threshold tau (default: 0.66)")
    parser.add_argument("--json_output", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    data = run_threshold_analysis(csv_path=args.csv, tau=args.tau)
    print_tables(data)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved JSON report to: {args.json_output}")


if __name__ == "__main__":
    main()
