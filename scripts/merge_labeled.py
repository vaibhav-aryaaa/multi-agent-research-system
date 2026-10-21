"""Merge report-extracted and synthetic human-labeled claims into validation_claims_labeled.csv.

Validates labels in {0, 1, ?}, checks human-generator agreement on the synthetic set
against data/synthetic_key.csv, adds the 'source' column ('report' vs 'synthetic'),
and enforces the >= 20 per class gate for evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_REPORTS_PATH = os.path.join(PROJECT_ROOT, "data", "validation_claims_labeled_reports.csv")
DEFAULT_SYNTHETIC_PATH = os.path.join(PROJECT_ROOT, "data", "validation_claims_synthetic_labeled.csv")
DEFAULT_KEY_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_key.csv")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "validation_claims_labeled.csv")


def clean_label_val(val: Any) -> str:
    """Normalize label to '0', '1', '?' or empty string."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s in ("0", "0.0"):
        return "0"
    if s in ("1", "1.0"):
        return "1"
    if s == "?":
        return "?"
    if s == "":
        return ""
    raise ValueError(f"Invalid label '{val}'. Only 0, 1, '?' or empty string are allowed.")


def validate_and_count_labels(df: pd.DataFrame, file_desc: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Validate labels and count occurrences of 0, 1, ?, and empty."""
    df_copy = df.copy()
    counts = {"0": 0, "1": 0, "?": 0, "blank": 0}

    cleaned_labels = []
    for idx, row in df_copy.iterrows():
        lbl_raw = row["label"]
        try:
            lbl_clean = clean_label_val(lbl_raw)
        except ValueError as e:
            raise ValueError(f"Error in {file_desc} at row {idx + 2}: {e}")

        if lbl_clean == "0":
            counts["0"] += 1
        elif lbl_clean == "1":
            counts["1"] += 1
        elif lbl_clean == "?":
            counts["?"] += 1
        else:
            counts["blank"] += 1

        cleaned_labels.append(lbl_clean)

    df_copy["label"] = cleaned_labels
    return df_copy, counts


def compute_agreement(df_synth: pd.DataFrame, key_path: str):
    """Compute and print human-vs-generator agreement 2x2 matrix."""
    if not os.path.exists(key_path):
        print(f"[WARN] Synthetic key not found at {key_path}. Skipping agreement table.")
        return

    df_key = pd.read_csv(key_path)
    merged = pd.merge(
        df_synth,
        df_key[["report_id", "claim_index", "gen_type"]],
        on=["report_id", "claim_index"],
        how="inner",
    )

    if len(merged) == 0:
        print("[WARN] No matching rows between synthetic claims and key.")
        return

    # Counts for 2x2
    f_0 = int(((merged["gen_type"] == "faithful") & (merged["label"] == "0")).sum())
    f_1 = int(((merged["gen_type"] == "faithful") & (merged["label"] == "1")).sum())
    f_q = int(((merged["gen_type"] == "faithful") & (merged["label"] == "?")).sum())
    f_b = int(((merged["gen_type"] == "faithful") & (merged["label"] == "")).sum())

    c_0 = int(((merged["gen_type"] == "corrupted") & (merged["label"] == "0")).sum())
    c_1 = int(((merged["gen_type"] == "corrupted") & (merged["label"] == "1")).sum())
    c_q = int(((merged["gen_type"] == "corrupted") & (merged["label"] == "?")).sum())
    c_b = int(((merged["gen_type"] == "corrupted") & (merged["label"] == "")).sum())

    total_binary = (f_0 + f_1 + c_0 + c_1)
    agreed = (f_1 + c_0)
    agreement_pct = (agreed / total_binary * 100.0) if total_binary > 0 else 0.0

    print("\n" + "=" * 70)
    print("HUMAN VS GENERATOR AGREEMENT (Synthetic Set)")
    print("=" * 70)
    print(f"{'Generator Type':<20} | {'Human: 0 (Unsup)':<16} | {'Human: 1 (Supp)':<15} | {'?':<6} | {'Total':<6}")
    print("-" * 70)
    print(f"{'Faithful (exp 1)':<20} | {f_0:<16} | {f_1:<15} | {f_q + f_b:<6} | {f_0 + f_1 + f_q + f_b:<6}")
    print(f"{'Corrupted (exp 0)':<20} | {c_0:<16} | {c_1:<15} | {c_q + c_b:<6} | {c_0 + c_1 + c_q + c_b:<6}")
    print("-" * 70)
    print(f"{'Total':<20} | {f_0 + c_0:<16} | {f_1 + c_1:<15} | {f_q + f_b + c_q + c_b:<6} | {len(merged):<6}")
    print("=" * 70)
    if total_binary > 0:
        print(f"Agreement on binary labels (0/1): {agreement_pct:.2f}% ({agreed}/{total_binary})")
    print("=" * 70 + "\n")


def merge_datasets(
    reports_path: str = DEFAULT_REPORTS_PATH,
    synthetic_path: str = DEFAULT_SYNTHETIC_PATH,
    key_path: str = DEFAULT_KEY_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Validate, merge report and synthetic labeled datasets, and write final CSV."""
    if not os.path.exists(reports_path):
        raise FileNotFoundError(f"Reports labeled file not found at: {reports_path}")
    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(
            f"Synthetic labeled file not found at: {synthetic_path}.\n"
            "The human labeler must first annotate data/validation_claims_synthetic.csv "
            f"and save to {synthetic_path}."
        )

    df_reports = pd.read_csv(reports_path)
    df_synth = pd.read_csv(synthetic_path)

    # Validate and count
    df_reports_clean, rep_counts = validate_and_count_labels(df_reports, "Reports File")
    df_synth_clean, synth_counts = validate_and_count_labels(df_synth, "Synthetic File")

    print("=" * 70)
    print("INPUT DATASET LABEL COUNTS")
    print("=" * 70)
    print(f"Reports ({reports_path}):   0={rep_counts['0']} | 1={rep_counts['1']} | ?={rep_counts['?']} | blank={rep_counts['blank']}")
    print(f"Synthetic ({synthetic_path}): 0={synth_counts['0']} | 1={synth_counts['1']} | ?={synth_counts['?']} | blank={synth_counts['blank']}")

    # Compute agreement
    compute_agreement(df_synth_clean, key_path)

    # Ensure source column
    df_reports_clean["source"] = "report"
    df_synth_clean["source"] = "synthetic"

    # Match column schemas
    target_columns = ["report_id", "topic", "claim_index", "claim", "evidence_top1", "evidence_top2", "label", "source"]
    for col in target_columns:
        if col not in df_reports_clean.columns:
            df_reports_clean[col] = ""
        if col not in df_synth_clean.columns:
            df_synth_clean[col] = ""

    df_merged = pd.concat(
        [df_reports_clean[target_columns], df_synth_clean[target_columns]],
        ignore_index=True,
    )

    # Class balance check on usable binary labels (0 and 1)
    usable_rows = df_merged[df_merged["label"].isin(["0", "1"])]
    usable_0 = int((usable_rows["label"] == "0").sum())
    usable_1 = int((usable_rows["label"] == "1").sum())

    rep_usable_0 = int(((usable_rows["source"] == "report") & (usable_rows["label"] == "0")).sum())
    rep_usable_1 = int(((usable_rows["source"] == "report") & (usable_rows["label"] == "1")).sum())
    syn_usable_0 = int(((usable_rows["source"] == "synthetic") & (usable_rows["label"] == "0")).sum())
    syn_usable_1 = int(((usable_rows["source"] == "synthetic") & (usable_rows["label"] == "1")).sum())

    print("=" * 70)
    print("FINAL MERGED USABLE PER-CLASS COUNTS (0 / 1)")
    print("=" * 70)
    print(f"Class 0 (Unsupported): Total={usable_0} | Report={rep_usable_0} | Synthetic={syn_usable_0}")
    print(f"Class 1 (Supported):   Total={usable_1} | Report={rep_usable_1} | Synthetic={syn_usable_1}")
    print(f"Total Usable Annotations: {len(usable_rows)} (out of {len(df_merged)} total rows)")
    print("=" * 70)

    # Enforce >= 20 per class gate
    if usable_0 < 20 or usable_1 < 20:
        msg = (
            f"ERROR: Class balance gate failed! Class 0 has {usable_0} rows, Class 1 has {usable_1} rows. "
            "Both classes must have at least 20 usable human annotations for valid evaluation."
        )
        print(f"\n[GATE FAILED] {msg}\n")
        raise SystemExit(msg)

    # Save merged file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"\nMerged dataset successfully saved to: {output_path} ({len(df_merged)} rows)")

    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Merge labeled report claims and synthetic claims")
    parser.add_argument("--reports", default=DEFAULT_REPORTS_PATH, help="Path to labeled reports CSV")
    parser.add_argument("--synthetic", default=DEFAULT_SYNTHETIC_PATH, help="Path to labeled synthetic CSV")
    parser.add_argument("--key", default=DEFAULT_KEY_PATH, help="Path to synthetic key CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output path for merged labeled CSV")
    args = parser.parse_args()

    merge_datasets(
        reports_path=args.reports,
        synthetic_path=args.synthetic,
        key_path=args.key,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
