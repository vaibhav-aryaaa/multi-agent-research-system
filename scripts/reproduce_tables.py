#!/usr/bin/env python3
"""Offline Reproduction Verification Script (Task J3).

Verifies that all statistical analysis tables can be reproduced deterministically
from the committed evaluation and RAGAS result JSON files without API keys.

Workflow:
1. Re-executes scripts/analyze_results.py and scripts/analyze_confounds.py into a temporary directory.
2. Compares the freshly generated CSV files against the committed CSV files in results/analysis/.
3. Evaluates numeric tolerance (default 1e-4) and exact string column matching.
4. Reports PASS/FAIL per table and exits non-zero if any comparison fails.
"""

import os
import sys
import tempfile
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.analyze_results import run_analysis
from scripts.analyze_confounds import run_confound_analysis

DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "eval")
DEFAULT_RAGAS_DIR = os.path.join(PROJECT_ROOT, "results", "ragas")
DEFAULT_COMMITTED_DIR = os.path.join(PROJECT_ROOT, "results", "analysis")

MANDATORY_CHECK_FILES = [
    "per_system.csv",
    "paired.csv",
    "matched_s3_s4_summary.csv",
    "dose_response_correlations.csv",
]

ADDITIONAL_CHECK_FILES = [
    "cost.csv",
    "judge_agreement_summary.csv",
    "retry_effect.csv",
    "judge_disagreement.csv",
    "dose_response.csv",
    "matched_s3_s4.csv",
]


def compare_csv_files(
    committed_path: str,
    generated_path: str,
    tol: float = 1e-4,
) -> Tuple[bool, str]:
    """Compare two CSV files for structural and numerical equivalence.
    
    Returns (is_match, reason).
    """
    if not os.path.exists(committed_path):
        return False, f"Committed file missing: {committed_path}"
    if not os.path.exists(generated_path):
        return False, f"Generated file missing: {generated_path}"

    try:
        df_comm = pd.read_csv(committed_path)
        df_gen = pd.read_csv(generated_path)
    except Exception as e:
        return False, f"Failed to parse CSV: {e}"

    # Check dimensions
    if df_comm.shape != df_gen.shape:
        return False, f"Shape mismatch: committed {df_comm.shape} vs generated {df_gen.shape}"

    # Check columns
    if list(df_comm.columns) != list(df_gen.columns):
        return False, f"Column mismatch: committed {list(df_comm.columns)} vs generated {list(df_gen.columns)}"

    # Check values column by column
    for col in df_comm.columns:
        s_comm = df_comm[col]
        s_gen = df_gen[col]

        # Check if column is numeric
        is_num_comm = pd.api.types.is_numeric_dtype(s_comm)
        is_num_gen = pd.api.types.is_numeric_dtype(s_gen)

        if is_num_comm and is_num_gen:
            # Handle float/int comparison with NaN tolerance
            arr_comm = s_comm.to_numpy(dtype=float)
            arr_gen = s_gen.to_numpy(dtype=float)

            both_nan = np.isnan(arr_comm) & np.isnan(arr_gen)
            diffs = np.abs(arr_comm - arr_gen)
            diffs[both_nan] = 0.0

            max_diff = np.nanmax(diffs) if len(diffs) > 0 else 0.0
            if max_diff > tol:
                idx = int(np.nanargmax(diffs))
                return False, f"Numeric mismatch in '{col}' at row {idx}: committed {arr_comm[idx]} vs generated {arr_gen[idx]} (max diff {max_diff:.6f} > {tol})"
        else:
            # String/Boolean/Categorical comparison
            str_comm = s_comm.fillna("").astype(str).tolist()
            str_gen = s_gen.fillna("").astype(str).tolist()
            for r_idx, (v_c, v_g) in enumerate(zip(str_comm, str_gen)):
                if v_c != v_g:
                    # Handle possible string representation of floats e.g. "0.0" vs "0.00"
                    try:
                        f_c, f_g = float(v_c), float(v_g)
                        if abs(f_c - f_g) <= tol:
                            continue
                    except ValueError:
                        pass
                    return False, f"Value mismatch in '{col}' at row {r_idx}: committed '{v_c}' vs generated '{v_g}'"

    return True, f"Identical within tolerance {tol}"


def reproduce_tables(
    eval_dir: str = DEFAULT_EVAL_DIR,
    ragas_dir: str = DEFAULT_RAGAS_DIR,
    committed_dir: str = DEFAULT_COMMITTED_DIR,
    tol: float = 1e-4,
    check_all: bool = True,
) -> Dict[str, Tuple[bool, str]]:
    """Re-generate all analysis tables into a temporary directory and verify equivalence."""
    with tempfile.TemporaryDirectory(prefix="reproduce_analysis_") as tmp_dir:
        # 1. Execute analysis pipeline into temporary directory
        run_analysis(eval_dir=eval_dir, ragas_dir=ragas_dir, out_dir=tmp_dir)

        # 2. Compare target files
        check_list = MANDATORY_CHECK_FILES + (ADDITIONAL_CHECK_FILES if check_all else [])
        results = {}

        for filename in check_list:
            comm_file = os.path.join(committed_dir, filename)
            gen_file = os.path.join(tmp_dir, filename)
            is_match, reason = compare_csv_files(comm_file, gen_file, tol=tol)
            results[filename] = (is_match, reason)

        return results


def main():
    parser = argparse.ArgumentParser(description="Deterministic Offline Reproduction Checker")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, help="Path to results/eval")
    parser.add_argument("--ragas-dir", default=DEFAULT_RAGAS_DIR, help="Path to results/ragas")
    parser.add_argument("--committed-dir", default=DEFAULT_COMMITTED_DIR, help="Path to committed results/analysis")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Floating-point numeric tolerance")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("REPRODUCTION VERIFICATION: Statistical Tables & Figures")
    print(f"Eval inputs:        {args.eval_dir}")
    print(f"Ragas inputs:       {args.ragas_dir}")
    print(f"Committed baseline: {args.committed_dir}")
    print(f"Numeric tolerance:  {args.tolerance}")
    print("=" * 80 + "\n")

    results = reproduce_tables(
        eval_dir=args.eval_dir,
        ragas_dir=args.ragas_dir,
        committed_dir=args.committed_dir,
        tol=args.tolerance,
    )

    all_passed = True
    print(f"{'Table / File':<35} | {'Status':<8} | {'Details'}")
    print("-" * 80)
    for filename, (passed, details) in results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{filename:<35} | {status_str:<8} | {details}")
    print("-" * 80 + "\n")

    if all_passed:
        print("[SUCCESS] All statistical tables reproduced deterministically from result files!")
        sys.exit(0)
    else:
        print("[FAILURE] One or more tables failed reproduction check.")
        sys.exit(1)


if __name__ == "__main__":
    main()
