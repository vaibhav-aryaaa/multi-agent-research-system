"""Unit test to ensure README.md Key Results table strictly mirrors results/analysis/per_system.csv.

Fails if any cell differs from per_system.csv by more than 0.0005.
"""

import os
import re
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
PER_SYSTEM_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "analysis", "per_system.csv")


def test_readme_key_results_table_matches_per_system_csv():
    assert os.path.exists(README_PATH), f"README.md not found at {README_PATH}"
    assert os.path.exists(PER_SYSTEM_CSV_PATH), f"per_system.csv not found at {PER_SYSTEM_CSV_PATH}"

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Extract all rows matching | **S1** |, | **S2** |, | **S3** |, | **S4** |
    table_rows = [
        line.strip() for line in readme_text.splitlines()
        if re.match(r"^\|\s*\*\*S[1-4]\*\*\s*\|", line.strip())
    ]

    assert len(table_rows) == 4, f"Expected 4 table rows for S1-S4 in README, found {len(table_rows)}"

    df_csv = pd.read_csv(PER_SYSTEM_CSV_PATH).set_index("system_code")
    assert set(df_csv.index) == {"S1", "S2", "S3", "S4"}

    tolerance = 0.0005

    for line in table_rows:
        parts = [p.strip() for p in line.split("|")[1:-1]]
        assert len(parts) >= 7, f"Row has insufficient columns: {line}"

        code = parts[0].replace("*", "")
        f_mean = float(parts[2].replace("*", ""))
        f_med = float(parts[3])
        ar_mean = float(parts[4])
        retries = float(parts[5])
        tokens = float(parts[6].replace(",", ""))

        assert code in df_csv.index, f"Unknown system code: {code}"
        csv_row = df_csv.loc[code]

        csv_f_mean = float(csv_row["faithfulness_mean"])
        csv_f_med = float(csv_row["faithfulness_median"])
        csv_ar_mean = float(csv_row["answer_relevancy_mean"])
        csv_retries = float(csv_row["mean_retry_count"])
        csv_tokens = float(csv_row["total_tokens"])

        diff_f_mean = abs(f_mean - csv_f_mean)
        diff_f_med = abs(f_med - csv_f_med)
        diff_ar_mean = abs(ar_mean - csv_ar_mean)
        diff_retries = abs(retries - csv_retries)
        diff_tokens = abs(tokens - csv_tokens)

        assert diff_f_mean <= tolerance, (
            f"[{code}] Mean Faithfulness mismatch: README={f_mean}, CSV={csv_f_mean:.4f} (diff={diff_f_mean:.6f} > {tolerance})"
        )
        assert diff_f_med <= tolerance, (
            f"[{code}] Median Faithfulness mismatch: README={f_med}, CSV={csv_f_med:.4f} (diff={diff_f_med:.6f} > {tolerance})"
        )
        assert diff_ar_mean <= tolerance, (
            f"[{code}] Mean Relevancy mismatch: README={ar_mean}, CSV={csv_ar_mean:.4f} (diff={diff_ar_mean:.6f} > {tolerance})"
        )
        assert diff_retries <= tolerance, (
            f"[{code}] Mean Retries mismatch: README={retries}, CSV={csv_retries:.4f} (diff={diff_retries:.6f} > {tolerance})"
        )
        assert diff_tokens <= tolerance, (
            f"[{code}] Total Tokens mismatch: README={tokens}, CSV={csv_tokens} (diff={diff_tokens:.6f} > {tolerance})"
        )
