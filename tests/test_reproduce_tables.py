"""Unit tests for reproduction comparison function (Task J3).

Tests:
- CSV comparison identical numeric and string values.
- CSV comparison tolerance boundaries.
- CSV comparison failure on column mismatch, shape mismatch, or value divergence.
"""

import os
import pytest
import pandas as pd
from scripts.reproduce_tables import compare_csv_files


def test_compare_csv_identical(tmp_path):
    df = pd.DataFrame({
        "system": ["S1", "S2", "S3"],
        "faithfulness": [0.0849, 0.2357, 0.3663],
        "count": [20, 20, 20],
    })
    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    df.to_csv(f1, index=False)
    df.to_csv(f2, index=False)

    is_match, reason = compare_csv_files(str(f1), str(f2), tol=1e-4)
    assert is_match is True
    assert "Identical" in reason


def test_compare_csv_within_tolerance(tmp_path):
    df1 = pd.DataFrame({"score": [0.123456]})
    df2 = pd.DataFrame({"score": [0.123480]})  # diff = 0.000024 < 1e-4

    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    is_match, reason = compare_csv_files(str(f1), str(f2), tol=1e-4)
    assert is_match is True


def test_compare_csv_exceeds_tolerance(tmp_path):
    df1 = pd.DataFrame({"score": [0.123]})
    df2 = pd.DataFrame({"score": [0.150]})  # diff = 0.027 > 1e-4

    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    is_match, reason = compare_csv_files(str(f1), str(f2), tol=1e-4)
    assert is_match is False
    assert "Numeric mismatch" in reason


def test_compare_csv_column_mismatch(tmp_path):
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [1], "c": [2]})

    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    is_match, reason = compare_csv_files(str(f1), str(f2))
    assert is_match is False
    assert "Column mismatch" in reason


def test_compare_csv_shape_mismatch(tmp_path):
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1]})

    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    is_match, reason = compare_csv_files(str(f1), str(f2))
    assert is_match is False
    assert "Shape mismatch" in reason
