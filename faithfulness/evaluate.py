"""Evaluate faithfulness classifiers on human-labeled in-domain evidence.

Evaluates either the Logistic Regression baseline ('lr') or the DistilBERT
Cross-Encoder ('cross_encoder') using an identical 50/50 stratified dev/heldout split.
The decision threshold (tau) is tuned strictly on the dev split and evaluated on heldout.
Baselines (always supported, cosine-only threshold) are computed on the same heldout rows.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables (.env) are loaded first
load_dotenv()
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from faithfulness.embedder import embed
from faithfulness.features import build_claim_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LR_PATH = os.path.join(MODELS_DIR, "faithfulness_lr.joblib")
DEFAULT_CSV_PATH = os.path.join("data", "validation_claims_labeled.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faithfulness Classifier on In-Domain Labeled Data")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to human-labeled claims CSV (claim, evidence_top1, evidence_top2, label)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["lr", "cross_encoder"],
        default="lr",
        help="Classifier backend: 'lr' or 'cross_encoder' (default: lr)",
    )
    parser.add_argument(
        "--output", "--eval_output",
        dest="output",
        type=str,
        default=None,
        help="Path to save evaluation JSON (default: faithfulness/models/indomain_eval_<backend>.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 50/50 stratified split (default: 42)",
    )
    parser.add_argument(
        "--skip_sample_gate",
        action="store_true",
        help="Skip 100-row sample size verification gate (for testing purposes)",
    )
    return parser.parse_args()


def load_and_validate_labeled_data(csv_path: str, skip_sample_gate: bool = False) -> pd.DataFrame:
    """Load labeled dataset, drop unannotated/unusable rows, and strictly validate label syntax and balance.
    
    Accepts columns:
    - claim
    - evidence_top1 and evidence_top2 (preferred; used directly without re-splitting)
    - evidence (backward compatibility)
    - label (must be strictly 0 or 1; blank or '?' are dropped and counted)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labeled CSV not found at '{csv_path}'.\n"
            "Phase 3 Part D requires collecting and human-labeling claims in data/validation_claims_labeled.csv first."
        )

    df_raw = pd.read_csv(csv_path)

    # Normalize column names
    col_map = {}
    for col in df_raw.columns:
        c_lower = col.strip().lower()
        if c_lower in ("claim", "hypothesis", "statement"):
            col_map[col] = "claim"
        elif c_lower in ("label", "supported", "target", "annotation"):
            col_map[col] = "label"
        elif c_lower in ("evidence_top1", "top1_evidence", "evidence_1"):
            col_map[col] = "evidence_top1"
        elif c_lower in ("evidence_top2", "top2_evidence", "evidence_2"):
            col_map[col] = "evidence_top2"
        elif c_lower in ("evidence", "passage", "context", "research"):
            col_map[col] = "evidence"
        elif c_lower in ("topic", "query", "subject"):
            col_map[col] = "topic"

    df = df_raw.rename(columns=col_map)
    if "topic" not in df.columns:
        if "report_id" in df.columns:
            df["topic"] = df["report_id"]
        else:
            df["topic"] = "default_topic"

    if "claim" not in df.columns:
        raise ValueError(f"CSV must contain 'claim' column. Found columns: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError(f"CSV must contain 'label' column. Found columns: {list(df.columns)}")

    has_top_evidence = "evidence_top1" in df.columns or "evidence_top2" in df.columns
    has_single_evidence = "evidence" in df.columns

    if not has_top_evidence and not has_single_evidence:
        raise ValueError(
            "CSV must contain either ('evidence_top1', 'evidence_top2') or a single 'evidence' column. "
            f"Found: {list(df.columns)}"
        )

    # Label validation and filtering
    dropped_unlabeled = 0
    valid_rows = []

    for idx, row in df.iterrows():
        val = row["label"]
        row_display_idx = idx + 2  # 1-indexed plus header line for easy human tracking

        # Check if empty, NaN, or '?'
        if pd.isna(val):
            dropped_unlabeled += 1
            continue

        str_val = str(val).strip()
        if str_val == "" or str_val == "?":
            dropped_unlabeled += 1
            continue

        # Strict acceptance of 0 or 1 only
        if str_val in ("0", "0.0"):
            clean_lbl = 0
        elif str_val in ("1", "1.0"):
            clean_lbl = 1
        else:
            raise ValueError(
                f"Invalid label '{val}' at row {row_display_idx}. "
                "Only 0 or 1 are accepted; blank or '?' are ignored."
            )

        row_dict = dict(row)
        row_dict["label"] = int(clean_lbl)
        row_dict["claim"] = str(row["claim"]).strip()
        row_dict["topic"] = str(row.get("topic", "default_topic")).strip()
        if "source" in row and pd.notna(row["source"]):
            row_dict["source"] = str(row["source"]).strip().lower()
        else:
            row_dict["source"] = "report"

        # Build passage list directly: do NOT re-split with split_evidence
        passages: List[str] = []
        if has_top_evidence:
            for ev_col in ("evidence_top1", "evidence_top2"):
                if ev_col in row and pd.notna(row[ev_col]):
                    ev_text = str(row[ev_col]).strip()
                    if ev_text:
                        passages.append(ev_text)
        elif has_single_evidence and pd.notna(row["evidence"]):
            ev_text = str(row["evidence"]).strip()
            if ev_text:
                passages.append(ev_text)

        row_dict["passages"] = passages
        valid_rows.append(row_dict)

    df_clean = pd.DataFrame(valid_rows)

    if dropped_unlabeled > 0:
        print(f"Dropped {dropped_unlabeled} unannotated rows (blank or '?').")

    if len(df_clean) == 0:
        raise ValueError(f"No usable labeled rows found in {csv_path}.")

    # Label balance check
    labels = df_clean["label"].to_numpy()
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    total_usable = len(df_clean)

    print(f"Label Balance: Total Usable={total_usable} | Class 1 (Supported)={n_pos} | Class 0 (Unsupported)={n_neg}")

    if not skip_sample_gate:
        if total_usable < 100:
            raise ValueError(
                f"Dataset has {total_usable} usable rows (< 100 required). "
                "The protocol requires at least 100 human-labeled rows to report evaluation metrics."
            )
        if n_pos < 20 or n_neg < 20:
            raise ValueError(
                f"Severe class imbalance: Class 1 has {n_pos} rows, Class 0 has {n_neg} rows. "
                "Both classes must have at least 20 rows to compute stratified metrics reliably."
            )

    return df_clean


def score_dataset_lr(df: pd.DataFrame, lr_path: str = DEFAULT_LR_PATH, model_stub: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Score claims with Logistic Regression baseline."""
    if model_stub is not None:
        model = model_stub
    else:
        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"LR model artifact not found at '{lr_path}'. Run 'python -m faithfulness.train' first.")
        artifact = joblib.load(lr_path)
        model = artifact["model"]

    probs = []
    top1_cosines = []

    print(f"Embedding and scoring {len(df)} samples with Logistic Regression backend...")
    for _, row in df.iterrows():
        claim = row["claim"]
        passages = row["passages"]

        c_emb = embed([claim])[0]
        if passages:
            p_embs = embed(passages)
            feats, top1_sim, _ = build_claim_features(c_emb, p_embs, k=2)
            prob = float(model.predict_proba(feats.reshape(1, -1))[0, 1])
        else:
            feats, top1_sim, _ = build_claim_features(c_emb, np.zeros((0, len(c_emb))), k=2)
            prob = 0.0

        probs.append(prob)
        top1_cosines.append(top1_sim)

    return np.array(probs, dtype=np.float32), np.array(top1_cosines, dtype=np.float32)


def score_dataset_cross_encoder(df: pd.DataFrame, model_stub: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Score claims with DistilBERT Cross-Encoder."""
    from faithfulness.cross_encoder import score_claims

    probs = []
    top1_cosines = []

    print(f"Scoring {len(df)} samples with Cross-Encoder backend...")
    for _, row in df.iterrows():
        claim = row["claim"]
        passages = row["passages"]

        scored = score_claims([claim], passages, k=3, model=model_stub)
        probs.append(scored[0]["p_supported"])
        top1_cosines.append(scored[0]["top1_cosine"])

    return np.array(probs, dtype=np.float32), np.array(top1_cosines, dtype=np.float32)


def optimize_tau(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Sweep threshold tau in [0.05, 0.95] to maximize macro F1 on the dev set."""
    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.05, 0.95, 91):
        preds = (scores >= tau).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_tau = float(tau)
    return best_tau, float(best_f1)


def evaluate_backend(
    csv_path: str = DEFAULT_CSV_PATH,
    backend: str = "lr",
    output_path: str = None,
    seed: int = 42,
    skip_sample_gate: bool = False,
    model_stub: Any = None,
) -> Dict[str, Any]:
    df = load_and_validate_labeled_data(csv_path, skip_sample_gate=skip_sample_gate)
    labels = df["label"].to_numpy()

    print("=" * 70)
    print(f"FAITHFULNESS EVALUATION — BACKEND: {backend.upper()}")
    print("=" * 70)

    # Split dev / heldout by topic (StratifiedGroupKFold, 2 splits, grouped on topic)
    groups = df["topic"].astype(str).to_numpy()
    if len(np.unique(groups)) >= 2:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        dev_idx, heldout_idx = next(sgkf.split(df, y=labels, groups=groups))
    else:
        indices = np.arange(len(df))
        dev_idx, heldout_idx = train_test_split(
            indices,
            test_size=0.5,
            random_state=seed,
            stratify=labels,
        )

    y_dev = labels[dev_idx]
    y_heldout = labels[heldout_idx]
    dev_topics = sorted(list(set(groups[dev_idx])))
    heldout_topics = sorted(list(set(groups[heldout_idx])))
    print(f"Split: Dev={len(dev_idx)} rows ({len(dev_topics)} topics), Heldout={len(heldout_idx)} rows ({len(heldout_topics)} topics)")
    print(f"  Dev topics:     {dev_topics}")
    print(f"  Heldout topics: {heldout_topics}")

    # Score full dataset with selected backend
    if backend == "lr":
        model_source = DEFAULT_LR_PATH
        probs, cos_sims = score_dataset_lr(df, lr_path=DEFAULT_LR_PATH, model_stub=model_stub)
    elif backend == "cross_encoder":
        ce_env = os.getenv("FAITHFULNESS_CE_PATH")
        if not ce_env and model_stub is None:
            raise RuntimeError(
                "FAITHFULNESS_CE_PATH environment variable is unset in .env! "
                "Must be set to the Hub repository ID (e.g. 'vaibhavarya/distilbert-faithfulness-nli') "
                "or valid checkpoint."
            )
        smoke_dirs = ["cross_encoder", "_smoke_cross_encoder_unused"]
        for s_dir in smoke_dirs:
            smoke_path = os.path.join(MODELS_DIR, s_dir)
            if ce_env and os.path.exists(ce_env) and os.path.abspath(ce_env) == os.path.abspath(smoke_path):
                raise RuntimeError(
                    f"FAITHFULNESS_CE_PATH resolved to smoke-test directory '{smoke_path}'! "
                    "Refusing to evaluate. Must point to Hub model vaibhavarya/distilbert-faithfulness-nli."
                )
        probs, cos_sims = score_dataset_cross_encoder(df, model_stub=model_stub)
        from faithfulness.cross_encoder import get_loaded_source
        model_source = get_loaded_source() or ce_env or "stub"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"\nModel Source Loaded: {model_source}")

    probs_dev = probs[dev_idx]
    probs_heldout = probs[heldout_idx]
    cos_dev = cos_sims[dev_idx]
    cos_heldout = cos_sims[heldout_idx]

    # 1. Tune tau on DEV ONLY
    best_tau, dev_macro_f1 = optimize_tau(y_dev, probs_dev)
    print(f"\n[DEV TUNING] Selected tau={best_tau:.3f} (Dev Macro-F1: {dev_macro_f1:.4f})")

    # 2. Evaluate model on HELDOUT ONLY using tuned tau
    preds_heldout = (probs_heldout >= best_tau).astype(int)
    acc = float(accuracy_score(y_heldout, preds_heldout))
    prec = float(precision_score(y_heldout, preds_heldout, zero_division=0))
    rec = float(recall_score(y_heldout, preds_heldout, zero_division=0))
    f1 = float(f1_score(y_heldout, preds_heldout, average="binary", zero_division=0))
    macro_f1 = float(f1_score(y_heldout, preds_heldout, average="macro", zero_division=0))
    cm = confusion_matrix(y_heldout, preds_heldout).tolist()

    # Bootstrap 95% CI (1000 resamples, fixed seed) for heldout macro-F1
    rng_boot = np.random.RandomState(seed)
    n_boot = len(y_heldout)
    boot_f1s = []
    for _ in range(1000):
        boot_idx = rng_boot.choice(n_boot, size=n_boot, replace=True)
        boot_f1s.append(f1_score(y_heldout[boot_idx], preds_heldout[boot_idx], average="macro", zero_division=0))
    ci_lower = float(np.percentile(boot_f1s, 2.5))
    ci_upper = float(np.percentile(boot_f1s, 97.5))

    # Calibration calculation: mean p_supported for label 1 vs label 0 on heldout
    calib_mean_1 = float(np.mean(probs_heldout[y_heldout == 1])) if np.sum(y_heldout == 1) > 0 else 0.0
    calib_mean_0 = float(np.mean(probs_heldout[y_heldout == 0])) if np.sum(y_heldout == 0) > 0 else 0.0

    # 3. Baselines on the SAME HELDOUT ROWS
    # Baseline A: Always "supported" (all 1s)
    always_1_preds = np.ones_like(y_heldout)
    b_always_acc = float(accuracy_score(y_heldout, always_1_preds))
    b_always_f1 = float(f1_score(y_heldout, always_1_preds, average="macro", zero_division=0))

    # Baseline B: Cosine-only threshold (tune tau_cos on DEV, test on HELDOUT)
    best_tau_cos, dev_cos_f1 = optimize_tau(y_dev, cos_dev)
    cos_preds_heldout = (cos_heldout >= best_tau_cos).astype(int)
    b_cos_acc = float(accuracy_score(y_heldout, cos_preds_heldout))
    b_cos_f1 = float(f1_score(y_heldout, cos_preds_heldout, average="macro", zero_division=0))

    print("\n" + "=" * 70)
    print(f"HELDOUT RESULTS ({backend.upper()}) [N={len(y_heldout)}]:")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1 (Binary):      {f1:.4f}")
    print(f"  Macro-F1:         {macro_f1:.4f} (95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"  Confusion Matrix: {cm}")
    print(f"  Calibration:      Mean p_supported: Label 1={calib_mean_1:.4f} | Label 0={calib_mean_0:.4f}")

    # Per-source breakdown on heldout split
    heldout_sources = (
        df["source"].iloc[heldout_idx].astype(str).str.lower().to_numpy()
        if "source" in df.columns
        else np.array(["report"] * len(heldout_idx))
    )
    unique_sources = sorted(list(set(heldout_sources)))
    metrics_by_source = {}
    print("\nHELDOUT METRICS BY SOURCE:")
    for src in unique_sources:
        src_mask = (heldout_sources == src)
        src_y = y_heldout[src_mask]
        src_preds = preds_heldout[src_mask]
        if len(src_y) > 0:
            src_acc = float(accuracy_score(src_y, src_preds))
            src_macro_f1 = float(f1_score(src_y, src_preds, average="macro", zero_division=0))
            metrics_by_source[src] = {
                "n": int(len(src_y)),
                "accuracy": src_acc,
                "macro_f1": src_macro_f1,
            }
            print(f"  Source '{src}' [N={len(src_y)}]: Macro-F1={src_macro_f1:.4f}, Accuracy={src_acc:.4f}")

    # Per-topic heldout breakdown (indicative due to small n)
    heldout_topics_arr = groups[heldout_idx]
    metrics_by_topic = {}
    print("\nPER-TOPIC HELDOUT METRICS (indicative due to small n):")
    for top in heldout_topics:
        top_mask = (heldout_topics_arr == top)
        top_y = y_heldout[top_mask]
        top_preds = preds_heldout[top_mask]
        top_acc = float(accuracy_score(top_y, top_preds)) if len(top_y) > 0 else 0.0
        top_f1 = float(f1_score(top_y, top_preds, average="macro", zero_division=0)) if len(top_y) > 0 else 0.0
        metrics_by_topic[top] = {
            "n": int(len(top_y)),
            "accuracy": top_acc,
            "macro_f1": top_f1,
            "note": "indicative",
        }
        print(f"  {top[:50]:<50} [N={len(top_y)}]: Acc={top_acc:.4f}, Macro-F1={top_f1:.4f}")

    # Error analysis CSV export
    heldout_df = df.iloc[heldout_idx].copy()
    heldout_df["p_supported"] = probs_heldout
    heldout_df["predicted"] = preds_heldout

    error_types = []
    confidences = []
    for _, r in heldout_df.iterrows():
        lbl = int(r["label"])
        prd = int(r["predicted"])
        p = float(r["p_supported"])
        if prd == 1 and lbl == 0:
            error_types.append("FP")
            confidences.append(p)
        elif prd == 0 and lbl == 1:
            error_types.append("FN")
            confidences.append(1.0 - p)
        else:
            error_types.append("")
            confidences.append(0.0)
    heldout_df["error_type"] = error_types
    heldout_df["confidence"] = confidences

    error_df = heldout_df[heldout_df["error_type"] != ""].sort_values(by="confidence", ascending=False)
    if "evidence_top1" not in error_df.columns and "evidence" in error_df.columns:
        error_df["evidence_top1"] = error_df["evidence"]
    cols_to_save = ["claim", "evidence_top1", "label", "p_supported", "predicted", "error_type"]
    error_csv_path = os.path.join(MODELS_DIR, f"error_analysis_{backend}.csv")
    error_df[cols_to_save].to_csv(error_csv_path, index=False)
    print(f"\nSaved error analysis to: {error_csv_path} ({len(error_df)} error rows)")

    # Distribution of p_supported per class on heldout
    probs_1 = probs_heldout[y_heldout == 1]
    probs_0 = probs_heldout[y_heldout == 0]
    m1, s1, min1, max1 = float(np.mean(probs_1)), float(np.std(probs_1)), float(np.min(probs_1)), float(np.max(probs_1))
    m0, s0, min0, max0 = float(np.mean(probs_0)), float(np.std(probs_0)), float(np.min(probs_0)), float(np.max(probs_0))

    print("\nHELD-OUT P_SUPPORTED DISTRIBUTION PER CLASS:")
    print(f"  Class 1 (Supported,   N={len(probs_1)}): mean={m1:.4f}, std={s1:.4f}, min={min1:.4f}, max={max1:.4f}")
    print(f"  Class 0 (Unsupported, N={len(probs_0)}): mean={m0:.4f}, std={s0:.4f}, min={min0:.4f}, max={max0:.4f}")

    # Top 10 worst FP and FN
    fps = error_df[error_df["error_type"] == "FP"].sort_values(by="confidence", ascending=False)
    fns = error_df[error_df["error_type"] == "FN"].sort_values(by="confidence", ascending=False)

    print(f"\nTOP 10 WORST FALSE POSITIVES (Predicted 1, Label 0) [Total FP={len(fps)}]:")
    for rank, (_, row_err) in enumerate(fps.head(10).iterrows(), 1):
        print(f"  {rank}. [p_supp={row_err['p_supported']:.4f}] Claim: {row_err['claim'][:80]}...")
        print(f"     Evidence: {str(row_err['evidence_top1'])[:100]}...")

    print(f"\nTOP 10 WORST FALSE NEGATIVES (Predicted 0, Label 1) [Total FN={len(fns)}]:")
    for rank, (_, row_err) in enumerate(fns.head(10).iterrows(), 1):
        print(f"  {rank}. [p_supp={row_err['p_supported']:.4f}] Claim: {row_err['claim'][:80]}...")
        print(f"     Evidence: {str(row_err['evidence_top1'])[:100]}...")

    print("-" * 70)
    print("BASELINES (on same heldout split):")
    print(f"  Always Supported: Macro-F1={b_always_f1:.4f}, Accuracy={b_always_acc:.4f}")
    print(f"  Cosine Only:      Macro-F1={b_cos_f1:.4f}, Accuracy={b_cos_acc:.4f} (tau={best_tau_cos:.3f})")
    print("=" * 70)

    # Decision gate check
    passed_gate = macro_f1 >= 0.70
    status_str = "PASSED (>= 0.70)" if passed_gate else "FAILED (< 0.70 gate)"
    print(f"0.70 Macro-F1 Gate: {status_str}")

    results = {
        "backend": backend,
        "model_source": model_source,
        "n_total": len(df),
        "n_dev": len(dev_idx),
        "n_heldout": len(heldout_idx),
        "dev_topics": dev_topics,
        "heldout_topics": heldout_topics,
        "selected_tau": best_tau,
        "dev_macro_f1": dev_macro_f1,
        "heldout_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_binary": f1,
            "macro_f1": macro_f1,
            "bootstrap_95_ci": [ci_lower, ci_upper],
            "confusion_matrix": cm,
        },
        "score_distribution_heldout": {
            "class_1": {"n": int(len(probs_1)), "mean": m1, "std": s1, "min": min1, "max": max1},
            "class_0": {"n": int(len(probs_0)), "mean": m0, "std": s0, "min": min0, "max": max0},
        },
        "heldout_metrics_by_source": metrics_by_source,
        "heldout_metrics_by_topic": metrics_by_topic,
        "calibration": {
            "mean_p_supported_label_1": calib_mean_1,
            "mean_p_supported_label_0": calib_mean_0,
        },
        "baselines_heldout": {
            "always_supported": {
                "macro_f1": b_always_f1,
                "accuracy": b_always_acc,
            },
            "cosine_threshold": {
                "tau": best_tau_cos,
                "macro_f1": b_cos_f1,
                "accuracy": b_cos_acc,
            },
        },
        "gate_passed": passed_gate,
        "error_analysis_file": error_csv_path,
    }

    if output_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        output_path = os.path.join(MODELS_DIR, f"indomain_eval_{backend}.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation report to: {output_path}")

    return results


def main():
    args = parse_args()
    evaluate_backend(
        csv_path=args.csv,
        backend=args.backend,
        output_path=args.output,
        seed=args.seed,
        skip_sample_gate=args.skip_sample_gate,
    )


if __name__ == "__main__":
    main()
