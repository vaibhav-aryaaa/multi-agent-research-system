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
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from faithfulness.claims import split_evidence
from faithfulness.embedder import embed
from faithfulness.features import build_claim_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LR_PATH = os.path.join(MODELS_DIR, "faithfulness_lr.joblib")
DEFAULT_CSV_PATH = os.path.join("data", "dev_claims_labeled.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faithfulness Classifier on In-Domain Labeled Data")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to human-labeled claims CSV (claim, evidence, label)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["lr", "cross_encoder"],
        default="lr",
        help="Classifier backend: 'lr' or 'cross_encoder' (default: lr)",
    )
    parser.add_argument(
        "--output",
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
    return parser.parse_args()


def load_labeled_data(csv_path: str) -> pd.DataFrame:
    """Load labeled dataset and validate required fields."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labeled CSV not found at '{csv_path}'.\n"
            "Phase 3 Part D requires collecting and human-labeling claims in data/dev_claims_labeled.csv first."
        )

    df = pd.read_csv(csv_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ("claim", "hypothesis", "statement"):
            col_map[col] = "claim"
        elif c_lower in ("evidence", "passage", "context", "research"):
            col_map[col] = "evidence"
        elif c_lower in ("label", "supported", "target", "annotation"):
            col_map[col] = "label"

    df = df.rename(columns=col_map)
    required = ["claim", "evidence", "label"]
    for req in required:
        if req not in df.columns:
            raise ValueError(f"CSV must contain '{req}' column. Found columns: {list(df.columns)}")

    # Clean labels to binary 0 or 1
    def clean_label(val):
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return int(val)
        s = str(val).strip().lower()
        if s in ("1", "true", "supported", "entailment", "yes"):
            return 1
        return 0

    df["label"] = df["label"].apply(clean_label)
    df["claim"] = df["claim"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    return df


def score_dataset_lr(df: pd.DataFrame, lr_path: str = DEFAULT_LR_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Score claims with Logistic Regression baseline."""
    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"LR model artifact not found at '{lr_path}'. Run 'python -m faithfulness.train' first.")

    artifact = joblib.load(lr_path)
    model = artifact["model"]

    probs = []
    top1_cosines = []

    print(f"Embedding and scoring {len(df)} samples with Logistic Regression backend...")
    for _, row in df.iterrows():
        claim = row["claim"]
        evidence_text = row["evidence"]

        # Parse passages from evidence
        passages = split_evidence("", evidence_text)
        if not passages:
            passages = [evidence_text] if evidence_text else []

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


def score_dataset_cross_encoder(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Score claims with DistilBERT Cross-Encoder."""
    from faithfulness.cross_encoder import score_claims

    probs = []
    top1_cosines = []

    print(f"Scoring {len(df)} samples with Cross-Encoder backend...")
    for _, row in df.iterrows():
        claim = row["claim"]
        evidence_text = row["evidence"]

        passages = split_evidence("", evidence_text)
        if not passages:
            passages = [evidence_text] if evidence_text else []

        scored = score_claims([claim], passages, k=3)
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
) -> Dict[str, Any]:
    df = load_labeled_data(csv_path)
    labels = df["label"].to_numpy()

    print("=" * 70)
    print(f"FAITHFULNESS EVALUATION — BACKEND: {backend.upper()}")
    print("=" * 70)
    print(f"Total labeled samples: {len(df)} (Positives: {np.sum(labels == 1)}, Negatives: {np.sum(labels == 0)})")

    # Stratified 50/50 dev / heldout split
    indices = np.arange(len(df))
    dev_idx, heldout_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=seed,
        stratify=labels,
    )

    y_dev = labels[dev_idx]
    y_heldout = labels[heldout_idx]
    print(f"Split: Dev={len(dev_idx)} rows, Heldout={len(heldout_idx)} rows")

    # Score full dataset with selected backend
    if backend == "lr":
        probs, cos_sims = score_dataset_lr(df)
    elif backend == "cross_encoder":
        probs, cos_sims = score_dataset_cross_encoder(df)
    else:
        raise ValueError(f"Unknown backend: {backend}")

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
    print(f"  Macro-F1:         {macro_f1:.4f}")
    print(f"  Confusion Matrix: {cm}")
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
        "n_total": len(df),
        "n_dev": len(dev_idx),
        "n_heldout": len(heldout_idx),
        "selected_tau": best_tau,
        "dev_macro_f1": dev_macro_f1,
        "heldout_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_binary": f1,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
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
    )


if __name__ == "__main__":
    main()
