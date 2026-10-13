"""Train logistic regression faithfulness classifier on GLUE MNLI.

Domain Shift Warning:
MNLI premises are clean, single-sentence assertions, whereas retrieved academic/web evidence
consists of multi-sentence passages with noisy punctuation and formatting. This model provides
a zero-shot sentence-level NLI baseline. Final reported system performance must be evaluated
on human-labeled in-domain evidence from the actual research pipeline (Part D & E).
"""

import os
import json
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from faithfulness.embedder import embed

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def batch_pair_features(hypo_embs: np.ndarray, premise_embs: np.ndarray) -> np.ndarray:
    """Vectorized calculation of [claim, evidence, abs(diff), prod, cos_sim]."""
    diff = np.abs(hypo_embs - premise_embs)
    prod = hypo_embs * premise_embs
    # Since embeddings from embedder are L2 normalized, row dot product is cosine similarity
    sims = np.sum(hypo_embs * premise_embs, axis=1, keepdims=True).astype(np.float32)
    return np.hstack([hypo_embs, premise_embs, diff, prod, sims])

def train_faithfulness_model(total_samples: int = 30000, random_seed: int = 42):
    print("=" * 70)
    print("PHASE 3 PART C: TRAINING FAITHFULNESS CLASSIFIER")
    print("=" * 70)
    print(f"Loading GLUE MNLI dataset... (subsampling {total_samples} balanced pairs)")
    
    # Load MNLI dataset
    raw_ds = load_dataset("nyu-mll/glue", "mnli", split="train")

    # Map labels: entailment (0) -> 1, neutral (1) & contradiction (2) -> 0
    # Collect indices for balanced sampling
    entailment_indices = []
    non_entailment_indices = []

    for idx, row in enumerate(raw_ds):
        label = row["label"]
        if label == 0:
            entailment_indices.append(idx)
        elif label in (1, 2):
            non_entailment_indices.append(idx)
        if len(entailment_indices) >= total_samples and len(non_entailment_indices) >= total_samples:
            break

    np.random.seed(random_seed)
    n_per_class = total_samples // 2

    chosen_pos = np.random.choice(entailment_indices, size=n_per_class, replace=False)
    chosen_neg = np.random.choice(non_entailment_indices, size=n_per_class, replace=False)

    selected_indices = np.concatenate([chosen_pos, chosen_neg])
    np.random.shuffle(selected_indices)

    subset = raw_ds.select(selected_indices)

    premises = [row["premise"] for row in subset]
    hypotheses = [row["hypothesis"] for row in subset]
    # Label: 1 for entailment, 0 for neutral / contradiction
    labels = np.array([1 if row["label"] == 0 else 0 for row in subset], dtype=np.int32)

    print(f"Subsampled {len(labels)} pairs (Positives: {np.sum(labels == 1)}, Negatives: {np.sum(labels == 0)})")
    print("Holding out 10% as public test split...")

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.10,
        random_state=random_seed,
        stratify=labels,
    )

    print("Embedding premises and hypotheses using all-MiniLM-L6-v2 on CPU...")
    # Premise acts as evidence, Hypothesis acts as claim
    premise_embs = embed(premises)
    hypo_embs = embed(hypotheses)

    print("Building sentence-pair features (claim, evidence, diff, prod, cos_sim)...")
    X = batch_pair_features(hypo_embs, premise_embs)
    y = labels

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Small validation split from train to pick C
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_seed, stratify=y_train
    )

    print("Fitting Logistic Regression with grid search over C...")
    best_c = 1.0
    best_val_f1 = -1.0
    for c_val in [0.05, 0.2, 1.0, 5.0]:
        clf_candidate = LogisticRegression(C=c_val, max_iter=1000, random_state=random_seed, solver="lbfgs")
        clf_candidate.fit(X_tr_sub, y_tr_sub)
        val_preds = clf_candidate.predict(X_val)
        val_f1 = f1_score(y_val, val_preds)
        print(f"  C={c_val:<5} -> Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_val

    print(f"Best C selected: {best_c} (Val F1: {best_val_f1:.4f}). Retraining on full training split...")
    final_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=random_seed, solver="lbfgs")
    final_clf.fit(X_train, y_train)

    # Evaluate on held-out public test split
    y_pred = final_clf.predict(X_test)
    y_prob = final_clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 70)
    print("HELD-OUT PUBLIC TEST METRICS (GLUE MNLI):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion Matrix: {cm}")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "faithfulness_lr.joblib")
    eval_path = os.path.join(MODELS_DIR, "public_eval.json")

    joblib.dump(
        {
            "model": final_clf,
            "best_C": best_c,
            "feature_dim": X.shape[1],
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "n_train_samples": len(train_idx),
            "n_test_samples": len(test_idx),
            "domain_shift_warning": (
                "MNLI premises are single sentences, while our evidence is retrieved web text. "
                "In-domain human-labeled validation is required for real task reporting."
            ),
        },
        model_path,
    )
    print(f"Saved trained classifier to: {model_path}")

    public_eval_data = {
        "dataset": "nyu-mll/glue/mnli",
        "n_samples_total": total_samples,
        "n_test_samples": len(test_idx),
        "best_C": best_c,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(public_eval_data, f, indent=2)
    print(f"Saved public evaluation metrics to: {eval_path}")

if __name__ == "__main__":
    train_faithfulness_model()
