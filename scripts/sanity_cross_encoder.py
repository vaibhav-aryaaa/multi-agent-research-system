"""Diagnostic sanity check for DistilBERT cross-encoder faithfulness classifier.

Validates inference consistency against training on GLUE MNLI validation_matched
and tests fixed canonical premise-hypothesis pairs.

Features:
- Samples n balanced MNLI rows (default: 500, 250 entailment / 250 non-entailment).
- Evaluates via (a) score_claims([hypothesis], [premise]) from faithfulness.cross_encoder
  and (b) direct transformers tokenizer + model inference.
- Asserts parity: max probability difference <= 1e-4 and accuracy >= 0.78 (MNLI expected ~0.84).
- Tests 4 fixed domain / canonical pairs (Guitar entailment, Cat contradiction, Casimir claims).
- Exports metrics to docs/diagnostics/cross_encoder_sanity.json.
- Exits with code 1 if accuracy < 0.78 or parity difference > 1e-4.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from dotenv import load_dotenv

# Ensure repo root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from faithfulness.cross_encoder import score_claims, load_cross_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Encoder Inference Sanity Check")
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of balanced MNLI validation rows (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selection (default: 42)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "docs", "diagnostics", "cross_encoder_sanity.json"),
        help="Path to save diagnostic results JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score

    print("=" * 75)
    print("DISTILBERT CROSS-ENCODER INFERENCE SANITY CHECK")
    print("=" * 75)
    print(f"Sample size: {args.n} (balanced)")
    print(f"Random seed: {args.seed}")

    model, tokenizer, device = load_cross_encoder()
    model_path = os.getenv("FAITHFULNESS_CE_PATH", "default")
    print(f"Model: {model_path} on {device}")

    # 1. Load and balance GLUE MNLI validation_matched
    print("\n[Step 1] Loading GLUE MNLI validation_matched dataset...")
    raw_val = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")

    target_per_class = args.n // 2
    entailment_pool = [r for r in raw_val if r["label"] == 0]
    non_entailment_pool = [r for r in raw_val if r["label"] in (1, 2)]

    rng = np.random.RandomState(args.seed)
    chosen_ent_idx = rng.choice(len(entailment_pool), size=target_per_class, replace=False)
    chosen_non_idx = rng.choice(len(non_entailment_pool), size=target_per_class, replace=False)

    selected_ent = [entailment_pool[i] for i in chosen_ent_idx]
    selected_non = [non_entailment_pool[i] for i in chosen_non_idx]
    selected_rows = selected_ent + selected_non

    y_true = np.array([1] * target_per_class + [0] * target_per_class)
    premises = [r["premise"] for r in selected_rows]
    hypotheses = [r["hypothesis"] for r in selected_rows]

    print(f"Selected {len(selected_rows)} samples: {target_per_class} entailment (1), {target_per_class} non-entailment (0)")

    # 2. Path (a): faithfulness.cross_encoder.score_claims([hypothesis], [premise])
    print("\n[Step 2] Running Path (a): score_claims([hypothesis], [premise])...")
    probs_a = []
    for hyp, prem in zip(hypotheses, premises):
        res = score_claims([hyp], [prem], model=model, tokenizer=tokenizer)
        probs_a.append(res[0]["p_supported"])
    probs_a = np.array(probs_a, dtype=np.float32)
    preds_a = (probs_a >= 0.5).astype(int)

    acc_a = float(accuracy_score(y_true, preds_a))
    macro_f1_a = float(f1_score(y_true, preds_a, average="macro", zero_division=0))
    print(f"  Path (a) Accuracy: {acc_a:.4f} ({acc_a*100:.2f}%)")
    print(f"  Path (a) Macro-F1: {macro_f1_a:.4f}")

    # 3. Path (b): Plain transformers pipeline directly
    print("\n[Step 3] Running Path (b): Direct transformers pipeline (premise as text, hypothesis as text_pair)...")
    enc = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation="only_first",
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits_b = model(**enc).logits
        probs_b = torch.softmax(logits_b, dim=-1)[:, 1].cpu().numpy().astype(np.float32)

    preds_b = (probs_b >= 0.5).astype(int)
    acc_b = float(accuracy_score(y_true, preds_b))
    macro_f1_b = float(f1_score(y_true, preds_b, average="macro", zero_division=0))
    print(f"  Path (b) Accuracy: {acc_b:.4f} ({acc_b*100:.2f}%)")
    print(f"  Path (b) Macro-F1: {macro_f1_b:.4f}")

    # 4. Parity difference
    max_abs_diff = float(np.max(np.abs(probs_b - probs_a)))
    print(f"\n[Step 4] Parity check: max |probs_b - probs_a| = {max_abs_diff:.6e}")

    # 5. Fixed domain & canonical pairs
    print("\n[Step 5] Fixed Test Pairs (evidence -> claim):")
    casimir_ev = "The Casimir effect is an attractive force between two uncharged conducting plates in a vacuum."
    casimir_c1 = "Two uncharged metal plates in a vacuum attract each other."
    casimir_c2 = "The Casimir effect makes uncharged plates repel each other strongly."

    fixed_pairs_spec = [
        ("guitar_entailment", "A man is playing a guitar on stage.", "A man is playing an instrument."),
        ("cat_contradiction", "The cat is sleeping on the couch.", "The cat is running outside."),
        ("casimir_claim_a", casimir_ev, casimir_c1),
        ("casimir_claim_b", casimir_ev, casimir_c2),
    ]

    fixed_pairs_results = []
    for pair_id, ev, cl in fixed_pairs_spec:
        res = score_claims([cl], [ev], model=model, tokenizer=tokenizer)
        p_sup = float(res[0]["p_supported"])
        cos_sim = float(res[0]["top1_cosine"])
        fixed_pairs_results.append({
            "pair_id": pair_id,
            "evidence": ev,
            "claim": cl,
            "p_supported": p_sup,
            "top1_cosine": cos_sim,
        })
        print(f"  - {pair_id:<18}: p_supported = {p_sup:.4f} (cosine = {cos_sim:.4f})")
        print(f"      Evidence: \"{ev}\"")
        print(f"      Claim:    \"{cl}\"")

    # 6. Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "n_samples": args.n,
        "seed": args.seed,
        "path_a_score_claims": {
            "accuracy": acc_a,
            "macro_f1": macro_f1_a,
        },
        "path_b_transformers_direct": {
            "accuracy": acc_b,
            "macro_f1": macro_f1_b,
        },
        "parity_max_abs_diff": max_abs_diff,
        "fixed_pairs": fixed_pairs_results,
        "gate_passed": bool(acc_a >= 0.78 and max_abs_diff <= 1e-4),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved sanity report to: {args.output_json}")

    # 7. Exit code gate: exit 1 if accuracy < 0.78 or parity diff > 1e-4
    if acc_a < 0.78:
        print(f"\n[FAILURE] Path (a) accuracy {acc_a:.4f} < 0.78 threshold.")
        sys.exit(1)

    if max_abs_diff > 1e-4:
        print(f"\n[FAILURE] Parity difference {max_abs_diff:.6e} > 1e-4 threshold.")
        sys.exit(1)

    print("\n[SUCCESS] Sanity check PASSED: Accuracy >= 0.78 and parity difference <= 1e-4.")
    sys.exit(0)


if __name__ == "__main__":
    main()
