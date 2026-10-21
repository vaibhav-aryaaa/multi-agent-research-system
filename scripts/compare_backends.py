"""Compare Logistic Regression vs DistilBERT Cross-Encoder faithfulness backends.

Reads indomain_eval_lr.json and indomain_eval_cross_encoder.json,
prints a comparison table, identifies the winning backend based strictly
on DEV macro F1, and verifies the 0.70 macro-F1 decision gate.
"""

import os
import sys
import json
import datetime
from typing import Dict, Any, Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faithfulness", "models")


def load_backend_eval(backend_name: str) -> Optional[Dict[str, Any]]:
    """Try loading eval_<name>.json first, then indomain_eval_<name>.json."""
    paths = [
        os.path.join(MODELS_DIR, f"eval_{backend_name}.json"),
        os.path.join(MODELS_DIR, f"indomain_eval_{backend_name}.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[WARNING] Failed to read {p}: {exc}")
    return None


def main():
    print("=" * 80)
    print("FAITHFULNESS CLASSIFIER BACKEND COMPARISON (IN-DOMAIN)")
    print("=" * 80)

    lr_data = load_backend_eval("lr")
    ce_data = load_backend_eval("cross_encoder")

    if lr_data is None and ce_data is None:
        print(f"No in-domain evaluation reports found in {MODELS_DIR}.")
        print("Run evaluations first using:")
        print("  python -m faithfulness.evaluate --backend lr --eval_output faithfulness/models/eval_lr.json")
        print("  python -m faithfulness.evaluate --backend cross_encoder --eval_output faithfulness/models/eval_cross_encoder.json")
        sys.exit(1)

    print(f"{'Backend':<18} | {'Dev Macro-F1':<14} | {'Heldout Macro-F1':<18} | {'Heldout Acc':<12} | {'0.70 Gate'}")
    print("-" * 80)

    rows = []
    for name, data in [("lr", lr_data), ("cross_encoder", ce_data)]:
        if data is None:
            print(f"{name:<18} | {'[NOT EVALUATED]':<14} | {'[NOT EVALUATED]':<18} | {'-':<12} | -")
            continue

        dev_f1 = data.get("dev_macro_f1", 0.0)
        heldout_f1 = data.get("heldout_metrics", {}).get("macro_f1", 0.0)
        heldout_acc = data.get("heldout_metrics", {}).get("accuracy", 0.0)
        tau = data.get("selected_tau", 0.5)
        gate_str = "PASS" if heldout_f1 >= 0.70 else "FAIL (< 0.70)"

        rows.append({
            "name": name,
            "tau": tau,
            "dev_f1": dev_f1,
            "heldout_f1": heldout_f1,
            "heldout_acc": heldout_acc,
            "gate": gate_str,
            "raw": data,
        })
        print(f"{name:<18} | {dev_f1:<14.4f} | {heldout_f1:<18.4f} | {heldout_acc:<12.4f} | {gate_str}")

    print("=" * 80)

    if not rows:
        return

    # Select winning backend based strictly on higher DEV macro F1
    # Rule: Never choose by heldout!
    best_candidate = max(rows, key=lambda r: r["dev_f1"])
    passes_gate = bool(best_candidate["heldout_f1"] >= 0.70)

    print(f"\nSelection Decision (strictly by DEV Macro-F1):")
    print(f"-> Selected Backend: '{best_candidate['name']}' (Dev Macro-F1: {best_candidate['dev_f1']:.4f})")
    print(f"-> Selected Tau:     {best_candidate['tau']:.3f}")
    print(f"-> Unbiased Heldout Performance: Macro-F1={best_candidate['heldout_f1']:.4f}, Accuracy={best_candidate['heldout_acc']:.4f}")

    # Decision Gate Check
    if passes_gate:
        print(f"\n[DECISION GATE PASSED] Selected backend '{best_candidate['name']}' meets the >= 0.70 macro-F1 requirement.")
        print(f"Configure in config.py: CLASSIFIER_BACKEND = '{best_candidate['name']}'")
    else:
        print(f"\n[DECISION GATE FAILED] Selected backend '{best_candidate['name']}' scored {best_candidate['heldout_f1']:.4f} (< 0.70 threshold).")
        print("Per protocol: Transparently report this result without hiding the number.")

    # Write B4 decision file
    config_data = {
        "backend": best_candidate["name"],
        "tau": best_candidate["tau"],
        "dev_macro_f1": best_candidate["dev_f1"],
        "heldout_macro_f1": best_candidate["heldout_f1"],
        "passes_gate": passes_gate,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    config_path = os.path.join(MODELS_DIR, "classifier_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"\nWrote classifier configuration to: {config_path}")
    print(json.dumps(config_data, indent=2))


if __name__ == "__main__":
    main()
