"""Train DistilBERT cross-encoder faithfulness classifier on GLUE MNLI.

Domain Shift Warning:
MNLI premises are single sentences, whereas our evidence is retrieved web text, so the
human-labeled in-domain set decides the real performance. This model provides an end-to-end
cross-encoder NLI classifier that processes claim and evidence together to detect unsupported
statements and contradictions. Final reported system performance must be evaluated on
human-labeled in-domain evidence from the actual research pipeline (Part D & E).
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, Any

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_OUTPUT_DIR = os.path.join(MODELS_DIR, "cross_encoder")
DEFAULT_EVAL_PATH = os.path.join(MODELS_DIR, "public_eval_cross_encoder.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT Cross-Encoder on MNLI")
    parser.add_argument("--smoke", action="store_true", help="Run quick 200-sample CPU smoke test")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Hugging Face repo id to upload model")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save model")
    parser.add_argument("--eval_output", type=str, default=DEFAULT_EVAL_PATH, help="Path for evaluation metrics JSON")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and eval")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Heavy imports deferred to execution time
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    set_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("PHASE 3B: DISTILBERT CROSS-ENCODER TRAINING")
    print("=" * 70)
    print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Smoke test mode: {args.smoke}")

    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 1. Load GLUE MNLI dataset
    print("Loading GLUE MNLI dataset...")
    raw_train = load_dataset("nyu-mll/glue", "mnli", split="train")
    raw_val = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")

    target_train_samples = 200 if args.smoke else 100000
    n_per_class = target_train_samples // 2

    print(f"Selecting balanced training samples ({target_train_samples} total, {n_per_class} per class)...")
    entailment_indices = []
    non_entailment_indices = []

    for idx, row in enumerate(raw_train):
        lbl = row["label"]
        if lbl == 0:  # entailment -> supported (1)
            entailment_indices.append(idx)
        elif lbl in (1, 2):  # neutral or contradiction -> not supported (0)
            non_entailment_indices.append(idx)
        if len(entailment_indices) >= n_per_class and len(non_entailment_indices) >= n_per_class:
            break

    chosen_pos = np.random.choice(entailment_indices, size=n_per_class, replace=False)
    chosen_neg = np.random.choice(non_entailment_indices, size=n_per_class, replace=False)
    selected_indices = np.concatenate([chosen_pos, chosen_neg])
    np.random.shuffle(selected_indices)

    train_subset = raw_train.select(selected_indices)

    if args.smoke:
        val_subset = raw_val.select(range(min(50, len(raw_val))))
    else:
        val_subset = raw_val

    print(f"Training dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)} (validation_matched)")

    # 2. Tokenization and preprocessing: premise = evidence, hypothesis = claim
    # Truncation on premise side only ("only_first")
    def preprocess_batch(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            max_length=256,
            truncation="only_first",
            padding=False,
        )
        # Binary mapping: entailment (0) -> 1, neutral/contradiction (1, 2) -> 0
        tokenized["label"] = [1 if lbl == 0 else 0 for lbl in examples["label"]]
        return tokenized

    print("Tokenizing datasets (truncation on premise side only, max_length=256)...")
    train_dataset = train_subset.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_train.column_names,
    )
    val_dataset = val_subset.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_val.column_names,
    )

    # 3. Model initialization: 2 labels (0: not supported, 1: supported)
    print(f"Initializing {base_model_name} with num_labels=2...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    )

    use_fp16 = torch.cuda.is_available()
    batch_size = 16 if args.smoke else args.batch_size
    max_steps = 1 if args.smoke else -1
    num_epochs = 1 if args.smoke else args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=use_fp16,
        logging_steps=1 if args.smoke else 100,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 4. Evaluation on validation split
    print(f"Evaluating on validation split ({len(val_dataset)} rows)...")
    predictions_output = trainer.predict(val_dataset)
    logits = predictions_output.predictions
    y_true = np.array(predictions_output.label_ids)
    y_pred = np.argmax(logits, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\n" + "=" * 70)
    print("PUBLIC EVALUATION METRICS (GLUE MNLI validation_matched):")
    print(f"  Rows evaluated:   {len(val_dataset)}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1 Score:         {f1:.4f}")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"  Confusion Matrix: {cm}")
    print("=" * 70)

    # 5. Save model and evaluation JSON
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.eval_output)), exist_ok=True)

    print(f"Saving model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_data: Dict[str, Any] = {
        "model": base_model_name,
        "dataset": "nyu-mll/glue/mnli",
        "split": "validation_matched",
        "num_rows": len(val_dataset),
        "smoke": args.smoke,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "domain_shift_warning": (
            "MNLI premises are single sentences, whereas our evidence is retrieved web text. "
            "In-domain human-labeled validation is required for real task reporting."
        ),
    }

    with open(args.eval_output, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved evaluation metrics to: {args.eval_output}")

    # 6. Optional upload to Hugging Face Hub
    if args.push_to_hub:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print(f"Pushing model and tokenizer to Hugging Face Hub: {args.push_to_hub}...")
            # Note: Do not print or log the token
            model.push_to_hub(args.push_to_hub, token=hf_token)
            tokenizer.push_to_hub(args.push_to_hub, token=hf_token)
            print("Successfully uploaded to Hugging Face Hub.")
        else:
            print("[WARNING] --push_to_hub specified but HF_TOKEN environment variable is not set. Skipping upload.")


if __name__ == "__main__":
    main()
