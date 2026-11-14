# Evaluation and Experimental Results

This directory contains all experimental outputs, judge evaluations, and statistical artifacts across the four evaluated systems for the 20 benchmark questions (`Q01`–`Q20`).

---

## 1. Pipeline Execution Artifacts (`results/eval/`)
Contains raw generation and retry trajectories produced by the multi-agent pipeline:
- **`QXX_baseline.json`**: System 1 (single-prompt direct LLM baseline without retrieval or agents).
- **`QXX_first_pass.json`**: The shared attempt-0 generation state produced once per question and shared identically across Systems 2, 3, and 4 for experimental control.
- **`QXX_fixed.json`**: System 2 (fixed sequential retrieve-and-write pipeline, locked to attempt 0 without retries).
- **`QXX_adaptive_llm.json`**: System 3 (adaptive pipeline where an LLM judge decides whether to accept or retry up to 3 times based on critic feedback).
- **`QXX_adaptive_classifier.json`**: System 4 (adaptive pipeline where a fine-tuned DistilBERT cross-encoder NLI judge evaluates claim-level grounding to decide whether to accept or retry up to 3 times).

Each file records the topic, final report, retrieved evidence passages (`contexts`), retry count, critic score, attempt history, and token usage breakdown.

---

## 2. RAGAS Grounding Evaluations (`results/ragas/`)
Contains transactional per-sample RAGAS evaluation JSON records produced with local semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`):
- **`QXX_<system>.json`**: Grounding evaluation for each question and system pairing (80 files total).
  - **Judge Model Recorded**: `gemini-3.1-flash-lite` (verified identical across all 80 JSON records).
  - **Contexts Source**: System 1 borrows attempt-0 contexts from System 2 (`borrowed_from_S2`) for Faithfulness evaluation only; Systems 2, 3, and 4 evaluate against native pipeline evidence (`system_contexts`).
  - **Metrics Evaluated (Phase A)**: RAGAS Faithfulness and Answer Relevancy.
  - **Unsupported Claims**: Extracted statement-level NLI traces with claim text, judge reasoning, and evidence passages.
- **`summary.csv`**: Flattened table summarizing metric scores, execution seconds, requests used, and contexts trimming status for all 80 runs.
- **`usage_log.jsonl`**: Timestamped ledger tracking rolling API request consumption against the daily governor budget.

---

## 3. Statistical Analysis & Confound Checks (`results/analysis/`)
Contains offline statistical comparisons, confound checks, and publication figures:
- **`per_system.csv`**: Mean, median, standard deviation, and null counts for Faithfulness and Relevancy, alongside mean critic scores, mean retries, and total tokens.
- **`paired.csv`**: Paired differences (S2->S3, S2->S4, S3->S4), 10,000-resample bootstrap 95% CIs (seed 0), two-sided Wilcoxon signed-rank tests (dropping zero-difference pairs), and step-down Holm-Bonferroni corrections.
- **`cost.csv`**: Total token usage and marginal token expenditure per +0.1 gain in Faithfulness over S2.
- **`retry_effect.csv`**: Pre- and post-retry critic scores alongside Faithfulness for questions where retries occurred.
- **`judge_disagreement.csv`**: Per-question comparison of S3 vs S4 retry decisions and final Faithfulness.
- **`dose_response.csv` & `dose_response_correlations.csv`**: Spearman rank correlations assessing the relationship between Faithfulness and retries, passage counts, and character lengths.
- **`matched_s3_s4.csv` & `matched_s3_s4_summary.csv`**: Direct comparisons isolating questions where S3 and S4 underwent identical retry counts vs differing retry counts.
- **`judge_agreement.csv` & `judge_agreement_summary.csv`**: Categorization of attempt-0 decisions between the LLM judge and classifier judge.
- **`figures/`**: 200 DPI colorblind-safe publication figures (`fig_faithfulness_by_system.png`, `fig_relevancy_by_system.png`, `fig_paired_faithfulness.png`, `fig_cost_vs_faithfulness.png`, `fig_retries_vs_gain.png`).
