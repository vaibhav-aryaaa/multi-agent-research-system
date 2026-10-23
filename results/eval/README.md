# Evaluation Results Directory

This directory stores experimental evaluation outputs from `scripts/run_eval.py` across the 25 benchmark questions (`Q01` through `Q25`).

## File Naming Convention

For each evaluation question `<qid>`:
- **`<qid>_baseline.json`**: System 1 (Direct single-prompt LLM baseline without search, scraping, or agents).
- **`<qid>_fixed.json`**: System 2 (Fixed linear multi-agent pipeline; final report is the shared attempt-0 report with 0 retries).
- **`<qid>_adaptive_llm.json`**: System 3 (Adaptive bounded loop-back with LLM critic judge and up to 3 retries).
- **`<qid>_adaptive_classifier.json`**: System 4 (Adaptive bounded loop-back with DistilBERT NLI Cross-Encoder faithfulness classifier judge and up to 3 retries).

### Diagnostics and Logs
- **`<qid>_ERROR.json`**: Generated if an unhandled exception occurred during question evaluation, capturing the error message and full stack trace.
- **`usage_log.jsonl`**: Append-only log recording every completed LLM call with an ISO-8601 timestamp and token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`), used for rolling 24-hour rate limit management.

## JSON Schema

Each evaluation result JSON adheres to the `save_run` schema with added experimental identifiers:
- `qid`: Evaluation question identifier (`Q01`..`Q25`).
- `system`: One of `baseline`, `fixed`, `adaptive_llm`, `adaptive_classifier`.
- `topic`: Exact question topic string.
- `model_name`: Pinned generator/critic model (e.g., `openai/gpt-oss-120b`).
- `timestamp`: Execution timestamp.
- `final_report`: Complete Markdown text of the final generated report.
- `contexts`: List of fine-grained evidence passages gathered by the pipeline (empty for baseline).
- `retry_count`: Total number of retry iterations executed (0 for S1 and S2; 0–3 for S3 and S4).
- `quality_threshold`: Threshold used for evaluation decisions.
- `queries_used`: Web search queries issued during the run.
- `used_urls`: Web URLs scraped during the run.
- `final_score`: Critic score or faithfulness metric on the final report.
- `attempts`: Detailed telemetry per attempt (report, critic score, critique, judge verdict, routing decision).
- `token_usage`: Total token consumption for the system.
- `token_usage_first_pass`: Token consumption during the shared attempt-0 pass.
- `token_usage_retries`: Token consumption incurred during retry loops.
