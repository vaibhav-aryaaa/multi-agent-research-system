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

## How to run a session

The batch pipeline runner is pause-friendly and designed for short laptop sessions. Result files and first-pass checkpoints are written atomically (via temporary files and `os.replace`), preventing corrupted files if the terminal closes, the process is killed (`Ctrl-C`), or the laptop goes to sleep.

To run a timed session:
```bash
python scripts/run_eval.py --sleep 20 --daily-token-budget 195000 --max-minutes 60
```
*(About 4–5 questions per day is the free-tier limit, so ~45–60 minutes of laptop time per day is enough).*

### Controls and Sentinels
- **`--max-minutes N`**: Automatically stops the runner cleanly after finishing a question once N minutes have elapsed.
- **`results/eval/STOP`**: Creating this file (`touch results/eval/STOP`) requests a clean stop between systems or after the current question finishes. The runner will stop cleanly without deleting the STOP file; remove it (`rm results/eval/STOP`) to resume.
- **`--only-incomplete`** (default when `--only` is omitted): Automatically skips completed questions and executes the lowest-numbered incomplete question next.
- **SIGINT / SIGTERM (`Ctrl-C`)**: Stops cleanly between steps and discards only the in-flight system without corrupting any completed files.

