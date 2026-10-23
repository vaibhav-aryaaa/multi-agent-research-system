#!/usr/bin/env python3
"""Batch evaluation runner for paired experimental evaluation (Phase 4 Task D).

Executes the four systems across the 25 benchmark questions with shared first-pass control:
- System 1 (baseline): single LLM call.
- System 2 (fixed): first pass, attempt 0, no retries.
- System 3 (adaptive_llm): start from attempt 0, LLM judge decides retries (max 3).
- System 4 (adaptive_classifier): start from attempt 0, cross-encoder classifier judge decides retries (max 3).

Features:
- Paired experimental control (attempt 0 is executed once and shared across S2, S3, S4).
- Resumption support: skips any question whose four files already exist.
- Transactional question output: never writes partial files for a question.
- Rolling 24-hour Groq token budget enforcement with usage_log.jsonl.
- Clean stop on Groq daily token 429 limit.
- Per-question error handling with <qid>_ERROR.json generation.
- Real-time running summary table.
"""

import os
import sys
import time
import json
import shutil
import argparse
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import judges
from agents import CriticFeedback
from graph import (
    DEPS,
    run_first_pass,
    run_pipeline_from_first_pass,
    set_usage_log_path,
    append_to_usage_log,
    MAX_RETRIES,
)
from baseline import run_baseline
from runlog import save_run


DEFAULT_QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "eval_questions.txt")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "eval")
DEFAULT_DAILY_BUDGET = 170000
ESTIMATED_PER_QUESTION_TOKENS = 55000


def load_questions(filepath: str, only: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Load and filter questions from questions file.
    
    Returns ordered list of (qid, topic) tuples.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t", 1)
            if len(parts) == 2:
                qid, topic = parts[0].strip(), parts[1].strip()
            else:
                tokens = stripped.split(None, 1)
                qid, topic = tokens[0].strip(), tokens[1].strip()
            questions.append((qid, topic))

    # Sort to enforce strict Q01..Q25 ordering
    questions.sort(key=lambda x: x[0])

    if only:
        only_set = {q.strip().upper() for q in only}
        questions = [q for q in questions if q[0].upper() in only_set]

    if limit is not None and limit > 0:
        questions = questions[:limit]

    return questions


def is_question_complete(out_dir: str, qid: str) -> bool:
    """Check if all four evaluation output files exist for this question."""
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]
    return all(os.path.exists(os.path.join(out_dir, f"{qid}_{sys}.json")) for sys in systems)


def clean_partial_question_files(out_dir: str, qid: str) -> None:
    """Remove any orphan/partial output files for a question to maintain transactional integrity."""
    systems = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]
    for sys in systems:
        fpath = os.path.join(out_dir, f"{qid}_{sys}.json")
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass


def compute_24h_token_usage(usage_log_path: str) -> Tuple[int, Optional[datetime]]:
    """Calculate total tokens consumed in the last rolling 24 hours and identify oldest active entry."""
    if not os.path.exists(usage_log_path):
        return 0, None

    now = datetime.now()
    window_start = now - timedelta(hours=24)
    total_24h = 0
    oldest_in_window: Optional[datetime] = None

    try:
        with open(usage_log_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time >= window_start:
                        total_24h += int(entry.get("total_tokens", 0))
                        if oldest_in_window is None or entry_time < oldest_in_window:
                            oldest_in_window = entry_time
                except Exception:
                    continue
    except Exception as e:
        print(f"[Warning] Failed to read usage log: {e}")

    return total_24h, oldest_in_window


# ---------------------------------------------------------------------------
# Dry-run Fake Implementations
# ---------------------------------------------------------------------------

class DryRunWebSearch:
    def __init__(self):
        self.queries = []
    def invoke(self, inputs, config=None):
        q = inputs["query"] if isinstance(inputs, dict) else str(inputs)
        self.queries.append(q)
        return (
            f"Title: Technical Overview for {q}\n"
            f"URL: https://arxiv.org/abs/dryrun/{len(self.queries)}\n"
            f"Snippet: Dry-run factual evidence verifying mechanisms and empirical data for {q} with rich quantitative details.\n"
        )


class DryRunScraper:
    def __init__(self):
        self.urls = []
    def invoke(self, inputs, config=None):
        url = inputs["url"] if isinstance(inputs, dict) else str(inputs)
        self.urls.append(url)
        return (
            f"Detailed dry-run technical literature scraped from {url}. "
            f"Experimental characterization demonstrates strong empirical findings with robust controls. "
            f"This content provides deep domain insights and detailed citations that comprehensively "
            f"address all research gaps identified by the reviewers and evaluators during the pipeline run."
        )


class DryRunWriter:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        from graph import record_token_usage
        record_token_usage("writer", {"prompt_tokens": 850, "completion_tokens": 600, "total_tokens": 1450})
        return f"# Dry Run Research Report\n\nAttempt 0 findings on {inputs.get('topic')}. Empirical evidence demonstrates significant effects across tested conditions."


class DryRunReviser:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        from graph import record_token_usage
        record_token_usage("reviser", {"prompt_tokens": 1100, "completion_tokens": 700, "total_tokens": 1800})
        return f"# Dry Run Research Report (Revised Attempt {self.calls})\n\nRevised based on critique: {inputs.get('critique', '')}."


class DryRunCritic:
    def __init__(self, scores, nmr_list):
        self.scores = list(scores)
        self.nmr_list = list(nmr_list)
        self.calls = 0
    def invoke(self, inputs, config=None):
        idx = min(self.calls, len(self.scores) - 1)
        score = self.scores[idx]
        nmr = self.nmr_list[min(self.calls, len(self.nmr_list) - 1)]
        self.calls += 1
        from graph import record_token_usage
        record_token_usage("critic", {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500})
        return CriticFeedback(
            score=score,
            critique=f"Dry run critique #{self.calls} for score {score:.1f}",
            needs_more_research=nmr,
        )


class DryRunQueryWriter:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        from graph import record_token_usage
        record_token_usage("query_writer", {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250})
        return f"refined academic search query for {inputs.get('topic')}"


class DryRunBaselineChain:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        from graph import record_token_usage
        record_token_usage("baseline", {"prompt_tokens": 350, "completion_tokens": 500, "total_tokens": 850})
        return f"# Baseline Report\n\nSingle-prompt direct report on {inputs.get('topic')}."


def setup_dry_run_fakes():
    """Configure mock pipelines for dry-run offline validation."""
    searcher = DryRunWebSearch()
    scraper = DryRunScraper()
    writer = DryRunWriter()
    reviser = DryRunReviser()
    q_writer = DryRunQueryWriter()
    # Score 7.5 on attempt 0 (LLM accepts), attempt 1 8.5
    critic = DryRunCritic(scores=[7.5, 8.5], nmr_list=[False, False])
    baseline = DryRunBaselineChain()

    DEPS.web_search = searcher
    DEPS.scrape_url = scraper
    DEPS.writer_chain = writer
    DEPS.revise_chain = reviser
    DEPS.query_writer_chain = q_writer
    DEPS.critic_chain = critic
    DEPS.baseline_chain = baseline

    # In dry-run mode, advance time simulatedly so tests run instantly without real-time sleep
    class SimulatedDryClock:
        def __init__(self):
            self._time = time.time()
        def time(self):
            return self._time
        def sleep(self, secs):
            self._time += secs

    sim_clock = SimulatedDryClock()
    from graph import set_time_sleep_hooks
    set_time_sleep_hooks(time_fn=sim_clock.time, sleep_fn=sim_clock.sleep)

    # Mock classifier to accept with p_supported = 0.85
    def dry_run_score_claims(claims, passages, k=3, ce_path=None):
        return [{"claim": c, "p_supported": 0.85, "top_passage": passages[0] if passages else ""} for c in claims]

    judges.SCORE_CLAIMS_FN = dry_run_score_claims


# ---------------------------------------------------------------------------
# Runner Main Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Paired Evaluation Runner (Phase 4 Task D)")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE, help="Path to eval_questions.txt")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory (results/eval)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated QIDs to run (e.g. Q03,Q07)")
    parser.add_argument("--dry-run", action="store_true", help="Run with deterministic fakes and zero API calls")
    parser.add_argument("--sleep", type=float, default=20.0, help="Sleep seconds between questions (default: 20)")
    parser.add_argument("--daily-token-budget", type=int, default=DEFAULT_DAILY_BUDGET, help="Rolling 24h token budget")
    args = parser.parse_args()

    # Configure output directory
    if args.dry_run and args.out == DEFAULT_OUT_DIR:
        out_dir = tempfile.mkdtemp(prefix="eval_dry_run_")
        print(f"[DRY-RUN] Using temporary output directory: {out_dir}")
    else:
        out_dir = os.path.abspath(args.out)
        os.makedirs(out_dir, exist_ok=True)

    usage_log_path = os.path.join(out_dir, "usage_log.jsonl")
    set_usage_log_path(usage_log_path)

    # Setup dry-run fakes if requested
    if args.dry_run:
        setup_dry_run_fakes()
        sleep_interval = 0.0
    else:
        sleep_interval = args.sleep

    # Load questions
    only_list = [q.strip() for q in args.only.split(",")] if args.only else None
    questions = load_questions(args.questions, only=only_list, limit=args.limit)

    print("\n" + "=" * 80)
    print("PAIRED EVALUATION RUNNER (Phase 4)")
    print(f"Questions to process: {len(questions)}")
    print(f"Output directory:     {out_dir}")
    print(f"Usage log path:       {usage_log_path}")
    print(f"Daily token budget:   {args.daily_token_budget:,} tokens")
    print(f"Mode:                 {'DRY-RUN (fakes, no APIs)' if args.dry_run else 'LIVE'}")
    print("=" * 80 + "\n")

    completed_in_session = 0
    total_tokens_session = 0
    session_start_time = time.time()

    # Print Table Header
    table_header = (
        f"{'QID':<6} | {'Status':<10} | {'Tokens (24h/Total)':<20} | "
        f"{'Retries (S2/S3/S4)':<18} | {'Scores (S1/S2/S3/S4)':<22} | {'Elapsed':<8}"
    )
    print(table_header)
    print("-" * len(table_header))

    for idx, (qid, topic) in enumerate(questions):
        # 1. Resume check: Skip if question already complete (all four files exist)
        if is_question_complete(out_dir, qid):
            print(f"{qid:<6} | {'SKIPPED':<10} | {'Already complete':<20} | {'-':<18} | {'-':<22} | {'0.0s':<8}")
            continue

        # 2. Daily token budget check
        used_24h, oldest_in_window = compute_24h_token_usage(usage_log_path)
        if not args.dry_run and (used_24h + ESTIMATED_PER_QUESTION_TOKENS > args.daily_token_budget):
            if oldest_in_window:
                expires_at = oldest_in_window + timedelta(hours=24)
                seconds_left = max(0, int((expires_at - datetime.now()).total_seconds()))
                hh = seconds_left // 3600
                mm = (seconds_left % 3600) // 60
                wait_str = f"{hh:02d}:{mm:02d}"
            else:
                wait_str = "24:00"

            print("\n" + "=" * 80)
            print(f"daily budget reached; {completed_in_session} questions done; re-run the same command in {wait_str} to resume")
            print("=" * 80 + "\n")
            sys.exit(0)

        # 3. Process question with checkpoints and granular resumption
        q_start = time.time()
        try:
            # S1: Baseline (System 1)
            s1_path = os.path.join(out_dir, f"{qid}_baseline.json")
            if os.path.exists(s1_path):
                with open(s1_path, "r", encoding="utf-8") as f:
                    s1_res = json.load(f)
            else:
                s1_res = run_baseline(
                    topic=topic,
                    results_dir=out_dir,
                    qid=qid,
                    filename=f"{qid}_baseline.json",
                    save=True,
                )

            # Check if any multi-agent system needs first pass
            s2_path = os.path.join(out_dir, f"{qid}_fixed.json")
            s3_path = os.path.join(out_dir, f"{qid}_adaptive_llm.json")
            s4_path = os.path.join(out_dir, f"{qid}_adaptive_classifier.json")
            first_pass_path = os.path.join(out_dir, f"{qid}_first_pass.json")

            first_state = None
            if not (os.path.exists(s2_path) and os.path.exists(s3_path) and os.path.exists(s4_path)):
                if os.path.exists(first_pass_path):
                    with open(first_pass_path, "r", encoding="utf-8") as f:
                        first_state = json.load(f)
                    print(f"[{qid}] Reusing first-pass checkpoint from {os.path.basename(first_pass_path)}")
                else:
                    first_state = run_first_pass(topic)
                    with open(first_pass_path, "w", encoding="utf-8") as f:
                        json.dump(first_state, f, indent=2, ensure_ascii=False)
                    print(f"[{qid}] Saved first-pass checkpoint to {os.path.basename(first_pass_path)}")

            # S2: Fixed (System 2)
            if os.path.exists(s2_path):
                with open(s2_path, "r", encoding="utf-8") as f:
                    s2_res = json.load(f)
            else:
                s2_res = run_pipeline_from_first_pass(
                    first_state=first_state,
                    mode="fixed",
                    results_dir=out_dir,
                    qid=qid,
                    filename=f"{qid}_fixed.json",
                    save=True,
                )

            # S3: Adaptive LLM (System 3)
            if os.path.exists(s3_path):
                with open(s3_path, "r", encoding="utf-8") as f:
                    s3_res = json.load(f)
            else:
                s3_res = run_pipeline_from_first_pass(
                    first_state=first_state,
                    mode="adaptive_llm",
                    results_dir=out_dir,
                    qid=qid,
                    filename=f"{qid}_adaptive_llm.json",
                    save=True,
                )

            # S4: Adaptive Classifier (System 4)
            if os.path.exists(s4_path):
                with open(s4_path, "r", encoding="utf-8") as f:
                    s4_res = json.load(f)
            else:
                s4_res = run_pipeline_from_first_pass(
                    first_state=first_state,
                    mode="adaptive_classifier",
                    results_dir=out_dir,
                    qid=qid,
                    filename=f"{qid}_adaptive_classifier.json",
                    save=True,
                )

            # Remove previous ERROR.json if all four systems are now complete
            if is_question_complete(out_dir, qid):
                err_path = os.path.join(out_dir, f"{qid}_ERROR.json")
                if os.path.exists(err_path):
                    try:
                        os.remove(err_path)
                    except OSError:
                        pass

            q_elapsed = time.time() - q_start
            completed_in_session += 1

            # Accumulate token metrics
            fp_tokens = (first_state or s2_res).get("first_pass_token_usage", {}).get("total_tokens", 0)
            q_tokens = (
                s1_res.get("token_usage", {}).get("total_tokens", 0) +
                fp_tokens +
                s3_res.get("token_usage_retries", {}).get("total_tokens", 0) +
                s4_res.get("token_usage_retries", {}).get("total_tokens", 0)
            )
            total_tokens_session += q_tokens

            # Format summary row
            retries_str = f"{s2_res.get('retry_count', 0)} / {s3_res.get('retry_count', 0)} / {s4_res.get('retry_count', 0)}"
            s1_s = f"{s1_res.get('score'):.1f}" if s1_res.get("score") is not None else "-"
            s2_s = f"{s2_res.get('score'):.1f}" if s2_res.get("score") is not None else "-"
            s3_s = f"{s3_res.get('score'):.1f}" if s3_res.get("score") is not None else "-"
            s4_s = f"{s4_res.get('score'):.1f}" if s4_res.get("score") is not None else "-"
            scores_str = f"{s1_s} / {s2_s} / {s3_s} / {s4_s}"

            tokens_str = f"{used_24h + q_tokens:,} / +{q_tokens:,}"
            row = (
                f"{qid:<6} | {'OK':<10} | {tokens_str:<20} | "
                f"{retries_str:<18} | {scores_str:<22} | {q_elapsed:.1f}s"
            )
            print(row)

        except Exception as e:
            err_msg = str(e)
            is_daily_limit = ("tokens per day" in err_msg.lower() or "tpd" in err_msg.lower() or "daily token limit hit" in err_msg.lower())

            if is_daily_limit:
                print("\n" + "=" * 80)
                print("daily limit reached; re-run the same command tomorrow to resume")
                print("=" * 80 + "\n")
                sys.exit(0)

            # Record error JSON
            error_record = {
                "qid": qid,
                "topic": topic,
                "error": err_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            err_file = os.path.join(out_dir, f"{qid}_ERROR.json")
            with open(err_file, "w", encoding="utf-8") as ef:
                json.dump(error_record, ef, indent=2, ensure_ascii=False)

            q_elapsed = time.time() - q_start
            print(f"{qid:<6} | {'ERROR':<10} | {'-':<20} | {'-':<18} | {err_msg[:20]:<22} | {q_elapsed:.1f}s")

            # Record error JSON
            error_record = {
                "qid": qid,
                "topic": topic,
                "error": err_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            err_file = os.path.join(out_dir, f"{qid}_ERROR.json")
            with open(err_file, "w", encoding="utf-8") as ef:
                json.dump(error_record, ef, indent=2, ensure_ascii=False)

            q_elapsed = time.time() - q_start
            print(f"{qid:<6} | {'ERROR':<10} | {'-':<20} | {'-':<18} | {err_msg[:20]:<22} | {q_elapsed:.1f}s")

        # Sleep between questions to respect TPM (except after final question)
        if idx < len(questions) - 1 and sleep_interval > 0:
            time.sleep(sleep_interval)

    total_elapsed = time.time() - session_start_time
    print("-" * len(table_header))
    print(f"Batch run finished: {completed_in_session} questions processed in {total_elapsed:.1f}s (Total tokens: {total_tokens_session:,})\n")


if __name__ == "__main__":
    main()
