import os
import sys
import json
import time
import tempfile
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph
from graph import (
    DEPS,
    invoke_with_retry,
    set_time_sleep_hooks,
    reset_rolling_window,
    record_rolling_usage,
    get_rolling_usage,
    estimate_tokens,
    govern_rate_limit,
    run_first_pass,
    run_pipeline_from_first_pass,
    GOVERNOR_TPM_LIMIT,
)


class MockClock:
    """Deterministic simulated clock for testing the rolling governor."""
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.sleep_calls = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float):
        self.sleep_calls.append(seconds)
        self.current_time += seconds


def test_rolling_governor_under_7000_limit():
    """Verify governor tracks rolling 60s usage and sleeps until used + estimate <= 7,000."""
    clock = MockClock(start_time=1000.0)
    set_time_sleep_hooks(time_fn=clock.time, sleep_fn=clock.sleep)
    reset_rolling_window()

    try:
        # Call 1 at t=1000.0 with 3,500 tokens
        record_rolling_usage(3500)
        assert get_rolling_usage() == 3500

        # Call 2 at t=1010.0 with 2,500 tokens
        clock.current_time = 1010.0
        record_rolling_usage(2500)
        assert get_rolling_usage() == 6000

        # Now at t=1020.0, Call 3 arrives with estimate = 2,000 tokens
        # 6,000 + 2,000 = 8,000 > 7,000 limit!
        clock.current_time = 1020.0

        # Call governor: should sleep until Call 1 (at t=1000.0) drops out at t=1060.0
        # Expected sleep: 1060.1 - 1020.0 = 40.1s
        govern_rate_limit(est_total=2000)

        assert len(clock.sleep_calls) >= 1
        total_slept = sum(clock.sleep_calls)
        assert total_slept >= 40.0, f"Expected sleep >= 40.0s, got {total_slept}s"

        # After sleep, Call 1 has dropped out; remaining usage is Call 2 (2,500)
        # 2,500 + 2,000 = 4,500 <= 7,000, so it fits!
        assert get_rolling_usage() <= 5000
    finally:
        set_time_sleep_hooks(None, None)
        reset_rolling_window()


def test_governor_rejects_single_request_over_7000():
    """Verify single request with estimate > 7,000 tokens fails immediately with clear error."""
    clock = MockClock(start_time=1000.0)
    set_time_sleep_hooks(time_fn=clock.time, sleep_fn=clock.sleep)
    reset_rolling_window()

    try:
        # 30,000 chars / 3.5 = ~8,571 prompt tokens > 7,000
        large_input = "x" * 28000
        with pytest.raises(RuntimeError) as exc_info:
            invoke_with_retry(lambda x: "ok", large_input)
        assert "exceeds the 7000 per-minute limit" in str(exc_info.value).lower()
        # Ensure it didn't sleep in a retry loop
        assert len(clock.sleep_calls) == 0
    finally:
        set_time_sleep_hooks(None, None)
        reset_rolling_window()


def test_700_word_report_and_evidence_fits_under_7000_tokens():
    """Verify that a 700-word report plus 10,000 characters of evidence fits under 7,000 estimated tokens
    for both the critic call and the revise call (Task D7).
    """
    # 700-word report: realistic academic text (~7 chars per word)
    sample_sentence = "The research study examines novel materials and mechanisms for energy storage applications in engineering systems ."
    words = (sample_sentence.split() * 50)[:700]
    report = " ".join(words)
    assert len(report.split()) == 700

    evidence = ("evidence text " * 1000)[:10000]
    assert len(evidence) == 10000

    # 1. Critic Call
    critic_inputs = {
        "report": report,
        "research": evidence,
    }
    critic_tokens = estimate_tokens(critic_inputs, expected_completion=1000)
    assert critic_tokens < GOVERNOR_TPM_LIMIT, f"Critic call estimated {critic_tokens} >= {GOVERNOR_TPM_LIMIT}"

    # 2. Revise Call (critique capped at 1,500 chars, completion estimated from previous report length)
    critique = "Critique feedback text. " * 100
    critique_capped = critique[:1500]
    assert len(critique_capped) == 1500

    revise_inputs = {
        "topic": "Hydrogen embrittlement mechanisms in high-strength steels used for gas pipelines",
        "research": evidence,
        "previous_report": report,
        "critique": critique_capped,
    }
    revise_completion = max(100, int(len(report) / 3.5))
    revise_tokens = estimate_tokens(revise_inputs, expected_completion=revise_completion)
    assert revise_tokens < GOVERNOR_TPM_LIMIT, f"Revise call estimated {revise_tokens} >= {GOVERNOR_TPM_LIMIT}"


def test_parse_429_hint_and_sleep_plus_two_seconds():
    """Verify 429 rate limit parses 'try again in Xs' hint, adds 2.0s, and allows up to 6 attempts."""
    clock = MockClock(start_time=2000.0)
    set_time_sleep_hooks(time_fn=clock.time, sleep_fn=clock.sleep)
    reset_rolling_window()

    class FlakyRunnable:
        def __init__(self):
            self.attempts = 0
        def invoke(self, inputs, config=None):
            self.attempts += 1
            if self.attempts < 4:
                # Groq rate limit message
                raise RuntimeError(
                    "Error code: 429 - {'error': {'message': 'Rate limit reached ... Please try again in 3.5s.'}}"
                )
            return "SUCCESS AFTER RATE LIMIT"

    flaky = FlakyRunnable()
    try:
        res = invoke_with_retry(flaky, {"text": "small prompt"}, max_attempts=6, expected_completion=50)
        assert res == "SUCCESS AFTER RATE LIMIT"
        assert flaky.attempts == 4

        # Verify sleeps: each 429 sleep must be hint (3.5s) + 2.0s = 5.5s
        assert len(clock.sleep_calls) == 3
        for slept in clock.sleep_calls:
            assert abs(slept - 5.5) < 0.1, f"Expected sleep ~5.5s, got {slept}s"
    finally:
        set_time_sleep_hooks(None, None)
        reset_rolling_window()


def test_429_daily_limit_stops_immediately():
    """Verify daily token limit error (TPD) stops immediately without retry sleeps."""
    clock = MockClock(start_time=3000.0)
    set_time_sleep_hooks(time_fn=clock.time, sleep_fn=clock.sleep)
    reset_rolling_window()

    class DailyLimitRunnable:
        def invoke(self, inputs, config=None):
            raise RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per day (TPD)'}}")

    try:
        with pytest.raises(RuntimeError) as exc_info:
            invoke_with_retry(DailyLimitRunnable(), "test")
        assert "DAILY TOKEN LIMIT HIT" in str(exc_info.value)
        assert len(clock.sleep_calls) == 0
    finally:
        set_time_sleep_hooks(None, None)
        reset_rolling_window()


def test_resume_with_crashed_s3_reuses_first_pass():
    """Verify that when S3 crashes:
    1. First pass is written to <qid>_first_pass.json and preserved.
    2. S1 and S2 succeed and their result files exist.
    3. On resume, S1, S2, and first pass are NOT rerun; first_pass is reused.
    4. S3 and S4 run to completion, yielding all 4 result files.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a single test question in a custom questions file
        q_file = os.path.join(tmpdir, "test_questions.txt")
        with open(q_file, "w") as f:
            f.write("Q99\tTest Resilience and Checkpoint Recovery\n")

        # 1. Pre-create S1 baseline file (simulating S1 previously completed)
        s1_data = {
            "run_id": "test_s1_run",
            "qid": "Q99",
            "system": "baseline",
            "topic": "Test Resilience and Checkpoint Recovery",
            "final_report": "Pre-existing baseline report.",
            "retry_count": 0,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
            "token_usage_first_pass": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
            "token_usage_retries": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "attempts": [],
            "contexts": [],
        }
        with open(os.path.join(tmpdir, "Q99_baseline.json"), "w") as f:
            json.dump(s1_data, f, indent=2)

        # 2. Pre-create first-pass checkpoint (simulating first pass completed)
        fp_data = {
            "topic": "Test Resilience and Checkpoint Recovery",
            "search_results": "Evidence on resilience.",
            "latest_search_results": "Evidence on resilience.",
            "scraped_content": "Scraped facts on resilience.",
            "report": "Initial attempt-0 report on resilience.",
            "feedback": "Critique: good report.",
            "score": 7.5,
            "attempt": 0,
            "retry_count": 0,
            "decision": "end",
            "judge": "llm",
            "retry_enabled": False,
            "used_urls": ["https://example.org/resilience"],
            "queries_used": ["Test Resilience and Checkpoint Recovery"],
            "attempts": [],
            "first_pass_token_usage": {"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
            "model_name": "test-model",
        }
        with open(os.path.join(tmpdir, "Q99_first_pass.json"), "w") as f:
            json.dump(fp_data, f, indent=2)

        # 3. Pre-create S2 fixed file (simulating S2 previously completed)
        s2_data = {
            "run_id": "test_s2_run",
            "qid": "Q99",
            "system": "fixed",
            "topic": "Test Resilience and Checkpoint Recovery",
            "final_report": "Initial attempt-0 report on resilience.",
            "retry_count": 0,
            "token_usage": {"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
            "token_usage_first_pass": {"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
            "token_usage_retries": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "attempts": [],
            "contexts": ["Evidence on resilience.", "Scraped facts on resilience."],
        }
        with open(os.path.join(tmpdir, "Q99_fixed.json"), "w") as f:
            json.dump(s2_data, f, indent=2)

        # At this point: Q99_baseline.json, Q99_first_pass.json, Q99_fixed.json exist
        # Q99_adaptive_llm.json and Q99_adaptive_classifier.json are missing (simulating crash before S3 finishes)
        assert os.path.exists(os.path.join(tmpdir, "Q99_baseline.json"))
        assert os.path.exists(os.path.join(tmpdir, "Q99_first_pass.json"))
        assert os.path.exists(os.path.join(tmpdir, "Q99_fixed.json"))
        assert not os.path.exists(os.path.join(tmpdir, "Q99_adaptive_llm.json"))
        assert not os.path.exists(os.path.join(tmpdir, "Q99_adaptive_classifier.json"))

        # Run scripts/run_eval.py on this dir with --dry-run
        # It must reuse Q99_first_pass.json, skip S1 and S2, and finish S3 and S4!
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "run_eval.py"),
            "--dry-run",
            "--questions", q_file,
            "--out", tmpdir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"run_eval resume failed: {proc.stderr}"
        assert "Reusing first-pass checkpoint from Q99_first_pass.json" in proc.stdout

        # Verify all four system files now exist
        assert os.path.exists(os.path.join(tmpdir, "Q99_baseline.json"))
        assert os.path.exists(os.path.join(tmpdir, "Q99_fixed.json"))
        assert os.path.exists(os.path.join(tmpdir, "Q99_adaptive_llm.json"))
        assert os.path.exists(os.path.join(tmpdir, "Q99_adaptive_classifier.json"))
