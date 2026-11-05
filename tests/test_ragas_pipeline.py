"""Offline Unit Tests for RAGAS Evaluation Pipeline (Task E4 / Task F2).

Runs 100% offline with zero external API calls or network requests.
"""

import os
import sys
import json
import tempfile
import math
import pytest
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_ragas import (
    RagasGovernor,
    GovernedJudgeChatModel,
    DryRunJudgeChatModel,
    trim_contexts_to_budget,
    generate_pooled_reference,
    run_ragas_evaluation,
    write_summary_csv,
    build_judge_llm,
)


class MockClock:
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.sleep_calls = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float):
        self.sleep_calls.append(seconds)
        self.current_time += seconds


def create_mock_eval_results(out_dir: str, qid: str = "Q01"):
    """Create a full set of mock eval results for testing."""
    os.makedirs(out_dir, exist_ok=True)
    topic = "Mechanisms of antibiotic persistence in Pseudomonas aeruginosa biofilms"

    for sys_name in ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]:
        contexts = [] if sys_name == "baseline" else [
            f"Evidence 1 for {sys_name} on biofilms.",
            f"Evidence 2 for {sys_name} discussing persister cell dormant states.",
        ]
        data = {
            "qid": qid,
            "system": sys_name,
            "topic": topic,
            "final_report": f"Report for {sys_name} analyzing persister cell formation and antibiotic tolerance.",
            "contexts": contexts,
            "retry_count": 0 if sys_name in ("baseline", "fixed") else 1,
            "score": 7.5,
            "attempts": [],
        }
        with open(os.path.join(out_dir, f"{qid}_{sys_name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def test_sample_building_and_s1_borrowing():
    """Verify S1 borrows S2 contexts and is labelled borrowed_from_S2."""
    with tempfile.TemporaryDirectory() as eval_dir, tempfile.TemporaryDirectory() as ragas_dir:
        create_mock_eval_results(eval_dir, qid="Q01")

        run_ragas_evaluation(
            eval_results_dir=eval_dir,
            ragas_out_dir=ragas_dir,
            phase="A",
            only_qids=["Q01"],
            dry_run=True,
        )

        s1_file = os.path.join(ragas_dir, "Q01_baseline.json")
        assert os.path.exists(s1_file)
        with open(s1_file) as f:
            s1_data = json.load(f)

        assert s1_data["contexts_source"] == "borrowed_from_S2"
        assert s1_data["scores"]["faithfulness"] is not None

        s2_file = os.path.join(ragas_dir, "Q01_fixed.json")
        with open(s2_file) as f:
            s2_data = json.load(f)
        assert s2_data["contexts_source"] == "system_contexts"


def test_s1_has_no_context_precision_or_recall_in_phase_b():
    """Verify S1 receives null for context precision and recall in Phase B (never 0)."""
    with tempfile.TemporaryDirectory() as eval_dir, tempfile.TemporaryDirectory() as ragas_dir:
        create_mock_eval_results(eval_dir, qid="Q01")

        run_ragas_evaluation(
            eval_results_dir=eval_dir,
            ragas_out_dir=ragas_dir,
            phase="B",
            only_qids=["Q01"],
            dry_run=True,
        )

        s1_file = os.path.join(ragas_dir, "Q01_baseline.json")
        with open(s1_file) as f:
            s1_data = json.load(f)

        assert s1_data["scores"]["context_precision"] is None
        assert s1_data["scores"]["context_recall"] is None


def test_resumption_skips_finished_samples():
    """Verify existing output JSONs are skipped on resume."""
    with tempfile.TemporaryDirectory() as eval_dir, tempfile.TemporaryDirectory() as ragas_dir:
        create_mock_eval_results(eval_dir, qid="Q01")

        # Pre-seed Q01_baseline.json
        preseeded = {
            "qid": "Q01",
            "system": "baseline",
            "scores": {"faithfulness": 0.95, "answer_relevancy": 0.90},
            "contexts_source": "borrowed_from_S2",
            "requests_used": 0,
            "seconds": 0.01,
        }
        with open(os.path.join(ragas_dir, "Q01_baseline.json"), "w") as f:
            json.dump(preseeded, f)

        run_ragas_evaluation(
            eval_results_dir=eval_dir,
            ragas_out_dir=ragas_dir,
            phase="A",
            only_qids=["Q01"],
            dry_run=True,
        )

        # Baseline must retain preseeded score
        with open(os.path.join(ragas_dir, "Q01_baseline.json")) as f:
            loaded = json.load(f)
        assert loaded["scores"]["faithfulness"] == 0.95


def test_nan_to_null_and_error_handling():
    """Verify NaN metric outputs are stored as null with error details, not 0."""
    with tempfile.TemporaryDirectory() as ragas_dir:
        summary_path = os.path.join(ragas_dir, "summary.csv")
        records = [
            {
                "qid": "Q01",
                "system": "baseline",
                "phase": "A",
                "scores": {"faithfulness": None, "answer_relevancy": 0.88},
                "contexts_source": "borrowed_from_S2",
                "requests_used": 1,
                "seconds": 1.0,
                "contexts_trimmed_chars": 0,
                "think_stripped_count": 0,
            }
        ]
        write_summary_csv(records, summary_path)
        assert os.path.exists(summary_path)

        with open(summary_path) as f:
            content = f.read()
        assert "faithfulness" in content
        # Missing value in CSV appears as empty between commas, not 0.0
        lines = content.strip().split("\n")
        header = lines[0].split(",")
        row = lines[1].split(",")
        f_idx = header.index("faithfulness")
        assert row[f_idx] == "" or row[f_idx] is None, f"Expected empty null, got {row[f_idx]}"


def test_think_tag_stripping():
    """Verify <think>...</think> tags are stripped and counted."""
    clock = MockClock()
    with tempfile.TemporaryDirectory() as tmpdir:
        usage_log = os.path.join(tmpdir, "usage.jsonl")
        gov = RagasGovernor(usage_log, max_rpm=10, time_fn=clock.time, sleep_fn=clock.sleep)
        fake_chat = DryRunJudgeChatModel(governor=gov)

        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="Break down each sentence in this text.")
        res = fake_chat.invoke([msg])

        assert "<think>" not in res.content
        assert "</think>" not in res.content
        assert gov.think_stripped_count > 0


def test_request_size_guard_trims_oldest_contexts():
    """Verify context trimming trims oldest contexts first when prompt exceeds token limit."""
    topic = "Test Topic"
    report = "Test Report"
    # Create contexts exceeding 1,000 tokens (approx 3,500 chars)
    contexts = [
        "Newest context passage 1" * 50,  # ~1200 chars
        "Middle context passage 2" * 50,  # ~1200 chars
        "Oldest context passage 3" * 50,  # ~1200 chars
    ]

    safe_contexts, trimmed = trim_contexts_to_budget(topic, report, contexts, max_tokens=500)
    assert trimmed > 0
    # Oldest context should be dropped first
    assert len(safe_contexts) < len(contexts)
    assert safe_contexts[0].startswith("Newest context passage 1")


def test_daily_request_budget_clean_stop():
    """Verify governor cleanly exits when 24h request budget is reached."""
    clock = MockClock()
    with tempfile.TemporaryDirectory() as tmpdir:
        usage_log = os.path.join(tmpdir, "usage.jsonl")
        gov = RagasGovernor(usage_log, max_rpm=10, daily_budget=2, time_fn=clock.time, sleep_fn=clock.sleep)

        # First request succeeds
        gov.acquire_request("fake", "fake_model")
        assert gov.total_requests_session == 1

        # Second request succeeds
        gov.acquire_request("fake", "fake_model")
        assert gov.total_requests_session == 2

        # Third request exceeds budget of 2 -> should raise SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            gov.acquire_request("fake", "fake_model")
        assert exc_info.value.code == 0
