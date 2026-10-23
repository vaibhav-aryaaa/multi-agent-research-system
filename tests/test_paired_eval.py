import os
import sys
import copy
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_MODEL", "test-pinned-model")
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_test")

import judges
from agents import CriticFeedback
from graph import (
    DEPS,
    run_first_pass,
    run_pipeline_from_first_pass,
    MAX_RETRIES,
)


class FakeWebSearch:
    def __init__(self):
        self.queries = []
    def invoke(self, inputs, config=None):
        query = inputs["query"] if isinstance(inputs, dict) else str(inputs)
        self.queries.append(query)
        return (
            "Title: Technical Study on Biofilms\n"
            "URL: https://example.org/study-biofilms\n"
            "Snippet: Antibiotic persistence in biofilms is mediated by metabolic dormancy and efflux pumps.\n"
        )


class FakeScraper:
    def __init__(self):
        self.scraped_urls = []
    def invoke(self, inputs, config=None):
        url = inputs["url"] if isinstance(inputs, dict) else str(inputs)
        self.scraped_urls.append(url)
        return (
            "Detailed technical evidence on antibiotic persistence mechanisms. "
            "Persister cells survive high-dose antibiotic treatments through metabolic shutdown. "
            "Efflux pump regulation prevents intracellular accumulation of aminoglycosides."
        )


class FakeWriter:
    def __init__(self, report_text="REPORT attempt-0: Biofilms exhibit high antibiotic persistence."):
        self.report_text = report_text
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        return self.report_text


class FakeReviser:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        critique = inputs.get("critique", "")
        return f"REPORT attempt-{self.calls} (revised with: {critique})"


class FakeQueryWriter:
    def __init__(self):
        self.calls = 0
    def invoke(self, inputs, config=None):
        self.calls += 1
        return "refined search query for persistence mechanisms"


class FakeCritic:
    def __init__(self, scores, needs_more_research_list):
        self.scores = list(scores)
        self.needs_more_research_list = list(needs_more_research_list)
        self.calls = 0
    def invoke(self, inputs, config=None):
        idx = min(self.calls, len(self.scores) - 1)
        score = self.scores[idx]
        nmr = self.needs_more_research_list[min(self.calls, len(self.needs_more_research_list) - 1)]
        self.calls += 1
        return CriticFeedback(
            score=score,
            critique=f"Critique #{self.calls} (score {score})",
            needs_more_research=nmr,
        )


def setup_fakes(scores, needs_more_research_list, report_text="REPORT attempt-0: Biofilms exhibit high antibiotic persistence."):
    searcher = FakeWebSearch()
    scraper = FakeScraper()
    writer = FakeWriter(report_text=report_text)
    reviser = FakeReviser()
    q_writer = FakeQueryWriter()
    critic = FakeCritic(scores, needs_more_research_list)

    DEPS.web_search = searcher
    DEPS.scrape_url = scraper
    DEPS.writer_chain = writer
    DEPS.revise_chain = reviser
    DEPS.query_writer_chain = q_writer
    DEPS.critic_chain = critic

    return searcher, scraper, writer, reviser, q_writer, critic


def test_paired_eval_both_accept():
    """Requirement: When both judges accept, S2 == S3 == S4 reports, retry_count is 0,
    and critic LLM is called exactly once for attempt 0."""
    scores = [8.5]
    nmr = [False]
    searcher, scraper, writer, reviser, q_writer, critic = setup_fakes(scores, nmr)

    # Mock classifier to accept (p_supported = 0.95)
    def mock_score_claims_accept(claims, passages, k=3, ce_path=None):
        return [{"claim": c, "p_supported": 0.95, "top_passage": passages[0] if passages else ""} for c in claims]

    judges.SCORE_CLAIMS_FN = mock_score_claims_accept

    try:
        topic = "Mechanisms of antibiotic persistence in Pseudomonas aeruginosa biofilms"
        # 1. First pass executed ONCE
        first_state = run_first_pass(topic)

        # Critic fake must be called exactly once during attempt 0
        assert critic.calls == 1, f"Expected critic to be called once for attempt 0, got {critic.calls}"
        attempt_0_report = first_state["report"]
        assert "attempt-0" in attempt_0_report

        # 2. S2 Fixed
        s2 = run_pipeline_from_first_pass(first_state, mode="fixed", save=False)
        assert s2["report"] == attempt_0_report, "S2 report must equal attempt-0 report"
        assert s2["retry_count"] == 0, "S2 fixed must have retry_count 0"
        assert critic.calls == 1, "S2 fixed must make no extra critic calls"

        # 3. S3 Adaptive LLM (score 8.5 >= 7.0 -> accepts immediately)
        s3 = run_pipeline_from_first_pass(first_state, mode="adaptive_llm", save=False)
        assert s3["retry_count"] == 0, "S3 must accept with retry_count 0"
        assert s3["report"] == attempt_0_report, "S3 report must equal attempt-0 report"
        assert critic.calls == 1, "S3 must not call critic again on immediate accept"

        # 4. S4 Adaptive Classifier (fraction 1.0 >= 0.10 -> accepts immediately)
        s4 = run_pipeline_from_first_pass(first_state, mode="adaptive_classifier", save=False)
        assert s4["retry_count"] == 0, "S4 must accept with retry_count 0"
        assert s4["report"] == attempt_0_report, "S4 report must equal attempt-0 report"
        assert critic.calls == 1, "S4 must not call critic again on immediate accept"

        # When both accept: S2 == S3 == S4
        assert s2["report"] == s3["report"] == s4["report"]
    finally:
        judges.SCORE_CLAIMS_FN = None


def test_paired_eval_llm_accepts_classifier_rejects():
    """Requirement: When LLM judge accepts and classifier rejects:
    - S2 report == attempt-0 report
    - S3 has retry_count 0 (report == attempt-0 report)
    - S4 has retry_count >= 1 (report is revised)
    - Critic fake is called exactly once for attempt 0, and called during S4 retries only."""
    # Attempt 0: score 8.0 (LLM accepts)
    # Attempt 1: score 8.5
    scores = [8.0, 8.5, 9.0]
    nmr = [False, False, False]
    searcher, scraper, writer, reviser, q_writer, critic = setup_fakes(scores, nmr)

    # Classifier rejects on attempt 0 (p_supported = 0.0), then accepts on attempt 1 (p_supported = 0.9)
    call_count = {"count": 0}
    def mock_score_claims_dynamic(claims, passages, k=3, ce_path=None):
        call_count["count"] += 1
        if call_count["count"] == 1:
            # Attempt 0 rejection
            return [{"claim": c, "p_supported": 0.0, "top_passage": ""} for c in claims]
        else:
            # Attempt 1 acceptance
            return [{"claim": c, "p_supported": 0.9, "top_passage": passages[0] if passages else ""} for c in claims]

    judges.SCORE_CLAIMS_FN = mock_score_claims_dynamic

    try:
        topic = "Mechanisms of antibiotic persistence in Pseudomonas aeruginosa biofilms"
        # 1. First pass executed ONCE
        first_state = run_first_pass(topic)
        assert critic.calls == 1, f"Critic called {critic.calls} times, expected 1 for attempt 0"
        attempt_0_report = first_state["report"]

        # 2. S2 Fixed
        s2 = run_pipeline_from_first_pass(first_state, mode="fixed", save=False)
        assert s2["report"] == attempt_0_report
        assert s2["retry_count"] == 0
        assert critic.calls == 1

        # 3. S3 Adaptive LLM (score 8.0 >= 7.0 -> accepts immediately)
        s3 = run_pipeline_from_first_pass(first_state, mode="adaptive_llm", save=False)
        assert s3["retry_count"] == 0
        assert s3["report"] == attempt_0_report
        assert critic.calls == 1

        # 4. S4 Adaptive Classifier (attempt 0 rejected -> retries)
        s4 = run_pipeline_from_first_pass(first_state, mode="adaptive_classifier", save=False)
        assert s4["retry_count"] >= 1, f"S4 must have retry_count >= 1, got {s4['retry_count']}"
        assert s4["report"] != attempt_0_report, "S4 report must be revised"
        assert "revised" in s4["report"]
        assert critic.calls > 1, f"Critic should have been called on S4 retry, got {critic.calls}"

        # Token usage separation checks
        assert "token_usage_first_pass" in s4
        assert "token_usage_retries" in s4
        assert s3["token_usage_retries"]["total_tokens"] == 0
    finally:
        judges.SCORE_CLAIMS_FN = None


def test_baseline_execution_and_token_tracking():
    """Verify S1 baseline runs with fake chain and stores required schema fields."""
    import tempfile
    from baseline import run_baseline

    class FakeBaselineChain:
        def __init__(self):
            self.calls = 0
        def invoke(self, inputs, config=None):
            self.calls += 1
            from graph import record_token_usage
            record_token_usage("baseline", {"prompt_tokens": 150, "completion_tokens": 300, "total_tokens": 450})
            return "BASELINE REPORT: Direct LLM generation."

    fake_b = FakeBaselineChain()
    DEPS.baseline_chain = fake_b

    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_baseline(
            topic="Test Topic for Baseline",
            results_dir=tmpdir,
            qid="Q99",
            filename="Q99_baseline.json",
            save=True,
        )
        assert res["qid"] == "Q99"
        assert res["retry_count"] == 0
        assert "BASELINE REPORT" in res["report"]
        assert res["token_usage"]["total_tokens"] > 0
        assert res["token_usage_first_pass"]["total_tokens"] > 0
        assert res["token_usage_retries"]["total_tokens"] == 0

        saved_path = os.path.join(tmpdir, "Q99_baseline.json")
        assert os.path.exists(saved_path)
        import json
        with open(saved_path) as f:
            data = json.load(f)
        assert data["qid"] == "Q99"
        assert data["system"] == "baseline"
        assert data["contexts"] == []
        assert "token_usage_first_pass" in data


def test_run_eval_dry_run_runner_cli():
    """Test scripts/run_eval.py in dry-run mode generating all 4 system files per question."""
    import subprocess
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "run_eval.py"),
            "--dry-run",
            "--out", tmpdir,
            "--only", "Q01,Q02",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"run_eval failed with: {proc.stderr}"
        assert "PAIRED EVALUATION RUNNER" in proc.stdout

        # Verify all 4 files for Q01 and Q02 exist
        for qid in ["Q01", "Q02"]:
            for sys_name in ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]:
                fpath = os.path.join(tmpdir, f"{qid}_{sys_name}.json")
                assert os.path.exists(fpath), f"Missing output file: {fpath}"
                with open(fpath) as f:
                    data = json.load(f)
                assert data["qid"] == qid
                assert data["system"] == sys_name
                assert "final_report" in data
                assert "token_usage_first_pass" in data
                assert "token_usage_retries" in data
                if sys_name != "baseline":
                    assert len(data["contexts"]) > 0

        # Verify usage_log.jsonl was written
        usage_log = os.path.join(tmpdir, "usage_log.jsonl")
        assert os.path.exists(usage_log)

        # Test resume: running again on Q01,Q02 skips both
        proc_resume = subprocess.run(cmd, capture_output=True, text=True)
        assert proc_resume.returncode == 0
        assert "SKIPPED" in proc_resume.stdout

