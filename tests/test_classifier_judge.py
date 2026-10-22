"""Unit tests for classifier_judge and adaptive_classifier mode (Phase 3 Task C).

Tests operate 100% offline with stubbed scorers:
- 12 claims: 1 supported (0.083) -> reject; 2 supported (0.167) -> accept; 0 claims -> accept.
- tau, backend, model_source read from config file.
- classifier unavailable -> raises without fallback.
- run_pipeline(mode='adaptive_classifier') flow with fake chains: retry on reject, judge_details persisted.
- LLM critic score 9.0 + classifier reject -> retry triggered (classifier is the sole decider).
"""

import os
import sys
import json
import tempfile
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("GROQ_MODEL", "test-pinned-model")
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_test")

from agents import CriticFeedback
import graph
from graph import DEPS, run_pipeline
import judges
from judges import classifier_judge, JudgeVerdict
from config import FAITHFULNESS_THRESHOLD

# 12 distinct factual sentences for testing
TWELVE_CLAIMS_REPORT = (
    "# Experimental Results\n\n"
    "1. The Casimir force between conducting plates is attractive at separation distances above ten nanometers.\n"
    "2. High-fidelity Cas9 variants decrease off-target cleavage by mutating contact residues.\n"
    "3. Roman pozzolanic concrete contains reactive lime clasts that enable autonomous crack healing.\n"
    "4. Scramjet intake efficiency degrades rapidly at angles of attack exceeding six degrees.\n"
    "5. Marine microplastics accumulate across trophic levels in pelagic food webs.\n"
    "6. Synaptic pruning during adolescence is mediated by microglial engulfment pathways.\n"
    "7. Nitrogen-vacancy centers in diamond provide high-sensitivity magnetic field detection.\n"
    "8. Sulfide-based solid electrolytes demonstrate chemical degradation when contacted with lithium metal.\n"
    "9. Perovskite solar cells undergo rapid degradation under high relative humidity conditions.\n"
    "10. CRISPR gene drive containment strategies include split drives and synthetic reversal targets.\n"
    "11. Hydrothermal vent microbial communities utilize sulfur oxidation pathways for chemolithoautotrophy.\n"
    "12. Methane emissions from thermokarst lakes increase as permafrost thaw accelerates across the Arctic.\n"
)


def test_classifier_judge_threshold_decisions():
    """Verify threshold decisions: 1/12 (0.083) -> reject; 2/12 (0.167) -> accept; 0 claims -> accept."""
    state_base = {
        "report": TWELVE_CLAIMS_REPORT,
        "search_results": "Title: Test\nURL: https://example.com\nSnippet: Background evidence\n",
        "scraped_content": "Scraped evidence content.",
    }

    # Case 1: 1 supported claim out of 12 (fraction = 0.0833 < 0.10) -> reject
    def stub_one_supported(claims, passages=None, **kwargs):
        # First claim p_supported = 0.95 (>= 0.66), remaining 11 claims p_supported = 0.10 (< 0.66)
        return [{"claim": c, "p_supported": 0.95 if i == 0 else 0.10} for i, c in enumerate(claims)]

    verdict_reject = classifier_judge({**state_base, "classifier_stub": stub_one_supported})
    assert verdict_reject["accept"] is False
    assert verdict_reject["score"] == pytest.approx(1 / 12, abs=1e-3)
    assert verdict_reject["details"]["n_claims"] == 12
    assert verdict_reject["details"]["n_supported"] == 1
    assert "below threshold" in verdict_reject["reason"]

    # Case 2: 2 supported claims out of 12 (fraction = 0.1667 >= 0.10) -> accept
    def stub_two_supported(claims, passages=None, **kwargs):
        return [{"claim": c, "p_supported": 0.95 if i < 2 else 0.10} for i, c in enumerate(claims)]

    verdict_accept = classifier_judge({**state_base, "classifier_stub": stub_two_supported})
    assert verdict_accept["accept"] is True
    assert verdict_accept["score"] == pytest.approx(2 / 12, abs=1e-3)
    assert verdict_accept["details"]["n_claims"] == 12
    assert verdict_accept["details"]["n_supported"] == 2
    assert "meets threshold" in verdict_accept["reason"]

    # Case 3: Zero claims in report -> accept with reason 'no checkable claims'
    verdict_zero = classifier_judge({"report": "Overall, this report investigates perpetual energy."})
    assert verdict_zero["accept"] is True
    assert verdict_zero["details"]["n_claims"] == 0
    assert "no checkable claims" in verdict_zero["reason"].lower()


def test_config_file_reading():
    """Verify tau, backend, and model_source are dynamically read from configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_file = os.path.join(tmpdir, "custom_classifier_config.json")
        custom_cfg = {
            "backend": "cross_encoder",
            "tau": 0.75,
            "model_source": "test-custom-distilbert",
            "passes_gate": True,
        }
        with open(cfg_file, "w") as f:
            json.dump(custom_cfg, f)

        orig_env = os.environ.get("CLASSIFIER_CONFIG_PATH")
        try:
            os.environ["CLASSIFIER_CONFIG_PATH"] = cfg_file

            def stub_scorer(claims, passages=None, **kwargs):
                return [{"claim": c, "p_supported": 0.80} for c in claims]

            state = {
                "report": TWELVE_CLAIMS_REPORT,
                "classifier_stub": stub_scorer,
            }
            verdict = classifier_judge(state)
            assert verdict["details"]["tau"] == 0.75
            assert verdict["details"]["backend"] == "cross_encoder"
            assert verdict["details"]["model_source"] == "test-custom-distilbert"
        finally:
            if orig_env is None:
                os.environ.pop("CLASSIFIER_CONFIG_PATH", None)
            else:
                os.environ["CLASSIFIER_CONFIG_PATH"] = orig_env


def test_classifier_unavailable_raises():
    """Verify that when classifier is unavailable, an error is raised with NO fallback to LLM judge."""
    orig_env = os.environ.get("CLASSIFIER_CONFIG_PATH")
    try:
        os.environ["CLASSIFIER_CONFIG_PATH"] = "/nonexistent/path/classifier_config.json"
        with pytest.raises(RuntimeError, match="Classifier configuration file not found"):
            classifier_judge({"report": TWELVE_CLAIMS_REPORT})
    finally:
        if orig_env is None:
            os.environ.pop("CLASSIFIER_CONFIG_PATH", None)
        else:
            os.environ["CLASSIFIER_CONFIG_PATH"] = orig_env


from tests.test_retry_flow import setup_fakes

def test_pipeline_adaptive_classifier_flow():
    """Verify run_pipeline(mode='adaptive_classifier') triggers retry on reject and persists judge_details."""
    setup_fakes(critic_scores=[8.0, 8.0], needs_more_research_list=[False, False])
    DEPS.writer_chain.invoke = lambda inputs, config=None: TWELVE_CLAIMS_REPORT
    DEPS.revise_chain.invoke = lambda inputs, config=None: TWELVE_CLAIMS_REPORT + "\n13. Revised report finding confirms hypothesis."

    # Scorer that rejects attempt 0 (all 0.0) and accepts attempt 1 (all 0.9)
    attempt_counter = [0]
    def alternating_stub(claims, passages=None, **kwargs):
        call_num = attempt_counter[0]
        attempt_counter[0] += 1
        if call_num == 0:
            # Reject attempt 0 (0% supported)
            return [{"claim": c, "p_supported": 0.05} for c in claims]
        else:
            # Accept attempt 1 (100% supported)
            return [{"claim": c, "p_supported": 0.95} for c in claims]

    orig_scorer = judges.SCORE_CLAIMS_FN
    try:
        judges.SCORE_CLAIMS_FN = alternating_stub

        with tempfile.TemporaryDirectory() as tmpdir:
            final_state = run_pipeline(
                topic="Synthetic Biology Containment",
                mode="adaptive_classifier",
                results_dir=tmpdir,
            )

            # Exactly one retry occurred
            assert final_state["retry_count"] == 1
            assert len(final_state["attempts"]) == 2

            attempt0 = final_state["attempts"][0]
            attempt1 = final_state["attempts"][1]

            assert attempt0["judge_accept"] is False
            assert attempt0["decision"] == "writer"
            assert "judge_details" in attempt0
            assert attempt0["judge_details"]["n_claims"] == 12
            assert attempt0["judge_details"]["n_supported"] == 0

            assert attempt1["judge_accept"] is True
            assert attempt1["decision"] == "end"
            assert "judge_details" in attempt1
            assert attempt1["judge_details"]["n_supported"] > 0

            # Verify saved run JSON contains judge_details in attempts
            json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            assert len(json_files) == 1
            with open(os.path.join(tmpdir, json_files[0])) as jf:
                saved_run = json.load(jf)

            assert saved_run["system"] == "adaptive_classifier"
            assert len(saved_run["attempts"]) == 2
            assert saved_run["attempts"][0]["judge_details"]["n_claims"] == 12
            assert saved_run["attempts"][1]["judge_details"]["n_supported"] > 0
    finally:
        judges.SCORE_CLAIMS_FN = orig_scorer


def test_llm_critic_gives_high_score_but_classifier_rejects():
    """Verify that even when LLM critic gives 9.0, classifier rejection forces a retry."""
    # LLM critic gives 9.0 (well above QUALITY_THRESHOLD=7.0)
    setup_fakes(critic_scores=[9.0, 9.0], needs_more_research_list=[False, False])
    DEPS.writer_chain.invoke = lambda inputs, config=None: TWELVE_CLAIMS_REPORT
    DEPS.revise_chain.invoke = lambda inputs, config=None: TWELVE_CLAIMS_REPORT + "\n13. Revised report finding."

    call_count = [0]
    def reject_then_accept(claims, passages=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First attempt rejected by classifier
            return [{"claim": c, "p_supported": 0.0} for c in claims]
        return [{"claim": c, "p_supported": 0.95} for c in claims]

    orig_scorer = judges.SCORE_CLAIMS_FN
    try:
        judges.SCORE_CLAIMS_FN = reject_then_accept

        with tempfile.TemporaryDirectory() as tmpdir:
            final_state = run_pipeline(
                topic="Casimir Energy Extraction",
                mode="adaptive_classifier",
                results_dir=tmpdir,
            )

            # Proves classifier is the decider: critic gave 9.0, but retry was forced
            assert final_state["attempts"][0]["critic_score"] == 9.0
            assert final_state["attempts"][0]["judge_accept"] is False
            assert final_state["attempts"][0]["decision"] == "writer"
            assert final_state["retry_count"] == 1
    finally:
        judges.SCORE_CLAIMS_FN = orig_scorer
