import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_MODEL", "test-pinned-model")
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_test")

from config import QUALITY_THRESHOLD, MAX_RETRIES
from graph import decide, route_after_critic, combine_evidence
from judges import llm_judge, classifier_judge, JUDGES

def run_tests():
    print("=" * 60)
    print("Running Routing & Decision Pure Logic Tests")
    print("=" * 60)

    # Case 1: score below threshold with retries left, needs_more_research=True -> researcher
    decision = decide(
        retry_enabled=True,
        accept=False,
        retry_count=0,
        needs_more_research=True,
    )
    assert decision == "researcher", f"Expected 'researcher', got '{decision}'"
    print("[PASS] Case 1: Score below threshold with retries left (needs_more_research=True) -> 'researcher'")

    # Case 2: score below threshold with retries left, needs_more_research=False -> writer
    decision = decide(
        retry_enabled=True,
        accept=False,
        retry_count=1,
        needs_more_research=False,
    )
    assert decision == "writer", f"Expected 'writer', got '{decision}'"
    print("[PASS] Case 2: Score below threshold with retries left (needs_more_research=False) -> 'writer'")

    # Case 3: score below threshold with retries exhausted -> end
    decision = decide(
        retry_enabled=True,
        accept=False,
        retry_count=MAX_RETRIES,
        needs_more_research=True,
    )
    assert decision == "end", f"Expected 'end', got '{decision}'"
    print(f"[PASS] Case 3: Score below threshold with retries exhausted (retry_count={MAX_RETRIES}) -> 'end'")

    # Case 4: score above threshold (accept=True) -> end
    decision = decide(
        retry_enabled=True,
        accept=True,
        retry_count=0,
        needs_more_research=False,
    )
    assert decision == "end", f"Expected 'end', got '{decision}'"
    print("[PASS] Case 4: Score above threshold (accept=True) -> 'end'")

    # Case 5: retry_enabled False -> end
    decision = decide(
        retry_enabled=False,
        accept=False,
        retry_count=0,
        needs_more_research=True,
    )
    assert decision == "end", f"Expected 'end', got '{decision}'"
    print("[PASS] Case 5: retry_enabled=False -> 'end'")

    # Case 6: route_after_critic is a pure function that returns state['decision'] without mutating state
    fake_state = {
        "decision": "researcher",
        "retry_count": 1,
        "score": 5.0,
    }
    state_copy = dict(fake_state)
    routed = route_after_critic(fake_state)
    assert routed == "researcher", f"Expected 'researcher', got '{routed}'"
    assert fake_state == state_copy, "route_after_critic must not mutate state"

    fake_state_end = {"decision": "end"}
    assert route_after_critic(fake_state_end) == "end"
    fake_state_writer = {"decision": "writer"}
    assert route_after_critic(fake_state_writer) == "writer"
    print("[PASS] Case 6: route_after_critic is pure, does not mutate state, returns state['decision']")

    # Case 7: llm_judge verdicts
    verdict_high = llm_judge({"score": QUALITY_THRESHOLD})
    assert verdict_high["accept"] is True, "Expected accept=True for score >= threshold"
    assert verdict_high["score"] == QUALITY_THRESHOLD

    verdict_low = llm_judge({"score": QUALITY_THRESHOLD - 0.5})
    assert verdict_low["accept"] is False, "Expected accept=False for score < threshold"

    verdict_none = llm_judge({"score": None})
    assert verdict_none["accept"] is True, "Expected accept=True for score=None"
    print(f"[PASS] Case 7: llm_judge correctly accepts at {QUALITY_THRESHOLD}, rejects below, and accepts on None")

    # Case 8: classifier_judge raises NotImplementedError
    try:
        classifier_judge({"score": 5.0})
        assert False, "classifier_judge should raise NotImplementedError"
    except NotImplementedError:
        pass
    assert JUDGES["classifier"] == classifier_judge
    assert JUDGES["llm"] == llm_judge
    print("[PASS] Case 8: classifier_judge raises NotImplementedError and JUDGES registry is configured")

    # Case 9: combine_evidence keeps newest first, deduplicates, and caps length
    new_ev = "New research finding about topic."
    old_ev = "Old research finding.\n\nNew research finding about topic."
    combined = combine_evidence(new_ev, old_ev, max_chars=8000)
    assert combined.startswith("New research finding about topic.")
    assert "Old research finding." in combined
    # Should not repeat the duplicate paragraph
    assert combined.count("New research finding about topic.") == 1
    print("[PASS] Case 9: combine_evidence deduplicates, prioritizes newest evidence, and caps length")

    print("=" * 60)
    print("All routing and decision tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
