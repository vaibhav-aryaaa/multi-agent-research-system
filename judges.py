from typing import TypedDict, Optional, Callable
from config import QUALITY_THRESHOLD

class JudgeVerdict(TypedDict):
    accept: bool
    score: Optional[float]   # judge's own quality/faithfulness number, if any
    reason: str              # short human-readable explanation

def llm_judge(state: dict) -> JudgeVerdict:
    """Accept if the critic's score >= QUALITY_THRESHOLD. If score is None, accept."""
    import os
    threshold = float(os.getenv("QUALITY_THRESHOLD", str(QUALITY_THRESHOLD)))
    score = state.get("score")
    if score is None:
        return JudgeVerdict(
            accept=True,
            score=None,
            reason="Critic score was unparseable; accepting by default.",
        )
    if score >= threshold:
        return JudgeVerdict(
            accept=True,
            score=score,
            reason=f"Critic score {score:.1f} meets quality threshold {threshold:.1f}.",
        )
    return JudgeVerdict(
        accept=False,
        score=score,
        reason=f"Critic score {score:.1f} below quality threshold {threshold:.1f}.",
    )

def classifier_judge(state: dict) -> JudgeVerdict:
    raise NotImplementedError("Implemented in Phase 3")

JUDGES: dict[str, Callable] = {"llm": llm_judge, "classifier": classifier_judge}
