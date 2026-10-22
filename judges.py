import os
import json
from typing import TypedDict, Optional, Callable, List, Dict, Any
from config import QUALITY_THRESHOLD, FAITHFULNESS_THRESHOLD

class JudgeVerdict(TypedDict, total=False):
    accept: bool
    score: Optional[float]   # judge's own quality/faithfulness number, if any
    reason: str              # short human-readable explanation
    details: Optional[dict]  # optional details dict

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "faithfulness",
    "models",
    "classifier_config.json",
)

# Test/mock injection hook
SCORE_CLAIMS_FN: Optional[Callable] = None

# Global cache for LR artifact to ensure single load per process
_CACHED_LR_MODEL = None
_CACHED_LR_PATH = None


def llm_judge(state: dict) -> JudgeVerdict:
    """Accept if the critic's score >= QUALITY_THRESHOLD. If score is None, accept."""
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


def score_claims_lr(claims: List[str], passages: List[str], k: int = 3, lr_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Score claims against passages using the cached Logistic Regression model."""
    global _CACHED_LR_MODEL, _CACHED_LR_PATH
    import joblib
    import numpy as np
    from faithfulness.embedder import embed
    from faithfulness.features import build_claim_features

    path = lr_path or os.getenv(
        "FAITHFULNESS_LR_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithfulness", "models", "faithfulness_lr.joblib"),
    )
    if _CACHED_LR_MODEL is None or _CACHED_LR_PATH != path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"LR model artifact not found at '{path}'.")
        artifact = joblib.load(path)
        _CACHED_LR_MODEL = artifact["model"]
        _CACHED_LR_PATH = path

    model = _CACHED_LR_MODEL
    c_embs = embed(claims)
    p_embs = embed(passages) if passages else np.zeros((0, c_embs.shape[1]), dtype=np.float32)

    results = []
    for i, claim in enumerate(claims):
        c_emb = c_embs[i]
        if len(passages) > 0:
            feats, top1_sim, _ = build_claim_features(c_emb, p_embs, k=min(k, 2))
            prob = float(model.predict_proba(feats.reshape(1, -1))[0, 1])
        else:
            prob = 0.0
            top1_sim = 0.0
        results.append({
            "claim": claim,
            "p_supported": prob,
            "top1_cosine": top1_sim,
        })
    return results


def classifier_judge(state: dict) -> JudgeVerdict:
    """Faithfulness classifier judge (System 4).
    
    1. Reads tau, backend, and model_source from classifier_config.json.
    2. Extracts all factual claims from state["report"].
    3. Cleans and splits accumulated evidence (search_results + scraped_content).
    4. Scores every claim against top-3 retrieved evidence passages (Way 2).
    5. Calculates supported_fraction and accepts if supported_fraction >= FAITHFULNESS_THRESHOLD.
    """
    # 1. Load config
    config_path = os.getenv("CLASSIFIER_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Classifier configuration file not found at '{config_path}'. "
            "Run evaluation to generate classifier_config.json."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load classifier configuration from '{config_path}': {exc}") from exc

    tau = float(cfg.get("tau", 0.66))
    backend = cfg.get("backend", "cross_encoder")
    cfg_model_source = cfg.get("model_source", "")

    # Environment override check
    env_backend = os.getenv("CLASSIFIER_BACKEND")
    if env_backend and env_backend != backend:
        print(f"[CLASSIFIER JUDGE] Overriding backend '{backend}' with environment setting '{env_backend}'")
        backend = env_backend

    # Resolve model source
    if backend == "cross_encoder":
        from faithfulness.cross_encoder import get_loaded_source
        model_source = get_loaded_source() or cfg_model_source or os.getenv("FAITHFULNESS_CE_PATH") or "vaibhavarya/distilbert-faithfulness-nli"
    elif backend == "lr":
        model_source = cfg_model_source or os.getenv(
            "FAITHFULNESS_LR_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithfulness", "models", "faithfulness_lr.joblib"),
        )
    else:
        model_source = cfg_model_source or backend

    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", str(FAITHFULNESS_THRESHOLD)))

    # 2. Extract claims from report
    from faithfulness.claims import extract_claims
    report = state.get("report") or ""
    claims = extract_claims(report)

    # Edge case: zero claims extracted -> accept with reason "no checkable claims"
    if not claims:
        details = {
            "n_claims": 0,
            "n_supported": 0,
            "tau": tau,
            "threshold": threshold,
            "backend": backend,
            "model_source": model_source,
            "unsupported_claims": [],
        }
        return JudgeVerdict(
            accept=True,
            score=1.0,
            reason="No checkable claims found in report; accepting by default.",
            details=details,
        )

    # 3. Clean and split accumulated evidence
    from scripts.collect_claims import clean_evidence_passages
    search_results = state.get("search_results") or ""
    scraped_content = state.get("scraped_content") or ""

    contexts: List[str] = []
    if search_results:
        for part in search_results.split("-----"):
            cleaned = part.strip()
            if cleaned:
                contexts.append(cleaned)
    if scraped_content and scraped_content.strip():
        contexts.append(scraped_content.strip())

    passages = clean_evidence_passages(contexts)

    # 4. Score claims
    scorer = state.get("classifier_stub") or SCORE_CLAIMS_FN
    if scorer is not None:
        try:
            scored = scorer(claims, passages, k=3)
        except TypeError:
            try:
                scored = scorer(claims, passages)
            except TypeError:
                scored = scorer(claims)
    else:
        if backend == "cross_encoder":
            from faithfulness.cross_encoder import score_claims
            scored = score_claims(claims, passages, k=3)
        elif backend == "lr":
            scored = score_claims_lr(claims, passages, k=3)
        else:
            raise ValueError(f"Unknown classifier backend: {backend}")

    # 5. Determine supported fraction and acceptance
    supported = []
    unsupported = []
    for idx, item in enumerate(scored):
        if isinstance(item, dict):
            c = item.get("claim", claims[idx] if idx < len(claims) else "")
            p = float(item.get("p_supported", 0.0))
        elif isinstance(item, (int, float)):
            c = claims[idx] if idx < len(claims) else ""
            p = float(item)
        else:
            raise TypeError(f"Unexpected scored claim format: {type(item)}")

        if p >= tau:
            supported.append({"claim": c, "p_supported": p})
        else:
            unsupported.append({"claim": c, "p_supported": p})

    n_claims = len(claims)
    n_supported = len(supported)
    supported_fraction = n_supported / n_claims if n_claims > 0 else 0.0
    accept = bool(supported_fraction >= threshold)

    if accept:
        reason = (
            f"Faithfulness score {supported_fraction:.3f} ({n_supported}/{n_claims} claims supported) "
            f"meets threshold {threshold:.2f} (tau={tau:.2f})."
        )
    else:
        reason = (
            f"Faithfulness score {supported_fraction:.3f} ({n_supported}/{n_claims} claims supported) "
            f"below threshold {threshold:.2f} (tau={tau:.2f})."
        )

    details = {
        "n_claims": n_claims,
        "n_supported": n_supported,
        "tau": tau,
        "threshold": threshold,
        "backend": backend,
        "model_source": model_source,
        "unsupported_claims": unsupported[:20],
    }

    return JudgeVerdict(
        accept=accept,
        score=supported_fraction,
        reason=reason,
        details=details,
    )


JUDGES: dict[str, Callable] = {"llm": llm_judge, "classifier": classifier_judge}
