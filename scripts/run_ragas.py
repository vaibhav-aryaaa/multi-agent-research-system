#!/usr/bin/env python3
"""RAGAS Evaluation Runner for Multi-Agent Researcher (Tasks E & F).

Supports:
- Evaluation of Phase A (Faithfulness, Answer Relevancy) and Phase B (Context Precision, Context Recall).
- Multiple judge providers (Gemini 3.1 Flash Lite on free tier, Groq Qwen-27b, and deterministic offline dry-run).
- Request-rate governor (max 10 requests/minute, rolling 60s window).
- Daily request budget (default 450 requests/day, tracked in results/ragas/usage_log.jsonl).
- Request-size guard (caps prompts at ~100,000 estimated tokens; trims contexts oldest-first).
- <think>...</think> stripping for reasoning models.
- Transactional per-sample JSON saving with granular resumption.
- Generation of pooled-evidence reference answers for Phase B.
- Summary generation to results/ragas/summary.csv with mean metrics and null tracking.
"""

import os
import sys
import time
import json
import math
import re
import uuid
import argparse
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import ragas

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "eval")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "ragas")
DEFAULT_DAILY_REQUEST_BUDGET = 450
DEFAULT_DAILY_TOKEN_BUDGET = 170000
MAX_REQUEST_SIZE_TOKENS = 100000
SYSTEM_NAMES = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]


# ---------------------------------------------------------------------------
# Rate Limiting & Usage Logging
# ---------------------------------------------------------------------------

class RagasGovernor:
    """Manages request-rate governor (<= 10 RPM) and rolling 24h request budget."""

    def __init__(
        self,
        usage_log_path: str,
        max_rpm: int = 10,
        daily_budget: int = DEFAULT_DAILY_REQUEST_BUDGET,
        time_fn=None,
        sleep_fn=None,
    ):
        self.usage_log_path = usage_log_path
        self.max_rpm = max_rpm
        self.daily_budget = daily_budget
        self.time_fn = time_fn or time.time
        self.sleep_fn = sleep_fn or time.sleep
        self.rolling_60s_timestamps: List[float] = []
        self.total_requests_session = 0
        self.think_stripped_count = 0

    def compute_24h_requests(self) -> Tuple[int, Optional[datetime]]:
        """Compute total requests in rolling 24 hours and oldest timestamp."""
        if not os.path.exists(self.usage_log_path):
            return 0, None

        now = datetime.now()
        window_start = now - timedelta(hours=24)
        total_reqs = 0
        oldest_time: Optional[datetime] = None

        try:
            with open(self.usage_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts >= window_start:
                            req_count = int(entry.get("requests", 1))
                            total_reqs += req_count
                            if oldest_time is None or ts < oldest_time:
                                oldest_time = ts
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Warning] Failed to read ragas usage log: {e}")

        return total_reqs, oldest_time

    def acquire_request(self, provider: str, model: str, prompt_chars: int = 0):
        """Enforce 10 RPM limit and daily budget before firing an LLM request."""
        now_dt = datetime.now()
        now_ts = self.time_fn()

        # 1. Enforce daily budget
        used_24h, oldest_in_window = self.compute_24h_requests()
        if used_24h + 1 > self.daily_budget:
            if oldest_in_window:
                expires_at = oldest_in_window + timedelta(hours=24)
                seconds_left = max(0, int((expires_at - now_dt).total_seconds()))
                hh = seconds_left // 3600
                mm = (seconds_left % 3600) // 60
                wait_str = f"{hh:02d}:{mm:02d}"
            else:
                wait_str = "24:00"

            print("\n" + "=" * 80)
            print(f"[RAGAS Governor] Daily request budget reached ({used_24h}/{self.daily_budget}).")
            print(f"Re-run in {wait_str} when the rolling 24h budget refreshes.")
            print("=" * 80 + "\n")
            sys.exit(0)

        # 2. Enforce rolling per-minute governor (max_rpm)
        cutoff = now_ts - 60.0
        self.rolling_60s_timestamps = [t for t in self.rolling_60s_timestamps if t > cutoff]

        if len(self.rolling_60s_timestamps) >= self.max_rpm:
            sleep_needed = 60.0 - (now_ts - self.rolling_60s_timestamps[0]) + 0.1
            if sleep_needed > 0:
                print(f"[RAGAS Governor] {len(self.rolling_60s_timestamps)} requests in last 60s (>= {self.max_rpm} RPM). Sleeping {sleep_needed:.1f}s...")
                self.sleep_fn(sleep_needed)
                now_ts = self.time_fn()
                self.rolling_60s_timestamps = [t for t in self.rolling_60s_timestamps if t > (now_ts - 60.0)]

        self.rolling_60s_timestamps.append(now_ts)
        self.total_requests_session += 1

        # 3. Log request
        os.makedirs(os.path.dirname(self.usage_log_path), exist_ok=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "requests": 1,
            "prompt_chars": prompt_chars,
        }
        with open(self.usage_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# Governed Chat Model Wrapper (<think> stripping, rate governor, billing check)
# ---------------------------------------------------------------------------

class GovernedJudgeChatModel(BaseChatModel):
    """Wraps an underlying LangChain chat model with rate limiting, think-tag stripping, and billing guards."""

    underlying_chat: Any
    governor: Any
    provider: str
    model_name: str

    def _strip_think_tags(self, text: str) -> str:
        if "<think>" in text:
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            self.governor.think_stripped_count += 1
            return cleaned
        return text

    def _check_billing_or_quota(self, exc: Exception):
        msg = str(exc).lower()
        if "402" in msg or "payment required" in msg or "billing account not found" in msg or "billing is not enabled" in msg:
            print("\n" + "!" * 80)
            print(f"[ERROR] API response indicated billing required: {exc}")
            print("Stopping immediately to maintain zero-cost guarantee.")
            print("!" * 80 + "\n")
            sys.exit(1)
        if "429" in msg or "quota" in msg or "rate limit" in msg:
            if "day" in msg or "daily" in msg or "tpd" in msg or "rpd" in msg:
                print(f"\n[RAGAS Governor] Daily quota exceeded (429): {exc}. Exiting cleanly.")
                sys.exit(0)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        prompt_chars = sum(len(str(m.content)) for m in messages)
        self.governor.acquire_request(self.provider, self.model_name, prompt_chars=prompt_chars)
        try:
            res = self.underlying_chat._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for gen in res.generations:
                cleaned = self._strip_think_tags(gen.text)
                gen.text = cleaned
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    gen.message.content = cleaned
            return res
        except Exception as e:
            self._check_billing_or_quota(e)
            raise

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        prompt_chars = sum(len(str(m.content)) for m in messages)
        self.governor.acquire_request(self.provider, self.model_name, prompt_chars=prompt_chars)
        try:
            res = await self.underlying_chat._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for gen in res.generations:
                cleaned = self._strip_think_tags(gen.text)
                gen.text = cleaned
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    gen.message.content = cleaned
            return res
        except Exception as e:
            self._check_billing_or_quota(e)
            raise

    @property
    def _llm_type(self) -> str:
        return f"governed_{self.provider}"


# ---------------------------------------------------------------------------
# Dry-run Fake Chat Model
# ---------------------------------------------------------------------------

class DryRunJudgeChatModel(BaseChatModel):
    """Deterministic fake chat model for offline tests and dry runs."""

    governor: Any
    think_stripped_count: int = 0

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        content = " ".join(str(m.content) for m in messages).lower()
        self.governor.acquire_request("dry_run", "fake-judge", prompt_chars=len(content))

        if "break down each sentence" in content:
            raw = "<think>Splitting report statements...</think>" + json.dumps({
                "statements": [
                    "The system demonstrates verifiable factual mechanisms.",
                    "The synthesis adheres strictly to retrieved research contexts.",
                ]
            })
        elif "judge the faithfulness" in content:
            raw = "<think>Cross-referencing claims against evidence...</think>" + json.dumps({
                "statements": [
                    {"statement": "The system demonstrates verifiable factual mechanisms.", "reason": "Directly grounded in evidence.", "verdict": 1},
                    {"statement": "The synthesis adheres strictly to retrieved research contexts.", "reason": "Supported by cited passages.", "verdict": 1},
                ]
            })
        elif "generate a question" in content:
            raw = "<think>Formulating relevant inquiry...</think>" + json.dumps({
                "question": "What mechanisms and findings were confirmed?",
                "noncommittal": 0
            })
        elif "context_precision" in content or "relevant" in content:
            raw = json.dumps({"verdict": 1, "reason": "Context passage contains direct technical findings."})
        elif "context_recall" in content or "attributed" in content:
            raw = json.dumps({"statements": ["The findings are fully verified in contexts."], "verdict": [1]})
        elif "pooled-evidence reference" in content or "reference answer" in content:
            raw = "Synthetic pooled-evidence reference summarizing the technical mechanisms and findings."
        else:
            raw = json.dumps({"verdict": 1, "reason": "Valid dry-run output."})

        # Strip think tags
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        self.governor.think_stripped_count += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=cleaned))])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "dry_run_judge"


# ---------------------------------------------------------------------------
# Judge Model Factory (Gemini, Groq, Dry-run)
# ---------------------------------------------------------------------------

def resolve_gemini_model_id(api_key: str, candidate_model: Optional[str] = None) -> str:
    """Resolve Gemini 3.1 Flash Lite API model ID from Google GenAI SDK."""
    if candidate_model and candidate_model.strip():
        return candidate_model.strip()

    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        all_models = list(client.models.list())
        names = [getattr(m, "name", str(m)) for m in all_models]

        # Match 3.1 flash lite or flash lite
        for name in names:
            low = name.lower()
            if "3.1" in low and "flash" in low and "lite" in low:
                print(f"[RAGAS Judge] Auto-resolved Gemini model id: {name}")
                return name

        for name in names:
            low = name.lower()
            if "flash" in low and "lite" in low:
                print(f"[RAGAS Judge] Selected Gemini Flash Lite model id: {name}")
                return name

        for name in names:
            low = name.lower()
            if "flash" in low:
                print(f"[RAGAS Judge] Selected Gemini Flash model id: {name}")
                return name

        print("\n[RAGAS Judge] Could not find '3.1 flash lite'. Available models:")
        for n in names[:20]:
            print(f"  - {n}")
        raise ValueError("No Gemini model matching '3.1 flash lite' found.")
    except Exception as e:
        if "Could not find '3.1 flash lite'" in str(e):
            raise
        print(f"[RAGAS Judge] Model listing warning: {e}. Defaulting to 'gemini-3.1-flash-lite'.")
        return "gemini-3.1-flash-lite"


def build_judge_llm(
    provider: str,
    governor: RagasGovernor,
    model_override: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[LangchainLLMWrapper, str]:
    """Factory creating governed RAGAS judge LLM."""
    if dry_run:
        fake_chat = DryRunJudgeChatModel(governor=governor)
        return LangchainLLMWrapper(fake_chat), "dry-run-fake-judge"

    provider_clean = (provider or "gemini").strip().lower()

    if provider_clean == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is missing or unconfigured. "
                "Set GOOGLE_API_KEY in your .env file or environment for the Gemini judge."
            )
        model_id = resolve_gemini_model_id(api_key, model_override)
        from langchain_google_genai import ChatGoogleGenerativeAI
        base_chat = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=0.0,
        )
        governed_chat = GovernedJudgeChatModel(
            underlying_chat=base_chat,
            governor=governor,
            provider="gemini",
            model_name=model_id,
        )
        return LangchainLLMWrapper(governed_chat), model_id

    elif provider_clean == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env.")
        model_id = model_override or os.getenv("RAGAS_JUDGE_MODEL", "qwen/qwen3.8-27b")
        from langchain_groq import ChatGroq
        base_chat = ChatGroq(
            model_name=model_id,
            api_key=api_key,
            temperature=0.0,
        )
        governed_chat = GovernedJudgeChatModel(
            underlying_chat=base_chat,
            governor=governor,
            provider="groq",
            model_name=model_id,
        )
        return LangchainLLMWrapper(governed_chat), model_id

    else:
        raise ValueError(f"Unsupported RAGAS judge provider: {provider}")


def build_local_embeddings(dry_run: bool = False):
    """Load local sentence-transformers MiniLM embeddings wrapper."""
    if dry_run:
        class FakeEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        return LangchainEmbeddingsWrapper(FakeEmbeddings())

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    raw_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(raw_emb)


# ---------------------------------------------------------------------------
# Request-size Guard & Context Trimming
# ---------------------------------------------------------------------------

def trim_contexts_to_budget(
    topic: str,
    report: str,
    contexts: List[str],
    max_tokens: int = MAX_REQUEST_SIZE_TOKENS,
) -> Tuple[List[str], int]:
    """Trim retrieved contexts oldest-first (tail of list) until under max_tokens."""
    non_context_chars = len(topic) + len(report)
    max_chars = int(max_tokens * 3.5)
    allowed_context_chars = max(0, max_chars - non_context_chars)

    original_chars = sum(len(c) for c in contexts)
    if original_chars <= allowed_context_chars:
        return contexts, 0

    trimmed_contexts = []
    accumulated = 0
    # Keep newest first (earlier elements), drop oldest (tail elements)
    for c in contexts:
        if accumulated + len(c) <= allowed_context_chars:
            trimmed_contexts.append(c)
            accumulated += len(c)
        else:
            remaining_chars = allowed_context_chars - accumulated
            if remaining_chars > 200:
                trimmed_contexts.append(c[:remaining_chars])
                accumulated += remaining_chars
            break

    trimmed_chars = original_chars - sum(len(c) for c in trimmed_contexts)
    if trimmed_chars > 0:
        print(f"[Guard] Trimmed {trimmed_chars:,} chars of oldest contexts to stay under {max_tokens:,} tokens.")
    return trimmed_contexts, trimmed_chars


# ---------------------------------------------------------------------------
# Pooled-Evidence Reference Generator (Phase B)
# ---------------------------------------------------------------------------

def generate_pooled_reference(
    qid: str,
    topic: str,
    s2_contexts: List[str],
    s3_contexts: List[str],
    s4_contexts: List[str],
    judge_wrapper: LangchainLLMWrapper,
    ref_dir: str,
    dry_run: bool = False,
) -> str:
    """Generate or load a <= 250 word pooled-evidence reference answer from union of S2, S3, S4."""
    os.makedirs(ref_dir, exist_ok=True)
    ref_file = os.path.join(ref_dir, f"{qid}.json")
    if os.path.exists(ref_file):
        with open(ref_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("reference", "")

    # Deduplicate contexts
    seen = set()
    deduped_contexts = []
    for c in (s2_contexts + s3_contexts + s4_contexts):
        snippet = c.strip()
        if snippet and snippet not in seen:
            seen.add(snippet)
            deduped_contexts.append(snippet)

    evidence_text = "\n\n---\n\n".join(deduped_contexts[:15])  # Cap to top 15 evidence snippets

    if dry_run:
        ref_text = f"Synthetic reference for {qid} ({topic}) summarizing consensus findings across pooled evidence."
    else:
        prompt = (
            f"You are a technical research evaluator. Synthesize an objective, authoritative reference answer "
            f"strictly under 250 words answering the following topic, based solely on the provided accumulated research evidence.\n\n"
            f"TOPIC: {topic}\n\n"
            f"ACCUMULATED EVIDENCE:\n{evidence_text}\n\n"
            f"REFERENCE ANSWER (concise, <= 250 words, factual, no fluff):"
        )
        msg = HumanMessage(content=prompt)
        res = judge_wrapper.langchain_llm.invoke([msg])
        ref_text = res.content.strip()

    record = {
        "qid": qid,
        "topic": topic,
        "reference": ref_text,
        "reference_type": "pooled_evidence",
        "created_at": datetime.now().isoformat(),
    }
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return ref_text


def extract_unsupported_claims(eval_res: Any) -> List[Dict[str, Any]]:
    """Extract claims marked unsupported (verdict == 0) from Ragas evaluation traces."""
    unsupported = []
    traces = getattr(eval_res, "ragas_traces", {})
    if not isinstance(traces, dict):
        return unsupported
    for node in traces.values():
        name = getattr(node, "name", "")
        if "n_l_i" in name:
            inputs = getattr(node, "inputs", {})
            data = inputs.get("data") if isinstance(inputs, dict) else None
            ctx = getattr(data, "context", "") if data else ""
            outputs = getattr(node, "outputs", {})
            out_list = outputs.get("output", []) if isinstance(outputs, dict) else []
            for item in out_list:
                for s in getattr(item, "statements", []):
                    if getattr(s, "verdict", None) == 0:
                        unsupported.append({
                            "claim": getattr(s, "statement", ""),
                            "reason": getattr(s, "reason", ""),
                            "evidence": ctx,
                        })
    return unsupported


# ---------------------------------------------------------------------------
# Core Evaluation Loop
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    eval_results_dir: str,
    ragas_out_dir: str,
    phase: str = "A",
    only_qids: Optional[List[str]] = None,
    systems_to_eval: Optional[List[str]] = None,
    limit: Optional[int] = None,
    daily_request_budget: int = DEFAULT_DAILY_REQUEST_BUDGET,
    max_rpm: Optional[int] = None,
    dry_run: bool = False,
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
):
    os.makedirs(ragas_out_dir, exist_ok=True)
    ref_dir = os.path.join(ragas_out_dir, "references")
    usage_log_path = os.path.join(ragas_out_dir, "usage_log.jsonl")

    actual_max_rpm = max_rpm if max_rpm is not None else (10000 if dry_run else 10)
    governor = RagasGovernor(usage_log_path, max_rpm=actual_max_rpm, daily_budget=daily_request_budget)
    judge_wrapper, resolved_model = build_judge_llm(
        provider=provider or os.getenv("RAGAS_JUDGE_PROVIDER", "gemini"),
        governor=governor,
        model_override=model_override or os.getenv("RAGAS_JUDGE_MODEL"),
        dry_run=dry_run,
    )
    embeddings_wrapper = build_local_embeddings(dry_run=dry_run)

    # Scan available complete questions
    candidate_qids = [f"Q{i:02d}" for i in range(1, 26)]
    if only_qids:
        candidate_qids = [q for q in candidate_qids if q in only_qids]

    eligible_questions = []
    for qid in candidate_qids:
        # Must have all four system files
        all_exist = all(
            os.path.exists(os.path.join(eval_results_dir, f"{qid}_{s}.json"))
            for s in SYSTEM_NAMES
        )
        if all_exist:
            eligible_questions.append(qid)

    if limit and limit > 0:
        eligible_questions = eligible_questions[:limit]

    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RUNNER (Phase A / Phase B)")
    print(f"Phase:                {phase}")
    print(f"Eligible questions:   {len(eligible_questions)} ({', '.join(eligible_questions)})")
    print(f"Judge Provider:       {governor.provider if hasattr(governor, 'provider') else (provider or 'gemini')}")
    print(f"Judge Model:          {resolved_model}")
    print(f"RAGAS Version:        {ragas.__version__}")
    print(f"Output directory:     {ragas_out_dir}")
    print(f"Daily request budget: {daily_request_budget} requests")
    print(f"Mode:                 {'DRY-RUN (fakes, no APIs)' if dry_run else 'LIVE'}")
    print("=" * 80 + "\n")

    target_systems = systems_to_eval or SYSTEM_NAMES
    all_sample_results = []

    # Table Header
    print(f"{'Sample':<30} | {'Faithfulness':<14} | {'Relevancy':<12} | {'Context Prec':<14} | {'Context Rec':<13} | {'Seconds':<8}")
    print("-" * 105)

    for qid in eligible_questions:
        # Load all four runs for borrowing contexts and pooling reference
        runs_data = {}
        for s in SYSTEM_NAMES:
            with open(os.path.join(eval_results_dir, f"{qid}_{s}.json"), "r", encoding="utf-8") as f:
                runs_data[s] = json.load(f)

        topic = runs_data["fixed"].get("topic", "")

        # Optional reference for Phase B
        reference_text = None
        if phase.upper() == "B":
            reference_text = generate_pooled_reference(
                qid=qid,
                topic=topic,
                s2_contexts=runs_data["fixed"].get("contexts", []),
                s3_contexts=runs_data["adaptive_llm"].get("contexts", []),
                s4_contexts=runs_data["adaptive_classifier"].get("contexts", []),
                judge_wrapper=judge_wrapper,
                ref_dir=ref_dir,
                dry_run=dry_run,
            )

        for s in target_systems:
            sample_id = f"{qid}_{s}"
            out_file = os.path.join(ragas_out_dir, f"{sample_id}.json")
            err_file = os.path.join(ragas_out_dir, f"{sample_id}_ERROR.json")

            # 1. Resumption check
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    scores = existing.get("scores", {})
                    need_eval = False
                    if phase.upper() == "A" and ("faithfulness" not in scores or "answer_relevancy" not in scores):
                        need_eval = True
                    elif phase.upper() == "B" and ("context_precision" not in scores or "context_recall" not in scores):
                        need_eval = True

                    if not need_eval:
                        f_val = f"{scores.get('faithfulness'):.3f}" if scores.get("faithfulness") is not None else "null"
                        a_val = f"{scores.get('answer_relevancy'):.3f}" if scores.get("answer_relevancy") is not None else "null"
                        p_val = f"{scores.get('context_precision'):.3f}" if scores.get("context_precision") is not None else "null"
                        r_val = f"{scores.get('context_recall'):.3f}" if scores.get("context_recall") is not None else "null"
                        print(f"{sample_id:<30} | {f_val:<14} | {a_val:<12} | {p_val:<14} | {r_val:<13} | {'REUSED':<8}")
                        all_sample_results.append(existing)
                        continue
                except Exception:
                    pass

            t0 = time.time()
            requests_before = governor.total_requests_session
            think_before = governor.think_stripped_count

            # 2. Context setup
            if s == "baseline":
                raw_contexts = list(runs_data["fixed"].get("contexts", []))
                contexts_source = "borrowed_from_S2"
            else:
                raw_contexts = list(runs_data[s].get("contexts", []))
                contexts_source = "system_contexts"

            report = runs_data[s].get("final_report", "")

            # 3. Request-size guard: trim contexts if needed
            safe_contexts, trimmed_chars = trim_contexts_to_budget(topic, report, raw_contexts, max_tokens=MAX_REQUEST_SIZE_TOKENS)

            # 4. Construct Ragas Sample & Dataset
            sample = SingleTurnSample(
                user_input=topic,
                response=report,
                retrieved_contexts=safe_contexts,
                reference=reference_text,
            )
            dataset = EvaluationDataset(samples=[sample])

            # Select metrics based on phase & system
            metrics_to_run = []
            errors_dict = {}

            if phase.upper() == "A":
                metrics_to_run = [faithfulness, answer_relevancy]
            elif phase.upper() == "B":
                if s == "baseline":
                    # S1 has no retrieval; Context Precision/Recall are strictly null
                    metrics_to_run = []
                else:
                    metrics_to_run = [context_precision, context_recall]

            scores_dict = {}
            unsupported_claims = []

            if metrics_to_run:
                try:
                    run_cfg = RunConfig(max_workers=1, timeout=300, max_retries=2)
                    eval_res = evaluate(
                        dataset=dataset,
                        metrics=metrics_to_run,
                        llm=judge_wrapper,
                        embeddings=embeddings_wrapper,
                        run_config=run_cfg,
                        raise_exceptions=False,
                    )
                    unsupported_claims = extract_unsupported_claims(eval_res)
                    df = eval_res.to_pandas()
                    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        if col in df.columns:
                            val = df[col].iloc[0]
                            if val is None or (isinstance(val, float) and math.isnan(val)):
                                scores_dict[col] = None
                                errors_dict[col] = "Metric returned NaN / Output parsing failed"
                            else:
                                scores_dict[col] = float(val)
                except Exception as exc:
                    err_msg = str(exc)
                    print(f"[{sample_id}] Evaluation error: {err_msg}")
                    with open(err_file, "w", encoding="utf-8") as ef:
                        json.dump({
                            "sample": sample_id,
                            "error": err_msg,
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat(),
                        }, ef, indent=2)
                    continue

            # Ensure all metrics for phase have keys
            if phase.upper() == "A":
                for k in ["faithfulness", "answer_relevancy"]:
                    if k not in scores_dict:
                        scores_dict[k] = None
            elif phase.upper() == "B":
                for k in ["context_precision", "context_recall"]:
                    if k not in scores_dict:
                        scores_dict[k] = None

            t_elapsed = time.time() - t0
            requests_used = governor.total_requests_session - requests_before
            think_stripped = governor.think_stripped_count - think_before

            # Merge with prior scores if output file already exists (e.g. from previous phase)
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        prior = json.load(f)
                    prior_scores = prior.get("scores", {})
                    prior_scores.update(scores_dict)
                    scores_dict = prior_scores
                    prior_errors = prior.get("errors", {})
                    prior_errors.update(errors_dict)
                    errors_dict = prior_errors
                    requests_used += int(prior.get("requests_used", 0))
                    t_elapsed += float(prior.get("seconds", 0.0))
                    think_stripped += int(prior.get("think_stripped_count", 0))
                except Exception:
                    pass

            record = {
                "qid": qid,
                "system": s,
                "topic": topic,
                "phase": phase.upper(),
                "scores": scores_dict,
                "contexts_source": contexts_source,
                "judge_provider": provider or os.getenv("RAGAS_JUDGE_PROVIDER", "gemini"),
                "judge_model": resolved_model,
                "ragas_version": ragas.__version__,
                "requests_used": requests_used,
                "seconds": round(t_elapsed, 2),
                "errors": errors_dict,
                "contexts_trimmed_chars": trimmed_chars,
                "think_stripped_count": think_stripped,
                "unsupported_claims": unsupported_claims,
                "timestamp": datetime.now().isoformat(),
            }

            # Transactional atomic write
            temp_out = f"{out_file}.tmp.{uuid.uuid4().hex[:6]}"
            with open(temp_out, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            os.replace(temp_out, out_file)

            if os.path.exists(err_file):
                try:
                    os.remove(err_file)
                except OSError:
                    pass

            all_sample_results.append(record)

            f_s = f"{scores_dict.get('faithfulness'):.3f}" if scores_dict.get("faithfulness") is not None else "null"
            a_s = f"{scores_dict.get('answer_relevancy'):.3f}" if scores_dict.get("answer_relevancy") is not None else "null"
            p_s = f"{scores_dict.get('context_precision'):.3f}" if scores_dict.get("context_precision") is not None else "null"
            r_s = f"{scores_dict.get('context_recall'):.3f}" if scores_dict.get("context_recall") is not None else "null"

            print(f"{sample_id:<30} | {f_s:<14} | {a_s:<12} | {p_s:<14} | {r_s:<13} | {t_elapsed:.1f}s")

    # Write summary CSV
    summary_path = os.path.join(ragas_out_dir, "summary.csv")
    write_summary_csv(all_sample_results, summary_path)


def write_summary_csv(records: List[Dict[str, Any]], summary_path: str):
    """Write tabular evaluation results to summary.csv and print per-system aggregations."""
    import pandas as pd

    if not records:
        print("\n[RAGAS] No records to summarize.")
        return

    flattened = []
    for r in records:
        scores = r.get("scores", {})
        row = {
            "qid": r.get("qid"),
            "system": r.get("system"),
            "phase": r.get("phase"),
            "contexts_source": r.get("contexts_source"),
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
            "context_recall": scores.get("context_recall"),
            "requests_used": r.get("requests_used"),
            "seconds": r.get("seconds"),
            "contexts_trimmed_chars": r.get("contexts_trimmed_chars"),
            "think_stripped_count": r.get("think_stripped_count"),
        }
        flattened.append(row)

    df = pd.DataFrame(flattened)
    df.to_csv(summary_path, index=False)
    print(f"\n[RAGAS] Wrote detailed summary to {summary_path}")

    # Print Per-System Means and Null Counts
    print("\n" + "=" * 80)
    print("PER-SYSTEM METRIC SUMMARY")
    print("=" * 80)
    print(f"{'System':<22} | {'Metric':<18} | {'Mean':<8} | {'N':<4} | {'Nulls':<5}")
    print("-" * 65)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for sys_name in SYSTEM_NAMES:
        sub = df[df["system"] == sys_name]
        for m in metrics:
            if m in sub.columns:
                series = sub[m]
                valid = series.dropna()
                mean_val = f"{valid.mean():.3f}" if len(valid) > 0 else "N/A"
                null_count = int(series.isna().sum())
                print(f"{sys_name:<22} | {m:<18} | {mean_val:<8} | {len(valid):<4} | {null_count:<5}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Runner (Tasks E & F)")
    parser.add_argument("--results", default=DEFAULT_RESULTS_DIR, help="Path to pipeline results (results/eval)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Path to output RAGAS results (results/ragas)")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated QIDs to run (e.g. Q01,Q02)")
    parser.add_argument("--systems", type=str, default=None, help="Comma-separated system names")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "a", "b"], help="Evaluation Phase (A or B)")
    parser.add_argument("--dry-run", action="store_true", help="Run with deterministic offline fake judge and fake embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--daily-request-budget", type=int, default=DEFAULT_DAILY_REQUEST_BUDGET, help="Daily request budget")
    parser.add_argument("--daily-token-budget", type=int, default=DEFAULT_DAILY_TOKEN_BUDGET, help="Daily token budget")
    parser.add_argument("--max-rpm", type=int, default=10, help="Max requests per minute")
    parser.add_argument("--provider", type=str, default=None, help="Judge provider (gemini or groq)")
    parser.add_argument("--model", type=str, default=None, help="Judge model name/id override")
    args = parser.parse_args()

    only_qids = [q.strip().upper() for q in args.only.split(",")] if args.only else None
    systems = [s.strip().lower() for s in args.systems.split(",")] if args.systems else None

    # In dry-run mode, default max_rpm to 10,000 unless user explicitly set it
    effective_max_rpm = args.max_rpm
    if args.dry_run and args.max_rpm == 10:
        effective_max_rpm = 10000

    run_ragas_evaluation(
        eval_results_dir=os.path.abspath(args.results),
        ragas_out_dir=os.path.abspath(args.out),
        phase=args.phase.upper(),
        only_qids=only_qids,
        systems_to_eval=systems,
        limit=args.limit,
        daily_request_budget=args.daily_request_budget,
        max_rpm=effective_max_rpm,
        dry_run=args.dry_run,
        provider=args.provider,
        model_override=args.model,
    )


if __name__ == "__main__":
    main()
