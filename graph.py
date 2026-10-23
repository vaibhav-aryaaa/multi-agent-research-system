import os
import operator
import re
import time
from typing import TypedDict, Optional, Annotated, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.callbacks import BaseCallbackHandler

from config import QUALITY_THRESHOLD, MAX_RETRIES
from tools import web_search, scrape_url
from retrieval import parse_search_results, build_fallback_query
from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    revise_chain,
    critic_chain,
    query_writer_chain,
    CriticFeedback,
    model_name,
)
from judges import JUDGES, llm_judge, JudgeVerdict
from runlog import save_run

class GraphDependencies:
    """Injectable dependency container for tools, LLM chains, and agents."""
    def __init__(self):
        self.web_search = web_search
        self.scrape_url = scrape_url
        self.writer_chain = writer_chain
        self.revise_chain = revise_chain
        self.critic_chain = critic_chain
        self.query_writer_chain = query_writer_chain
        self.build_search_agent = build_search_agent
        self.build_reader_agent = build_reader_agent
        self.baseline_chain = None

DEPS = GraphDependencies()

def invoke_tool(tool_obj: Any, arg_dict: dict) -> Any:
    """Safely invoke a tool (LangChain BaseTool or plain callable)."""
    if hasattr(tool_obj, "invoke"):
        try:
            return tool_obj.invoke(arg_dict)
        except TypeError:
            if len(arg_dict) == 1:
                return tool_obj.invoke(next(iter(arg_dict.values())))
            raise
    elif callable(tool_obj):
        try:
            return tool_obj(**arg_dict)
        except TypeError:
            return tool_obj(arg_dict)
    raise TypeError(f"Tool {tool_obj} is neither a tool nor callable")

# Global token usage tracker for the active run
TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

# Optional rolling usage log file (for batch eval runner)
USAGE_LOG_PATH: Optional[str] = None

def set_usage_log_path(path: Optional[str]):
    global USAGE_LOG_PATH
    USAGE_LOG_PATH = path

def append_to_usage_log(tokens_dict: dict, source: str = "llm_call"):
    global USAGE_LOG_PATH
    if not USAGE_LOG_PATH:
        return
    tot = tokens_dict.get("total_tokens", 0)
    prompt = tokens_dict.get("prompt_tokens", 0)
    comp = tokens_dict.get("completion_tokens", 0)
    if tot <= 0 and prompt <= 0 and comp <= 0:
        return
    from datetime import datetime
    import json
    record = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "total_tokens": tot,
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(USAGE_LOG_PATH)), exist_ok=True)
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[Warning] Failed to append to usage log: {e}")

def reset_token_usage():
    global TOKEN_USAGE
    TOKEN_USAGE.clear()
    TOKEN_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

def get_token_usage() -> dict:
    return dict(TOKEN_USAGE)

def record_token_usage(source_name: str, usage_dict_or_response: Any):
    """Extract and accumulate token usage."""
    global TOKEN_USAGE
    prompt = 0
    completion = 0
    total = 0
    
    if isinstance(usage_dict_or_response, dict):
        # Could be agent result with messages
        msgs = usage_dict_or_response.get("messages", [])
        for m in msgs:
            um = getattr(m, "usage_metadata", None)
            if um and isinstance(um, dict):
                prompt += um.get("input_tokens", 0)
                completion += um.get("output_tokens", 0)
                total += um.get("total_tokens", 0)
        # Or direct usage dict
        if "prompt_tokens" in usage_dict_or_response:
            prompt += usage_dict_or_response.get("prompt_tokens", 0)
            completion += usage_dict_or_response.get("completion_tokens", 0)
            total += usage_dict_or_response.get("total_tokens", 0)
    
    if total > 0 or prompt > 0 or completion > 0:
        TOKEN_USAGE["prompt_tokens"] += prompt
        TOKEN_USAGE["completion_tokens"] += completion
        TOKEN_USAGE["total_tokens"] += total
        append_to_usage_log({
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }, source=source_name)
        print(f"[{source_name}] Tokens: prompt={prompt}, completion={completion}, total={total}")
        token_limit = int(os.getenv("TOKEN_LIMIT_PER_RUN", "60000"))
        if TOKEN_USAGE["total_tokens"] > token_limit:
            raise RuntimeError(f"STOP CONDITION: Total token usage ({TOKEN_USAGE['total_tokens']}) exceeded {token_limit} limit.")

class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track LLM token usage across all LCEL chains."""
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        prompt = 0
        completion = 0
        total = 0
        if hasattr(response, "llm_output") and response.llm_output and "token_usage" in response.llm_output:
            tu = response.llm_output["token_usage"]
            prompt = tu.get("prompt_tokens", 0)
            completion = tu.get("completion_tokens", 0)
            total = tu.get("total_tokens", 0)
        elif hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None)
                    if um and isinstance(um, dict):
                        prompt += um.get("input_tokens", 0)
                        completion += um.get("output_tokens", 0)
                        total += um.get("total_tokens", 0)
        record_token_usage("llm_call", {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        })

TOKEN_CALLBACK = TokenUsageCallback()

def invoke_with_retry(runnable, inputs, max_attempts: int = 3, model_label: str = ""):
    """Safely invoke an agent or chain with bounded, visible backoff on rate limits."""
    target_model = model_label or model_name
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(inputs, config={"callbacks": [TOKEN_CALLBACK]})
        except Exception as e:
            status_code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            err_msg = str(e)
            
            # Check for 413 (request too large): raise immediately
            if status_code == 413:
                print(f"Error 413: Request too large for {target_model}. Raising immediately.")
                raise RuntimeError(f"REQUEST TOO LARGE (413): {err_msg}") from e

            # Check for 400 tool_use_failed / invalid_request_error: retry once (max 2 attempts)
            if status_code == 400:
                if attempt < 2:
                    print(f"HTTP 400 error ({err_msg[:100]}...), retrying with exponential backoff (attempt {attempt}/2)")
                    time.sleep(2.0 * attempt)
                    continue
                else:
                    raise
                
            # Check for Rate Limit (429)
            if status_code == 429:
                # Daily limit hit (TPD / tokens per day): fail fast
                if "tokens per day" in err_msg.lower() or "tpd" in err_msg.lower():
                    raise RuntimeError(f"DAILY TOKEN LIMIT HIT for {target_model}. Stop and wait for reset or change tier.") from e
                
                # Per-minute limits (TPM / RPM)
                match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg)
                if match:
                    mins = float(match.group(1)) if match.group(1) else 0.0
                    secs = float(match.group(2))
                    wait_time = mins * 60.0 + secs
                else:
                    wait_time = 5.0 * attempt
                    
                if wait_time > 90.0:
                    raise RuntimeError(f"Requested rate limit wait ({wait_time:.1f}s) exceeds 90s maximum for {target_model}. Raising.") from e
                    
                print(f"Rate limited, sleeping {wait_time:.1f}s (attempt {attempt}/{max_attempts})")
                if attempt >= max_attempts:
                    print(f"Exceeded {max_attempts} retry attempts. Raising.")
                    raise
                    
                time.sleep(wait_time)
            else:
                raise

class ResearchState(TypedDict, total=False):
    topic: str
    search_results: str
    latest_search_results: str
    scraped_content: str
    report: str
    feedback: str
    score: Optional[float]
    attempt: int
    retry_count: int
    decision: str
    judge: str
    retry_enabled: bool
    used_urls: list[str]
    queries_used: list[str]
    attempts: Annotated[list, operator.add]
    token_usage: dict

def combine_evidence(new_content: str, old_content: str, max_chars: int = 8000) -> str:
    """Accumulate evidence: newest first, deduplicated, capped at max_chars."""
    if not old_content:
        return (new_content or "")[:max_chars]
    if not new_content:
        return (old_content or "")[:max_chars]

    new_clean = new_content.strip()
    old_clean = old_content.strip()
    if new_clean == old_clean:
        return new_clean[:max_chars]

    if "-----" in new_clean and "-----" in old_clean:
        new_chunks = [c.strip() for c in new_clean.split("-----") if c.strip()]
        old_chunks = [c.strip() for c in old_clean.split("-----") if c.strip()]
        unique_old = []
        for oc in old_chunks:
            if not any(oc in nc or nc in oc for nc in new_chunks):
                unique_old.append(oc)
        if unique_old:
            combined = new_clean + "\n-----\n" + "\n-----\n".join(unique_old)
        else:
            combined = new_clean
    else:
        new_paras = [p.strip() for p in new_clean.split("\n\n") if p.strip()]
        old_paras = [p.strip() for p in old_clean.split("\n\n") if p.strip()]
        unique_old = [p for p in old_paras if p not in new_paras and p not in new_clean]
        if unique_old:
            combined = new_clean + "\n\n" + "\n\n".join(unique_old)
        else:
            combined = new_clean

    return combined[:max_chars]

def decide(
    retry_enabled: bool,
    accept: bool,
    retry_count: int,
    needs_more_research: bool,
) -> str:
    """Pure decision logic for routing after critic evaluation."""
    if not retry_enabled or accept or retry_count >= MAX_RETRIES:
        return "end"
    return "researcher" if needs_more_research else "writer"

def researcher_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    attempt = state.get("attempt", 0)
    queries_used = list(state.get("queries_used") or [])

    if attempt == 0:
        query = topic
        queries_used.append(query)
        new_search_results = invoke_tool(DEPS.web_search, {"query": query})
    else:
        critique = (state.get("feedback") or "")[:500]
        try:
            raw_query = invoke_with_retry(
                DEPS.query_writer_chain,
                {
                    "topic": topic,
                    "critique": critique,
                    "queries_used": ", ".join(queries_used) if queries_used else "None",
                }
            )
            query = str(raw_query).strip().strip('"\'').strip()
            lines = [l.strip() for l in query.splitlines() if l.strip()]
            query = lines[0] if lines else ""
            words = query.split()
            if len(words) > 15:
                query = " ".join(words[:15])
        except Exception as e:
            print(f"[researcher] Query writer LLM call failed ({e}); falling back.")
            query = ""

        if not query:
            query = build_fallback_query(topic, critique)

        # Never reuse an identical query; if the new query equals a previous one, append "evidence review"
        if query in queries_used:
            query = f"{query} evidence review".strip()

        queries_used.append(query)
        new_search_results = invoke_tool(DEPS.web_search, {"query": query})

    new_search_results_str = str(new_search_results)

    if attempt > 0:
        accumulated = combine_evidence(new_search_results_str, state.get("search_results", ""), max_chars=8000)
    else:
        accumulated = new_search_results_str[:8000]

    return {
        "search_results": accumulated,
        "latest_search_results": new_search_results_str,
        "queries_used": queries_used,
    }

def reader_node(state: ResearchState) -> dict:
    attempt = state.get("attempt", 0)
    used_urls = list(state.get("used_urls") or [])

    # From the NEW search results, pick first URL not in used_urls
    latest_text = state.get("latest_search_results") or state.get("search_results", "")
    parsed_entries = parse_search_results(latest_text)
    candidate_urls = [e["url"] for e in parsed_entries if e.get("url")]
    unused_urls = [u for u in candidate_urls if u not in used_urls]

    # If new results have no unused URLs, fallback to earlier search results
    if not unused_urls:
        all_text = state.get("search_results", "")
        all_parsed = parse_search_results(all_text)
        all_candidate_urls = [e["url"] for e in all_parsed if e.get("url")]
        unused_urls = [u for u in all_candidate_urls if u not in used_urls]

    # If still none, skip scraping and leave scraped_content as whatever was already there
    if not unused_urls:
        print("[reader] No unused URLs found; skipping scraping.")
        return {
            "scraped_content": state.get("scraped_content", ""),
            "used_urls": used_urls,
        }

    scraped_text = ""
    max_tries = min(3, len(unused_urls))
    for i in range(max_tries):
        url_to_try = unused_urls[i]
        used_urls.append(url_to_try)
        try:
            content = invoke_tool(DEPS.scrape_url, {"url": url_to_try})
            content_str = str(content)
            is_error = (
                len(content_str.strip()) < 200
                or "extraction failed" in content_str.lower()
                or "failed to scrape" in content_str.lower()
            )
            if not is_error:
                scraped_text = content_str
                break
            else:
                print(f"[reader] Scraping failed or returned <200 chars for {url_to_try}, trying next...")
        except Exception as e:
            print(f"[reader] Exception scraping {url_to_try}: {e}, trying next...")

    if not scraped_text:
        new_scraped = state.get("scraped_content", "")
    else:
        if attempt > 0:
            new_scraped = combine_evidence(scraped_text, state.get("scraped_content", ""), max_chars=8000)
        else:
            new_scraped = scraped_text[:8000]

    return {
        "scraped_content": new_scraped,
        "used_urls": used_urls,
    }

def writer_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    search_results = state.get("search_results", "")
    scraped_content = state.get("scraped_content", "")

    research_combined = (
        f"SEARCH RESULTS : \n {search_results} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {scraped_content}"
    )
    if state.get("attempt", 0) > 0:
        report = invoke_with_retry(DEPS.revise_chain, {
            "topic": topic,
            "research": research_combined,
            "previous_report": state.get("report", ""),
            "critique": state.get("feedback", ""),
        })
    else:
        report = invoke_with_retry(DEPS.writer_chain, {
            "topic": topic,
            "research": research_combined,
        })
    return {"report": str(report)}

def critic_node(state: ResearchState) -> dict:
    report = state.get("report", "")
    search_results = state.get("search_results", "")
    scraped_content = state.get("scraped_content", "")
    research_combined = (
        f"SEARCH RESULTS : \n {search_results} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {scraped_content}"
    )
    critic_output = invoke_with_retry(DEPS.critic_chain, {
        "report": report,
        "research": research_combined,
    })

    if isinstance(critic_output, CriticFeedback):
        score = float(critic_output.score)
        critique = critic_output.critique
        needs_more_research = bool(critic_output.needs_more_research)
    elif isinstance(critic_output, dict):
        score = float(critic_output.get("score", 0.0))
        critique = critic_output.get("critique", "")
        needs_more_research = bool(critic_output.get("needs_more_research", False))
    else:
        text = str(critic_output)
        match = re.search(r"score:\s*([\d\.]+)", text, re.IGNORECASE)
        score = float(match.group(1)) if match else None
        critique = text
        needs_more_research = False

    judge_key = state.get("judge", "llm")
    judge_fn = JUDGES.get(judge_key, llm_judge)
    judge_state = {
        **state,
        "score": score,
        "feedback": critique,
        "needs_more_research": needs_more_research,
    }
    verdict: JudgeVerdict = judge_fn(judge_state)

    retry_enabled = state.get("retry_enabled", True)
    retry_count = state.get("retry_count", 0)
    current_attempt = state.get("attempt", 0)

    decision = decide(
        retry_enabled=retry_enabled,
        accept=verdict["accept"],
        retry_count=retry_count,
        needs_more_research=needs_more_research,
    )

    attempt_record = {
        "attempt": current_attempt,
        "report": report,
        "critic_score": score,
        "critique": critique,
        "needs_more_research": needs_more_research,
        "judge_accept": verdict["accept"],
        "judge_score": verdict.get("score"),
        "judge_reason": verdict.get("reason", ""),
        "judge_details": verdict.get("details"),
        "decision": decision,
    }

    if decision != "end":
        next_retry_count = retry_count + 1
        next_attempt = current_attempt + 1
    else:
        next_retry_count = retry_count
        next_attempt = current_attempt

    return {
        "feedback": critique,
        "score": score,
        "decision": decision,
        "retry_count": next_retry_count,
        "attempt": next_attempt,
        "attempts": [attempt_record],
    }

def route_after_critic(state: ResearchState) -> str:
    """Pure routing function that returns state['decision']."""
    return state.get("decision", "end")

# Build the StateGraph
workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("reader", reader_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

# Add Edges
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "reader")
workflow.add_edge("reader", "writer")
workflow.add_edge("writer", "critic")
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {"end": END, "researcher": "researcher", "writer": "writer"}
)

# Compile Graph
graph = workflow.compile()

def run_pipeline(
    topic: str,
    mode: str = "adaptive_llm",
    retry_enabled: Optional[bool] = None,
    results_dir: Optional[str] = None,
) -> dict:
    """Public entry point for running the complete research pipeline.

    Args:
        topic: Research query topic.
        mode: Pipeline mode ('fixed', 'adaptive_llm', or 'adaptive_classifier').
        retry_enabled: Backward-compatible argument for setting mode.
        results_dir: Optional custom directory to write run log.
    """
    reset_token_usage()

    if isinstance(mode, bool):
        retry_enabled = mode
        mode = "adaptive_llm" if retry_enabled else "fixed"
    elif retry_enabled is not None:
        mode = "adaptive_llm" if retry_enabled else "fixed"

    if mode == "fixed":
        is_retry_enabled = False
        judge_name = "llm"
    elif mode == "adaptive_llm":
        is_retry_enabled = True
        judge_name = "llm"
    elif mode == "adaptive_classifier":
        is_retry_enabled = True
        judge_name = "classifier"
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'fixed', 'adaptive_llm', or 'adaptive_classifier'.")

    initial_state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "latest_search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
        "attempt": 0,
        "retry_count": 0,
        "decision": "",
        "judge": judge_name,
        "retry_enabled": is_retry_enabled,
        "used_urls": [],
        "queries_used": [],
        "attempts": [],
    }
    final_state = graph.invoke(initial_state)
    
    prompt_tok = TOKEN_USAGE["prompt_tokens"]
    comp_tok = TOKEN_USAGE["completion_tokens"]
    tot_tok = TOKEN_USAGE["total_tokens"]
    print(f"TOKENS this run: prompt={prompt_tok} completion={comp_tok} total={tot_tok}")
    final_state["token_usage"] = dict(TOKEN_USAGE)
    final_state["model_name"] = model_name

    save_run(final_state, system=mode, results_dir=results_dir)
    return final_state


def run_first_pass(topic: str) -> dict:
    """Execute the shared first pass (attempt 0): researcher -> reader -> writer -> critic LLM.

    The critic LLM chain is invoked exactly once and its critique, score, and
    needs_more_research flag are recorded in the returned state dictionary.
    """
    reset_token_usage()

    state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "latest_search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
        "attempt": 0,
        "retry_count": 0,
        "decision": "",
        "judge": "llm",
        "retry_enabled": False,
        "used_urls": [],
        "queries_used": [],
        "attempts": [],
    }

    # 1. Researcher
    res_out = researcher_node(state)
    state.update(res_out)

    # 2. Reader
    read_out = reader_node(state)
    state.update(read_out)

    # 3. Writer
    write_out = writer_node(state)
    state.update(write_out)

    # 4. Critic LLM evaluation (attempt 0)
    report = state.get("report", "")
    search_results = state.get("search_results", "")
    scraped_content = state.get("scraped_content", "")
    research_combined = (
        f"SEARCH RESULTS : \n {search_results} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {scraped_content}"
    )
    critic_output = invoke_with_retry(DEPS.critic_chain, {
        "report": report,
        "research": research_combined,
    })

    if isinstance(critic_output, CriticFeedback):
        score = float(critic_output.score)
        critique = critic_output.critique
        needs_more_research = bool(critic_output.needs_more_research)
    elif isinstance(critic_output, dict):
        score = float(critic_output.get("score", 0.0))
        critique = critic_output.get("critique", "")
        needs_more_research = bool(critic_output.get("needs_more_research", False))
    else:
        text = str(critic_output)
        match = re.search(r"score:\s*([\d\.]+)", text, re.IGNORECASE)
        score = float(match.group(1)) if match else None
        critique = text
        needs_more_research = False

    state["score"] = score
    state["feedback"] = critique
    state["needs_more_research"] = needs_more_research
    state["first_pass_token_usage"] = dict(TOKEN_USAGE)
    state["model_name"] = model_name
    return state


def run_pipeline_from_first_pass(
    first_state: dict,
    mode: str = "adaptive_llm",
    results_dir: Optional[str] = None,
    save: bool = True,
    qid: Optional[str] = None,
    filename: Optional[str] = None,
) -> dict:
    """Resume and execute pipeline from a shared attempt-0 first-pass state.

    Supports:
    - 'fixed' (System 2): no retry, final report = attempt-0 report.
    - 'adaptive_llm' (System 3): evaluates llm_judge on attempt-0; retries if rejected.
    - 'adaptive_classifier' (System 4): evaluates classifier_judge on attempt-0; retries if rejected.
    """
    import copy
    state = copy.deepcopy(first_state)
    first_pass_tokens = copy.deepcopy(first_state.get("first_pass_token_usage", {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    }))

    if mode == "fixed":
        is_retry_enabled = False
        judge_name = "llm"
    elif mode == "adaptive_llm":
        is_retry_enabled = True
        judge_name = "llm"
    elif mode == "adaptive_classifier":
        is_retry_enabled = True
        judge_name = "classifier"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    state["judge"] = judge_name
    state["retry_enabled"] = is_retry_enabled

    # Reset token usage tracker for the retry phase
    reset_token_usage()

    # Evaluate judge on attempt 0
    judge_fn = JUDGES.get(judge_name, llm_judge)
    judge_state = {
        **state,
        "score": state.get("score"),
        "feedback": state.get("feedback", ""),
        "needs_more_research": state.get("needs_more_research", False),
    }
    verdict: JudgeVerdict = judge_fn(judge_state)

    needs_more_research = state.get("needs_more_research", False)
    decision = decide(
        retry_enabled=is_retry_enabled,
        accept=verdict["accept"],
        retry_count=0,
        needs_more_research=needs_more_research,
    )

    attempt_0_record = {
        "attempt": 0,
        "report": state.get("report", ""),
        "critic_score": state.get("score"),
        "critique": state.get("feedback", ""),
        "needs_more_research": needs_more_research,
        "judge_accept": verdict["accept"],
        "judge_score": verdict.get("score"),
        "judge_reason": verdict.get("reason", ""),
        "judge_details": verdict.get("details"),
        "decision": decision,
    }
    state["attempts"] = [attempt_0_record]
    state["decision"] = decision

    if decision != "end":
        state["retry_count"] = 1
        state["attempt"] = 1
        target_node = decision

        while state["decision"] != "end" and state.get("retry_count", 0) <= MAX_RETRIES:
            if target_node == "researcher":
                res_out = researcher_node(state)
                state.update(res_out)
                read_out = reader_node(state)
                state.update(read_out)
                write_out = writer_node(state)
                state.update(write_out)
            elif target_node == "writer":
                write_out = writer_node(state)
                state.update(write_out)

            critic_out = critic_node(state)
            if "attempts" in critic_out:
                state["attempts"].extend(critic_out["attempts"])
            state.update({k: v for k, v in critic_out.items() if k != "attempts"})

            target_node = route_after_critic(state)
    else:
        state["retry_count"] = 0
        state["attempt"] = 0

    retry_tokens = dict(TOKEN_USAGE)

    state["token_usage_first_pass"] = first_pass_tokens
    state["token_usage_retries"] = retry_tokens
    state["token_usage"] = {
        "prompt_tokens": first_pass_tokens["prompt_tokens"] + retry_tokens["prompt_tokens"],
        "completion_tokens": first_pass_tokens["completion_tokens"] + retry_tokens["completion_tokens"],
        "total_tokens": first_pass_tokens["total_tokens"] + retry_tokens["total_tokens"],
    }
    if qid:
        state["qid"] = qid

    if save:
        save_run(state, system=mode, results_dir=results_dir, filename=filename, qid=qid)

    return state
