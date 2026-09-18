from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain, CriticFeedback
import re

QUALITY_THRESHOLD = 7.0
MAX_RETRIES = 3

class ResearchState(TypedDict, total=False):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    score: Optional[float]
    retry_count: int
    retry_enabled: bool
    _retrying_search: bool

def researcher_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    agent = build_search_agent()
    result = agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    search_results = result["messages"][-1].content
    rc = state.get("retry_count", 0)
    if state.get("score") is not None:
        rc += 1
    return {"search_results": search_results, "retry_count": rc, "_retrying_search": True}

def reader_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    search_results = state.get("search_results", "")
    agent = build_reader_agent()
    result = agent.invoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{search_results[:800]}"
        )]
    })
    scraped_content = result["messages"][-1].content
    return {"scraped_content": scraped_content}

def writer_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    search_results = state.get("search_results", "")
    scraped_content = state.get("scraped_content", "")
    
    research_combined = (
        f"SEARCH RESULTS : \n {search_results} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {scraped_content}"
    )
    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    rc = state.get("retry_count", 0)
    # If routed directly from critic to writer (researcher was skipped), increment retry_count
    if state.get("score") is not None and not state.get("_retrying_search", False):
        rc += 1
    return {"report": report, "retry_count": rc, "_retrying_search": False}

def critic_node(state: ResearchState) -> dict:
    report = state.get("report", "")
    critic_output = critic_chain.invoke({"report": report})
    
    if isinstance(critic_output, CriticFeedback):
        score = float(critic_output.score)
        feedback = critic_output.critique
    elif isinstance(critic_output, dict):
        score = float(critic_output.get("score", 0.0))
        feedback = critic_output.get("critique", "")
    else:
        text = str(critic_output)
        match = re.search(r"score:\s*([\d\.]+)", text, re.IGNORECASE)
        score = float(match.group(1)) if match else None
        feedback = text

    return {"feedback": feedback, "score": score}

def route_after_critic(state: ResearchState) -> str:
    """Decide next graph transition after critic review:
    1. If retry_enabled is False, immediately return 'end' (System 2 fixed pipeline).
    2. If score is None (unparseable), score >= QUALITY_THRESHOLD (7.0), or retry_count >= MAX_RETRIES (3), return 'end'.
    3. Otherwise, check feedback for signals of missing/insufficient source material.
       If detected ('insufficient sources', 'lacks evidence', 'no citations', 'unsupported'),
       route back to 'researcher' for more literature data.
       Otherwise, route back to 'writer' to redraft using existing material.
    """
    if not state.get("retry_enabled", True):
        return "end"

    score = state.get("score")
    retry_count = state.get("retry_count", 0)

    if score is None or score >= QUALITY_THRESHOLD or retry_count >= MAX_RETRIES:
        return "end"

    # Heuristic: inspect critic feedback for indications that new research/evidence is needed
    feedback_lower = (state.get("feedback") or "").lower()
    insufficient_signals = ["insufficient sources", "lacks evidence", "no citations", "unsupported"]
    if any(sig in feedback_lower for sig in insufficient_signals):
        return "researcher"
    return "writer"

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

def run_pipeline(topic: str, retry_enabled: bool = True) -> dict:
    """Public entry point for running the complete research pipeline.
    
    Args:
        topic: Research query topic.
        retry_enabled: When False, runs System 2 (fixed sequential pass).
                       When True, runs System 3 (adaptive bounded loop-back).
    """
    initial_state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
        "retry_count": 0,
        "retry_enabled": retry_enabled,
        "_retrying_search": False,
    }
    final_state = graph.invoke(initial_state)
    return final_state
