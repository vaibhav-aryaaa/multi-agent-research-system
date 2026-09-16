from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain, CriticFeedback
import re

class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    score: Optional[float]

def researcher_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    agent = build_search_agent()
    result = agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    search_results = result["messages"][-1].content
    return {"search_results": search_results}

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
    return {"report": report}

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

# Build the StateGraph
workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("reader", reader_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

# Add Edges (linear execution)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "reader")
workflow.add_edge("reader", "writer")
workflow.add_edge("writer", "critic")
workflow.add_edge("critic", END)

# Compile Graph
graph = workflow.compile()

def run_pipeline(topic: str) -> dict:
    """Public entry point for running the complete research pipeline."""
    initial_state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state
