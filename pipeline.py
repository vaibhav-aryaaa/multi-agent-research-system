from typing import Optional
from graph import graph, run_pipeline, QUALITY_THRESHOLD, MAX_RETRIES
from runlog import save_run

def run_research_pipeline(
    topic: str,
    mode: str = "adaptive_llm",
    retry_enabled: Optional[bool] = None,
) -> dict:
    if isinstance(mode, bool):
        retry_enabled = mode
        mode = "adaptive_llm" if retry_enabled else "fixed"
    elif retry_enabled is not None:
        mode = "adaptive_llm" if retry_enabled else "fixed"

    if mode == "fixed":
        is_retry_enabled = False
        judge_name = "llm"
        mode_label = "System 2 (Fixed Linear Pipeline)"
    elif mode == "adaptive_llm":
        is_retry_enabled = True
        judge_name = "llm"
        mode_label = "System 3 (Adaptive Bounded Loop-Back - LLM Judge)"
    elif mode == "adaptive_classifier":
        is_retry_enabled = True
        judge_name = "classifier"
        mode_label = "System 4 (Adaptive Bounded Loop-Back - Classifier Judge)"
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'fixed', 'adaptive_llm', or 'adaptive_classifier'.")

    initial_state = {
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
    state = dict(initial_state)

    print("\n" + "=" * 50)
    print(f"Running Research Pipeline: {mode_label}")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("Step 1 : search agent is working...")
    print("=" * 50)

    for output in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in output.items():
            prev_score = state.get("score")
            state.update(node_state)
            if node_name == "researcher":
                if state.get("attempt", 0) > 0:
                    print(f"\n>>> [Adaptive Loop-Back] Retry #{state.get('retry_count', 0)}: Re-running researcher agent for additional evidence...")
                print("\n search result\n", state.get("search_results", ""))
                print("\n" + "=" * 50)
                print("Step 2 : reader agent is scraping top resources...")
                print("=" * 50)
            elif node_name == "reader":
                print("\nscraped content\n", state.get("scraped_content", ""))
                print("\n" + "=" * 50)
                print("Step 3 : writer is drafting the report...")
                print("=" * 50)
            elif node_name == "writer":
                if state.get("attempt", 0) > 0 and prev_score is not None:
                    print(f"\n>>> [Adaptive Loop-Back] Retry #{state.get('retry_count', 0)}: Re-drafting report with revise chain...")
                print("\n Final Report \n", state.get("report", ""))
                print("\n" + "=" * 50)
                print("Step 4 : critic chain is reviewing the report...")
                print("=" * 50)
            elif node_name == "critic":
                print(f"\n critic score: {state.get('score')}/10")
                print("\n critic report \n", state.get("feedback", ""))
                print(f" critic decision: {state.get('decision')}")

    print("\n" + "=" * 50)
    print(f"Pipeline Complete. Final Score: {state.get('score')}/10 | Total Retries: {state.get('retry_count', 0)}")
    print("=" * 50)

    save_run(state, system=mode)
    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)