from graph import graph, run_pipeline, QUALITY_THRESHOLD, MAX_RETRIES

def run_research_pipeline(topic: str, retry_enabled: bool = True) -> dict:
    initial_state = {
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
    state = dict(initial_state)

    mode_label = "System 3 (Adaptive Bounded Loop-Back)" if retry_enabled else "System 2 (Fixed Linear Pipeline)"
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
                if state.get("retry_count", 0) > 0:
                    print(f"\n>>> [Adaptive Loop-Back] Retry #{state['retry_count']}: Re-running researcher agent for additional evidence...")
                print("\n search result\n", state["search_results"])
                print("\n" + "=" * 50)
                print("Step 2 : reader agent is scraping top resources...")
                print("=" * 50)
            elif node_name == "reader":
                print("\nscraped content\n", state["scraped_content"])
                print("\n" + "=" * 50)
                print("Step 3 : writer is drafting the report...")
                print("=" * 50)
            elif node_name == "writer":
                if state.get("retry_count", 0) > 0 and prev_score is not None:
                    print(f"\n>>> [Adaptive Loop-Back] Retry #{state['retry_count']}: Re-drafting report with writer chain...")
                print("\n Final Report \n", state["report"])
                print("\n" + "=" * 50)
                print("Step 4 : critic chain is reviewing the report...")
                print("=" * 50)
            elif node_name == "critic":
                print(f"\n critic score: {state.get('score')}/10")
                print("\n critic report \n", state["feedback"])

    print("\n" + "=" * 50)
    print(f"Pipeline Complete. Final Score: {state.get('score')}/10 | Total Retries: {state.get('retry_count', 0)}")
    print("=" * 50)
    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)