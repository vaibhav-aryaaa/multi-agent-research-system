from graph import graph, run_pipeline

def run_research_pipeline(topic: str) -> dict:
    initial_state = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
    }
    state = dict(initial_state)

    print("\n" + "=" * 50)
    print("Step 1 : search agent is working...")
    print("=" * 50)
    for output in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in output.items():
            state.update(node_state)
            if node_name == "researcher":
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
                print("\n Final Report \n", state["report"])
                print("\n" + "=" * 50)
                print("Step 4 : critic chain is reviewing the report...")
                print("=" * 50)
            elif node_name == "critic":
                print(f"\n critic score: {state.get('score')}/10")
                print("\n critic report \n", state["feedback"])

    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic :")
    run_research_pipeline(topic)