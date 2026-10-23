import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents import get_groq_model
from runlog import save_run
from graph import DEPS, invoke_with_retry, reset_token_usage, get_token_usage

load_dotenv()

model_name = get_groq_model()
llm = ChatGroq(model=model_name, temperature=0, max_retries=3)

baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports based purely on your own knowledge."),
    ("human", """Write a detailed research report on the topic below using only your internal knowledge. Do not use external tools or web search.

Topic: {topic}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion

Be detailed, factual and professional."""),
])

baseline_chain = baseline_prompt | llm | StrOutputParser()
DEPS.baseline_chain = baseline_chain

def run_baseline(
    topic: str,
    results_dir: Optional[str] = None,
    qid: Optional[str] = None,
    filename: Optional[str] = None,
    save: bool = True,
) -> dict:
    """Run single-prompt baseline with no agents, tools, or retrieval (System 1)."""
    reset_token_usage()
    chain = getattr(DEPS, "baseline_chain", None) or baseline_chain
    report = invoke_with_retry(chain, {"topic": topic})
    usage = get_token_usage()
    result = {
        "topic": topic,
        "report": str(report),
        "attempts": [],
        "retry_count": 0,
        "score": None,
        "token_usage": usage,
        "token_usage_first_pass": usage,
        "token_usage_retries": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if qid:
        result["qid"] = qid
    if save:
        save_run(result, system="baseline", results_dir=results_dir, filename=filename, qid=qid)
    return result

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    res = run_baseline(topic)
    print("\n--- System 1 (No-Agent Baseline) Report ---\n")
    print(res["report"])
