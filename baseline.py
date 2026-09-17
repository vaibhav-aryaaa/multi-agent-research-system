import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents import get_groq_model

load_dotenv()

model_name = get_groq_model()
llm = ChatGroq(model=model_name, temperature=0)

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

def run_baseline(topic: str) -> dict:
    """Run single-prompt baseline with no agents, tools, or retrieval (System 1)."""
    report = baseline_chain.invoke({"topic": topic})
    return {
        "topic": topic,
        "report": report
    }

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    res = run_baseline(topic)
    print("\n--- System 1 (No-Agent Baseline) Report ---\n")
    print(res["report"])
