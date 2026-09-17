import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

from groq import Groq

load_dotenv()

def get_groq_model() -> str:
    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        return env_model
    
    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    try:
        client = Groq()
        available = {m.id for m in client.models.list().data}
        for model in preferred_models:
            if model in available:
                return model
        for m in available:
            if not any(x in m for x in ["whisper", "guard", "safeguard", "orpheus"]):
                return m
    except Exception:
        pass
    return "llama-3.3-70b-versatile"

model_name = get_groq_model()
llm = ChatGroq(model=model_name, temperature=0)

# 1st Agent: Researcher (Search + Reasoning)
def build_search_agent():
    system_message = """You are a High-Fidelity Academic Researcher. 
    Your goal is to find peer-reviewed papers, formal preprints (ArXiv), and technical whitepapers for the user's research objective.
    
    GUIDELINES:
    1. PRIORITIZE sources from Nature, ArXiv, ScienceDirect, ResearchGate, and major universities.
    2. IGNORE blogs, social media (LinkedIn/Twitter), and general news unless they link to a formal study.
    3. SEARCH for formal evidence, experimental data, and technical methodologies.
    4. PROVIDE high-quality URLs and summaries of the core academic findings.
    
    If you cannot find direct papers, look for official technical documentation or institutional reports.

    CRITICAL: In your final response, you MUST list the Titles and URLs of the 
    top 3 sources you found so that the next agent can scrape them. 
    Do not just summarize; provide the data."""
    return create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=system_message
    )

# 2nd Agent: Deep Reader (Scraping + Extraction)
def build_reader_agent():
    system_message = (
        "You are an expert content extractor. Given a set of search results, "
        "you must use the scrape_url tool to visit the most relevant website, "
        "extract its core content, and provide a deep-dive summary. "
        "Always use the tool to get real data."
    )
    return create_react_agent(
        model=llm,
        tools=[scrape_url],
        prompt=system_message
    )

# Writer Chain (LCEL)
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

from pydantic import BaseModel, Field

# Critic Output Schema
class CriticFeedback(BaseModel):
    score: float = Field(description="Numerical evaluation score from 1.0 to 10.0")
    critique: str = Field(description="Detailed critique and constructive feedback evaluating the report")

# Critic Chain (Structured Output)
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific. Evaluate the report and provide a numeric score between 1.0 and 10.0 along with detailed constructive critique."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}"""),
])

critic_chain = critic_prompt | llm.with_structured_output(CriticFeedback)