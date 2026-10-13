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
    if not env_model:
        raise RuntimeError("Set GROQ_MODEL in .env")
    return env_model

provider = os.getenv("LLM_PROVIDER", "groq")
model_name = get_groq_model()
print(f"MODEL: {model_name}")

api_key = os.getenv("GROQ_API_KEY") or "fake_key_for_offline"
llm = ChatGroq(model=model_name, api_key=api_key, temperature=0, max_retries=0)

# LEGACY: Unused in deterministic retrieval (kept for backward compatibility)
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

# LEGACY: Unused in deterministic retrieval (kept for backward compatibility)
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

# Revision Chain for Writer (LCEL)
revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Revise the previous report to address the critique, use only facts present in the research, remove or soften any claim not supported by the research, and keep the same structure (Introduction, Key Findings, Conclusion, Sources)."),
    ("human", """Revise the previous research report on the topic below to address the critique, using the provided research evidence.

Topic: {topic}

Research Gathered:
{research}

Previous Report:
{previous_report}

Critique:
{critique}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

revise_chain = revise_prompt | llm | StrOutputParser()

# Search Query Writer Chain (Retry only - plain LLM call, no tools)
query_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert academic research query specialist. Given a research topic, a reviewer critique detailing information gaps, and a list of previously used search queries, output a single refined academic search query of at most 15 words targeting the missing evidence. Output ONLY the query itself, with no quotation marks, commentary, or explanation."),
    ("human", """Topic: {topic}

Reviewer Critique:
{critique}

Previously used queries:
{queries_used}

New search query (at most 15 words):"""),
])

query_writer_chain = query_writer_prompt | llm | StrOutputParser()

from pydantic import BaseModel, Field

# Critic Output Schema
class CriticFeedback(BaseModel):
    score: float = Field(description="Numerical evaluation score from 1.0 to 10.0")
    critique: str = Field(description="Detailed critique and constructive feedback evaluating the report")
    needs_more_research: bool = Field(
        default=False,
        description="true if the main problem is missing or insufficient source evidence rather than writing quality"
    )

# Critic Chain (Structured Output via json_schema)
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp, constructive, and evidence-aware research critic. Judge whether each claim in the report is supported by the research, flag unsupported or invented claims and citations, evaluate the report with a numeric score between 1.0 and 10.0 along with detailed constructive critique, and set needs_more_research only if evidence is missing."),
    ("human", """Review the research report below against the provided research evidence and evaluate it strictly. Judge whether each claim in the report is supported by the research, flag unsupported or invented claims and citations, and set needs_more_research only if evidence is missing.

Research Evidence:
{research}

Report:
{report}"""),
])

critic_chain = critic_prompt | llm.with_structured_output(CriticFeedback, method="json_schema")