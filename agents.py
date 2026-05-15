import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 1st Agent: Researcher (Search + Reasoning)
def build_search_agent():
    system_message = (
        "You are a professional web researcher. Your goal is to find the most "
        "accurate, recent, and detailed information on a topic. Use the search tool "
        "to gather data. \n\n"
        "CRITICAL: In your final response, you MUST list the Titles and URLs of the "
        "top 3 sources you found so that the next agent can scrape them. "
        "Do not just summarize; provide the data."
    )
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

# Critic Chain (LCEL)
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Critique: [your feedback]

..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()