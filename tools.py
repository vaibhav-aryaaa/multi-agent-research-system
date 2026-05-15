from requests import request
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information. Returns Titles, URLs, and high-quality snippets."""
    results = tavily.search(query=query, search_depth="advanced", max_results=5)
    
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n"
        )
    return "\n-----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL. Use this for deep-dives into a specific source."""
    try:
        extraction = tavily.extract(urls=[url])
        if extraction and extraction['results']:
            content = extraction['results'][0].get('raw_content', '')
            if not content:
                 content = extraction['results'][0].get('content', '')
            return content[:5000]
        return "Extraction failed: No content found at this URL."
    except Exception as e:
        return f"Failed to scrape URL: {str(e)}. Please try a different source."
