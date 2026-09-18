import os
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from rich import print

load_dotenv()

def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    return TavilyClient(api_key=api_key)

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information. Returns Titles, URLs, and high-quality snippets."""
    ACADEMIC_DOMAINS = [
        "arxiv.org", "nature.com", "sciencedirect.com", 
        "researchgate.net", "scholar.google.com", "jstor.org",
        "ieee.org", "science.org", "frontiersin.org", "mdpi.com"
    ]
    
    last_error = None
    for attempt in range(3):
        try:
            client = get_tavily_client()
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_domains=ACADEMIC_DOMAINS
            )
            
            out = []
            for r in results.get('results', []):
                snippet = (r.get('content') or '')[:600]
                out.append(
                    f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {snippet}\n"
                )
            if out:
                return "\n-----\n".join(out)
            
            # If domain filtering returned no results, retry with standard search fallback
            fallback = client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            for r in fallback.get('results', []):
                snippet = (r.get('content') or '')[:600]
                out.append(
                    f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {snippet}\n"
                )
            return "\n-----\n".join(out) if out else "No relevant search results found."
        except Exception as e:
            last_error = e
            time.sleep(0.5 * (attempt + 1))
            
    return f"Search query failed: {str(last_error)}. Please try a different query."

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL. Use this for deep-dives into a specific source."""
    last_error = None
    for attempt in range(3):
        try:
            client = get_tavily_client()
            extraction = client.extract(urls=[url])
            if extraction and extraction.get('results'):
                content = extraction['results'][0].get('raw_content', '')
                if not content:
                    content = extraction['results'][0].get('content', '')
                return content[:2500]
            return "Extraction failed: No content found at this URL."
        except Exception as e:
            last_error = e
            time.sleep(0.5 * (attempt + 1))
            
    return f"Failed to scrape URL: {str(last_error)}. Please try a different source."
