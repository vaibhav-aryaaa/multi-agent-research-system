import re
from typing import List, Dict

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "report", "critique", "review", "score", "reviewer",
    "found", "insufficient", "previous", "research", "please", "find"
}

def parse_search_results(text: str) -> List[Dict[str, str]]:
    """Parse output from web_search tool into structured entries.
    
    Each entry contains 'url', 'title', and 'snippet'.
    Handles results separated by '-----'.
    """
    if not text:
        return []
        
    entries: List[Dict[str, str]] = []
    blocks = [b.strip() for b in text.split("-----") if b.strip()]
    
    for block in blocks:
        url_match = re.search(r"URL:\s*([^\s\n\r]+)", block)
        title_match = re.search(r"Title:\s*([^\n\r]+)", block)
        snippet_match = re.search(r"Snippet:\s*([\s\S]+)", block)
        
        if url_match:
            url = url_match.group(1).strip()
            title = title_match.group(1).strip() if title_match else "N/A"
            snippet = snippet_match.group(1).strip() if snippet_match else ""
            entries.append({
                "url": url,
                "title": title,
                "snippet": snippet
            })
            
    # Fallback: if no blocks had 'URL: ...', search for any raw http/https links
    if not entries:
        raw_urls = re.findall(r"https?://[^\s\)\],\"'>]+", text)
        for u in raw_urls:
            entries.append({
                "url": u.strip(),
                "title": "N/A",
                "snippet": ""
            })
            
    return entries

def build_fallback_query(topic: str, critique: str) -> str:
    """Generate a fallback search query when the query LLM is unavailable.
    
    Takes the topic plus the first 8 non-stopwords from the critique.
    """
    cleaned = re.sub(r"[^\w\s]", " ", critique.lower())
    words = [w for w in cleaned.split() if w not in STOPWORDS]
    keywords = " ".join(words[:8])
    if keywords:
        return f"{topic} {keywords}".strip()
    return topic.strip()

NAV_BOILERPLATE_PATTERNS = [
    r"^skip to content",
    r"^download pdf",
    r"^order reprints",
    r"^version notes",
    r"^\*?\s*article\b",
    r"^\*?\s*open access",
    r"^author to whom correspondence",
    r"^\d+\s*pages$",
    r"^this article belongs to the special issue",
]

def clean_scraped_text(text: str) -> str:
    """Clean scraped text deterministically before character capping.
    
    Strips markdown images, link syntax, navigation/boilerplate lines,
    and collapses whitespace.
    """
    if not text:
        return ""
        
    # 1. Remove markdown images: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    
    # 2. Convert markdown links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    
    # 3. Filter lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(re.search(pat, lower) for pat in NAV_BOILERPLATE_PATTERNS):
            continue
        if stripped in ["*", "\\*", "---"]:
            continue
        cleaned_lines.append(stripped)
        
    cleaned_text = "\n\n".join(cleaned_lines)
    
    # 4. Collapse consecutive spaces / whitespace
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()

