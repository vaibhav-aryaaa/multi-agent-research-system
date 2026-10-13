"""Claim extraction and evidence passage splitting module for faithfulness evaluation.

Design Principles:
- extract_claims: Extracts atomic, factual claims from the report body, strictly
  excluding metadata, markdown headers, tables, and the Sources bibliography.
- Sentence splitting uses a deterministic regex splitter documenting punctuation boundaries
  to ensure 100% reproducible, offline execution without external corpus downloads.
- split_evidence: Splits retrieved evidence into fine-grained passages (~1-3 sentences, <= 400 chars).
"""

import re
from typing import List

def extract_claims(report: str) -> List[str]:
    """Extract factual claim sentences from a report body, stopping before Sources.
    
    Excludes:
    - Sources section and bibliography
    - Markdown headings (#, ##, etc.) and horizontal rules (---)
    - Table formatting lines (|)
    - Short fragments (< 6 words)
    - Bare URLs
    """
    if not report:
        return []

    # 1. Truncate at Sources section heading
    sources_pattern = r"(?im)^\s*(?:#{1,4}\s*(?:\d+\.\s*)?Sources\b|\*{1,2}Sources:?\*{1,2}|Sources\s*:)"
    parts = re.split(sources_pattern, report, maxsplit=1)
    body = parts[0]

    # 2. Process line by line to filter out structural markdown
    content_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop markdown headings
        if stripped.startswith("#"):
            continue
        # Drop horizontal rules
        if re.match(r"^[-*_]{3,}$", stripped):
            continue
        # Drop markdown table borders and table rows
        if stripped.startswith("|") or "|---" in stripped:
            continue
        # Clean leading list bullets, numbered lists, or bold markers
        cleaned_line = re.sub(r"^[\s*\-•>]+", "", stripped)
        cleaned_line = re.sub(r"^\d+[\.\)]\s*", "", cleaned_line)
        cleaned_line = re.sub(r"^\[\d+\]\s*", "", cleaned_line)
        if cleaned_line:
            content_lines.append(cleaned_line)

    text_to_split = " ".join(content_lines)

    # 3. Sentence splitter: regex based on punctuation followed by capital/quote
    # Documented choice: avoids heavy NLTK punkt download while reliably handling abbreviations
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", text_to_split)

    claims: List[str] = []
    for s in raw_sentences:
        clean_s = s.strip()
        # Clean markdown bold/italic tags
        clean_s = clean_s.replace("**", "").replace("__", "").strip()
        # Strip trailing reference brackets like [1], [2], [MDPI 2019]
        clean_s = re.sub(r"\[\d+\]", "", clean_s).strip()
        
        # Drop bare URLs
        if re.match(r"^https?://\S+$", clean_s):
            continue
        
        # Word count check: drop very short fragments (under 6 words)
        words = [w for w in clean_s.split() if re.search(r"\w", w)]
        if len(words) < 6:
            continue

        claims.append(clean_s)

    return claims

def split_evidence(search_results: str, scraped_content: str) -> List[str]:
    """Split search and scraped evidence into passages of 1 to 3 sentences (capped at ~400 chars).
    
    Drops empty and URL-only passages.
    """
    combined_raw = []
    if search_results:
        # Split search results by entry separator
        entries = search_results.split("-----")
        for e in entries:
            for line in e.splitlines():
                line_str = line.strip()
                if line_str.startswith("Snippet:"):
                    combined_raw.append(line_str[len("Snippet:"):].strip())
                elif line_str.startswith("Title:") or line_str.startswith("URL:"):
                    continue
                elif line_str:
                    combined_raw.append(line_str)
                    
    if scraped_content:
        for p in scraped_content.split("\n\n"):
            p_str = p.strip()
            if p_str and not p_str.startswith("http"):
                combined_raw.append(p_str)

    passages: List[str] = []
    for block in combined_raw:
        # Split block into individual sentences
        sentences = re.split(r"(?<=[.!?])\s+", block.strip())
        current_chunk: List[str] = []
        current_len = 0
        
        for sent in sentences:
            s_clean = sent.strip()
            if not s_clean or re.match(r"^https?://\S+$", s_clean):
                continue
            
            if current_chunk and (current_len + len(s_clean) > 400 or len(current_chunk) >= 3):
                passage_text = " ".join(current_chunk)
                if len(passage_text.split()) >= 6:
                    passages.append(passage_text[:400].strip())
                current_chunk = [s_clean]
                current_len = len(s_clean)
            else:
                current_chunk.append(s_clean)
                current_len += len(s_clean)
                
        if current_chunk:
            passage_text = " ".join(current_chunk)
            if len(passage_text.split()) >= 6:
                passages.append(passage_text[:400].strip())

    return passages
