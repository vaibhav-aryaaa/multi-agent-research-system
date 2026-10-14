"""Claim extraction and evidence passage splitting module for faithfulness evaluation.

Design Principles:
- extract_claims: Extracts atomic, factual claims from the report body, stopping before Sources.
  Parses markdown table body rows (skipping header/separator rows, dropping leading numeric # cells).
  Cleans <br> tags, asterisks, and LaTeX blocks. Protects abbreviations from premature splitting.
- split_evidence: Splits retrieved evidence into fine-grained passages (~1-3 sentences, <= 400 chars).
"""

import re
from typing import List

# Abbreviations to protect from sentence boundary detection
ABBREVIATION_PATTERNS = [
    (r"\b(Fig|Figs)\.", r"\1@@DOT@@"),
    (r"\b(Eq|Eqs)\.", r"\1@@DOT@@"),
    (r"\bet\s+al\.", r"et al@@DOT@@"),
    (r"\be\.g\.", r"e@@DOT@@g@@DOT@@"),
    (r"\bi\.e\.", r"i@@DOT@@e@@DOT@@"),
    (r"\b(vs|v)\.", r"\1@@DOT@@"),
    (r"\bNo\.", r"No@@DOT@@"),
]

def clean_report_text(text: str) -> str:
    """Clean markdown text: convert <br> to space, strip ** and *, and remove LaTeX blocks."""
    if not text:
        return ""
    # 1. Convert <br>, <br/>, <br /> to space
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    # 2. Remove LaTeX display blocks \[ ... \] and inline blocks \( ... \)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\\\([\s\S]*?\\\)", "", text)
    # 3. Remove ** and *
    text = text.replace("**", "").replace("*", "")
    # 4. Clean bullets, reference citations [1], [2], etc.
    text = text.replace("•", " ")
    text = re.sub(r"\[\d+\]", "", text)
    return text

def protect_abbreviations(text: str) -> str:
    """Replace periods in known abbreviations with temporary placeholders."""
    for pattern, replacement in ABBREVIATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def unprotect_abbreviations(text: str) -> str:
    """Restore periods in protected abbreviations."""
    return text.replace("@@DOT@@", ".")

def is_table_separator(cells: List[str]) -> bool:
    """Check if table row is a markdown separator (e.g., |---|---|)."""
    return len(cells) > 0 and all(re.match(r"^:?-+:?$", c) for c in cells if c)

def is_table_header(cells: List[str]) -> bool:
    """Check if table row is a header row (e.g., | # | Finding | Evidence & Reasoning |)."""
    if not cells:
        return False
    first_clean = re.sub(r"[*_\s]", "", cells[0]).lower()
    if first_clean in ["#", "no", "no.", "id", "item"]:
        return True
    header_keywords = {"finding", "findings", "evidence", "reasoning", "evidence & reasoning", "claim", "claims", "topic"}
    if any(c.lower() in header_keywords for c in cells):
        return True
    return False

def extract_claims(report: str) -> List[str]:
    """Extract factual claim sentences from a report body, stopping before Sources.
    
    Features:
    - Parses table body rows (skips header and separator rows, splits on |, drops leading numeric # cell).
    - Cleans <br> tags to space, strips ** and *, and removes LaTeX blocks.
    - Protects abbreviations (Fig., Eq., et al., e.g., i.e., vs., No.) from sentence splitting.
    - Excludes Sources section, headings (#, ##), and fragments under 6 words.
    """
    if not report:
        return []

    # 1. Truncate at Sources section heading
    sources_pattern = r"(?im)^\s*(?:#{1,4}\s*(?:\d+\.\s*)?Sources\b|\*{1,2}Sources:?\*{1,2}|Sources\s*:)"
    parts = re.split(sources_pattern, report, maxsplit=1)
    body = parts[0]

    # 2. Process lines and table rows
    content_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop markdown headings (#, ##, etc.)
        if stripped.startswith("#"):
            continue
        # Drop horizontal rules (---, ***, ___)
        if re.match(r"^[-*_]{3,}$", stripped):
            continue
        
        # Parse table rows
        if "|" in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip separator rows
            if is_table_separator(cells):
                continue
            # Skip header rows
            if is_table_header(cells):
                continue
            
            # Body row: drop leading numeric # cell if present
            first_clean = re.sub(r"[*_\s]", "", cells[0])
            if first_clean.isdigit():
                cells = cells[1:]
                
            for c in cells:
                cleaned_cell = clean_report_text(c).strip()
                if cleaned_cell:
                    content_lines.append(cleaned_cell)
        else:
            cleaned = clean_report_text(stripped)
            # Clean leading list bullets or numbered item markers
            cleaned = re.sub(r"^[\s*\-•>]+", "", cleaned)
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                content_lines.append(cleaned)

    # Combine into a single text block
    text_block = " ".join(content_lines)

    # 3. Protect abbreviations before splitting
    protected_text = protect_abbreviations(text_block)

    # 4. Split sentences at punctuation boundary
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", protected_text)

    # 5. Restore abbreviations and filter valid factual claims
    claims: List[str] = []
    for s in raw_sentences:
        clean_s = unprotect_abbreviations(s).strip()
        # Collapse whitespace
        clean_s = re.sub(r"\s+", " ", clean_s)

        # Drop bare URLs
        if re.match(r"^https?://\S+$", clean_s):
            continue

        # Drop short fragments (under 6 words)
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
