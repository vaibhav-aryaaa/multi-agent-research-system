"""Claim extraction and evidence passage splitting module for faithfulness evaluation.

Design Principles:
- Segment first, then split sentences inside each segment.
- A segment is one non-table line, one list item, or one table CELL.
- Never join segments into one string before splitting. Add terminal punctuation if missing.
- Exclude formula statements (LaTeX) as they cannot be validated against text evidence.
- Exclude meta-statements about the report itself and questions.
- Maintain abbreviation protection and 6-word minimum.
"""

import re
from typing import List, Tuple, Dict, Any

# Abbreviations to protect from sentence boundary detection
ABBREVIATION_PATTERNS = [
    (r"\b(Fig|Figs)\.", r"\1@@DOT@@"),
    (r"\b(Eq|Eqs)\.", r"\1@@DOT@@"),
    (r"\bet\s+al\.", r"et al@@DOT@@"),
    (r"\be\.g\.", r"e@@DOT@@g@@DOT@@"),
    (r"\bi\.e\.", r"i@@DOT@@e@@DOT@@"),
    (r"\b(vs|v)\.", r"\1@@DOT@@"),
    (r"\bNo\.", r"No@@DOT@@"),
    (r"\b(Ref|Refs)\.", r"\1@@DOT@@"),
]

# Non-claim prefixes: sentences starting with these describe the report/aim rather than facts
NON_CLAIM_PREFIXES = (
    "this report",
    "this revision",
    "the present report",
    "the present revision",
    "the aim",
    "the purpose",
    "overall,",
    "in conclusion",
    "in summary",
    "topic:",
    "research report",
)

# Non-claim section titles and bare labels
NON_CLAIM_EXACT = {
    "synthesis of findings",
    "executive summary",
    "key findings",
    "research report",
    "introduction",
    "background",
    "conclusion",
    "methodology",
}

# Header words used to identify and skip markdown table header rows
SHORT_HEADER_WORDS = {
    "#", "no", "no.", "id", "item", "finding", "findings",
    "evidence", "reasoning", "evidence & reasoning", "evidence and reasoning",
    "claim", "claims", "topic", "description", "source", "sources"
}

# Function words or incomplete clause endings (Rule b)
TRAILING_INCOMPLETE_WORDS = {
    "is", "are", "of", "to", "by", "the", "and", "roughly", "approximately", "about",
    "was", "were", "be", "been", "being", "with", "that", "than", "as", "at", "in", "for", "from", "or"
}

# Comprehensive vocabulary for verb detection heuristic (Rule e)
VERB_WORDS = {
    # Auxiliaries & Modals
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # Academic & common verbs
    "use", "uses", "used", "using", "make", "makes", "made", "making",
    "take", "takes", "took", "taken", "taking", "find", "finds", "found", "finding",
    "give", "gives", "gave", "given", "giving", "show", "shows", "showed", "shown", "showing",
    "lead", "leads", "led", "leading", "cause", "causes", "caused", "causing",
    "hold", "holds", "held", "holding", "set", "sets", "setting", "see", "sees", "saw", "seen", "seeing",
    "know", "knows", "knew", "known", "knowing", "get", "gets", "got", "gotten", "getting",
    "deliver", "delivers", "delivered", "delivering", "collect", "collects", "collected", "collecting",
    "correspond", "corresponds", "corresponded", "corresponding",
    "contribute", "contributes", "contributed", "contributing",
    "amplify", "amplifies", "amplified", "amplifying",
    "retain", "retains", "retained", "retaining", "exhibit", "exhibits", "exhibited", "exhibiting",
    "propose", "proposes", "proposed", "proposing", "disrupt", "disrupts", "disrupted", "disrupting",
    "impose", "imposes", "imposed", "imposing", "continue", "continues", "continued", "continuing",
    "destabilize", "destabilizes", "destabilized", "destabilizing", "enforce", "enforces", "enforced", "enforcing",
    "affect", "affects", "affected", "affecting", "dominate", "dominates", "dominated", "dominating",
    "require", "requires", "required", "requiring", "produce", "produces", "produced", "producing",
    "extract", "extracts", "extracted", "extracting", "generate", "generates", "generated", "generating",
    "preclude", "precludes", "precluded", "precluding", "consume", "consumes", "consumed", "consuming",
    "allow", "allows", "allowed", "allowing", "serve", "serves", "served", "serving",
    "draw", "draws", "drew", "drawn", "drawing", "pull", "pulls", "pulled", "pulling",
    "push", "pushes", "pushed", "pushing", "state", "states", "stated", "stating",
    "note", "notes", "noted", "noting", "report", "reports", "reported", "reporting",
    "measure", "measures", "measured", "measuring", "observe", "observes", "observed", "observing",
    "conclude", "concludes", "concluded", "concluding", "classify", "classifies", "classified", "classifying",
    "compare", "compares", "compared", "comparing", "reproduce", "reproduces", "reproduced", "reproducing",
    "identify", "identifies", "identified", "identifying", "exploit", "exploits", "exploited", "exploiting",
    "focus", "focuses", "focused", "focusing", "address", "addresses", "addressed", "addressing",
    "stand", "stands", "stood", "standing", "lie", "lies", "lay", "lain", "lying",
    "rely", "relies", "relied", "relying", "need", "needs", "needed", "needing",
    "improve", "improves", "improved", "improving", "achieve", "achieves", "achieved", "achieving",
    "alter", "alters", "altered", "altering", "harvest", "harvests", "harvested", "harvesting",
    "convert", "converts", "converted", "converting", "supply", "supplies", "supplied", "supplying",
    "reopen", "reopens", "reopened", "reopening", "obey", "obeys", "obeyed", "obeying",
    "leave", "leaves", "left", "leaving", "introduce", "introduces", "introduced", "introducing",
    "exist", "exists", "existed", "existing", "depend", "depends", "depended", "depending",
    "interact", "interacts", "interacted", "interacting", "bind", "binds", "bound", "binding",
    "inhibit", "inhibits", "inhibited", "inhibiting", "clear", "clears", "cleared", "clearing",
    "degrade", "degrades", "degraded", "degrading", "emit", "emits", "emitted", "emitting",
    "store", "stores", "stored", "storing", "drive", "drives", "drove", "driven", "driving",
    "switch", "switches", "switched", "switching", "tap", "taps", "tapped", "tapping",
    "act", "acts", "acted", "acting", "lose", "loses", "lost", "losing",
    "gain", "gains", "gained", "gaining", "link", "links", "linked", "linking",
    "correlate", "correlates", "correlated", "correlating", "prevent", "prevents", "prevented", "preventing",
    "detect", "detects", "detected", "detecting", "synthesize", "synthesizes", "synthesized", "synthesizing",
    "mediate", "mediates", "mediated", "mediating", "accumulate", "accumulates", "accumulated", "accumulating",
    "prune", "prunes", "pruned", "pruning", "heal", "heals", "healed", "healing",
    "suppress", "suppresses", "suppressed", "suppressing", "edit", "edits", "edited", "editing",
    "target", "targets", "targeted", "targeting", "enhance", "enhances", "enhanced", "enhancing",
    "form", "forms", "formed", "forming", "stabilize", "stabilizes", "stabilized", "stabilizing",
    "accelerate", "accelerates", "accelerated", "accelerating", "capture", "captures", "captured", "capturing",
    "compress", "compresses", "compressed", "compressing", "transfer", "transfers", "transferred", "transferring",
    "ingest", "ingests", "ingested", "ingesting", "engulf", "engulfs", "engulfed", "engulfing",
    "undergo", "undergoes", "underwent", "undergone", "undergoing", "govern", "governs", "governed", "governing",
    "modulate", "modulates", "modulated", "modulating", "mitigate", "mitigates", "mitigated", "mitigating",
    "induce", "induces", "induced", "inducing", "attenuate", "attenuates", "attenuated", "attenuating",
    "express", "expresses", "expressed", "expressing", "catalyze", "catalyzes", "catalyzed", "catalyzing",
    "infiltrate", "infiltrates", "infiltrated", "infiltrating", "possess", "possesses", "possessed", "possessing",
    "exert", "exerts", "exerted", "exerting", "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "indicate", "indicates", "indicated", "indicating", "suggest", "suggests", "suggested", "suggesting",
    "confirm", "confirms", "confirmed", "confirming", "derive", "derives", "derived", "deriving",
    "lower", "lowers", "lowered", "lowering", "raise", "raises", "raised", "raising",
    "overcome", "overcomes", "overcame", "overcoming", "provide", "provides", "provided", "providing",
    "operate", "operates", "operated", "operating", "verify", "verifies", "verified", "verifying",
    "omit", "omits", "omitted", "omitting", "yield", "yields", "yielded", "yielding",
    "utilize", "utilizes", "utilized", "utilizing", "decrease", "decreases", "decreased", "decreasing",
    "enable", "enables", "enabled", "enabling"
}

NON_VERBS = {
    "speed", "red", "bed", "feed", "seed", "need", "during", "thing", "spring", "ring", "wing", "king",
    "something", "anything", "nothing", "everything", "ceiling", "morning", "evening", "warning", "building"
}


def has_verb(text: str) -> bool:
    """Heuristic to check if a sentence/label contains at least one verb."""
    # Strip parentheticals first so parenthetical citation verbs do not count
    clean_no_parens = text
    while "(" in clean_no_parens and ")" in clean_no_parens:
        stripped = re.sub(r"\([^()]*\)", "", clean_no_parens)
        if stripped == clean_no_parens:
            break
        clean_no_parens = stripped

    tokens = [re.sub(r"^[^\w]+|[^\w]+$", "", w).lower() for w in clean_no_parens.split()]
    tokens = [t for t in tokens if t]
    for t in tokens:
        if t in VERB_WORDS:
            return True
        if len(t) >= 5 and (t.endswith("ed") or t.endswith("ing") or t.endswith("izes") or t.endswith("ated") or t.endswith("ates")):
            if t not in NON_VERBS:
                return True
    return False


def protect_abbreviations(text: str) -> str:
    """Replace periods in known abbreviations with temporary placeholders."""
    for pattern, replacement in ABBREVIATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def unprotect_abbreviations(text: str) -> str:
    """Restore periods in protected abbreviations."""
    return text.replace("@@DOT@@", ".")


def clean_text(text: str) -> str:
    """Clean markdown text: strip bold/italic, reference brackets, and leading bullets."""
    if not text:
        return ""
    t = text.replace("**", "").replace("*", "").replace("__", "")
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\[[A-Za-z0-9\s]+\]", "", t)  # e.g., [MDPI 2019]
    t = re.sub(r"^[\s*\-•>]+", "", t)
    t = re.sub(r"^\d+[\.\)]\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


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
    # If all non-empty cells are known short header keywords
    cleaned_cells = [re.sub(r"[*_\s]", "", c).lower() for c in cells if c.strip()]
    if cleaned_cells and all(c in SHORT_HEADER_WORDS for c in cleaned_cells):
        return True
    return False


def extract_claims_with_stats(report: str) -> Tuple[List[str], Dict[str, Any]]:
    """Extract factual claim sentences with detailed statistics on dropped items.
    
    Processing Steps:
    1. Truncate at Sources section.
    2. Segment first: each segment is one non-table line, list item, or table cell.
       Never join segments before splitting sentences.
    3. Ensure terminal punctuation for each segment so boundaries are clean.
    4. Detect LaTeX formulas (display and inline). Sentences containing formulas
       cannot be validated against text evidence, so they are dropped and counted.
    5. Drop non-claims: questions, meta-statements about the report, and section headers.
    6. Filter by word count (>= 6 words) and drop bare URLs.
    
    Returns:
    - claims: List of clean factual claim strings.
    - stats: Dict with dropped_latex, dropped_non_claims, and total_extracted counts.
    """
    if not report:
        return [], {"dropped_latex": 0, "dropped_non_claims": 0, "total_extracted": 0}

    stats = {"dropped_latex": 0, "dropped_non_claims": 0}

    # 1. Truncate at Sources section
    sources_pattern = r"(?im)^\s*(?:#{1,4}\s*(?:\d+\.\s*)?Sources\b|\*{1,2}Sources:?\*{1,2}|Sources\s*:)"
    parts = re.split(sources_pattern, report, maxsplit=1)
    body = parts[0]

    # Mask multiline display math and inline formulas before segment extraction
    latex_pattern = re.compile(r"\\\[[\s\S]*?\\\]|\\\(.*?\\\)|\$\$[\s\S]*?\$\$")
    body = latex_pattern.sub(" @@LATEX_FORMULA@@ ", body)

    # 2. Segment extraction: non-table lines, list items, and table cells
    segments: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if re.match(r"^[-*_]{3,}$", stripped):
            continue

        if "|" in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if is_table_separator(cells):
                continue
            if is_table_header(cells):
                continue

            # Body row: drop leading numeric # cell if present
            first_clean = re.sub(r"[*_\s]", "", cells[0])
            if first_clean.isdigit():
                cells = cells[1:]

            for cell in cells:
                # Split cell by <br> or newline to treat list items as separate segments
                cell_parts = re.split(r"(?i)<br\s*/?>|\n", cell)
                for part in cell_parts:
                    p_clean = clean_text(part)
                    if p_clean:
                        segments.append(part.strip())
        else:
            # Non-table line: split by <br> into sub-segments
            line_parts = re.split(r"(?i)<br\s*/?>", stripped)
            for part in line_parts:
                p_clean = clean_text(part)
                if p_clean:
                    segments.append(part.strip())

    claims: List[str] = []

    # 3. Process each segment independently
    for seg_idx, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue

        # If this segment is just a masked formula placeholder, count and skip
        if seg == "@@LATEX_FORMULA@@" or "@@LATEX_FORMULA@@" in seg:
            stats["dropped_latex"] += 1
            continue

        # Check if this segment directly preceded a masked formula segment (incomplete lead-in)
        next_is_formula = (seg_idx + 1 < len(segments) and segments[seg_idx + 1] == "@@LATEX_FORMULA@@")

        # Add terminal punctuation if missing or ending with colon/semicolon
        seg_ends_with_colon = bool(re.search(r"[:;]\s*$", seg))
        seg = re.sub(r"[:;]\s*$", "", seg)
        clean_punct = re.sub(r"[*_\s]+$", "", seg)
        if not re.search(r"[.!?]$", clean_punct):
            seg = seg + "."

        # Protect abbreviations from false sentence breaks
        prot_seg = protect_abbreviations(seg)

        # Split sentences within segment
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", prot_seg)

        for s_idx, s in enumerate(raw_sentences):
            clean_s = unprotect_abbreviations(s).strip()
            clean_s = clean_text(clean_s)
            clean_s = re.sub(r"\.\.+$", ".", clean_s)

            # Drop questions
            if clean_s.endswith("?"):
                stats["dropped_non_claims"] += 1
                continue

            lower_s = clean_s.lower()

            # Drop meta statements about the report/aim
            if any(lower_s.startswith(prefix) for prefix in NON_CLAIM_PREFIXES):
                stats["dropped_non_claims"] += 1
                continue

            # Drop bare section labels
            if lower_s.rstrip(".:") in NON_CLAIM_EXACT:
                stats["dropped_non_claims"] += 1
                continue

            # Drop bare URLs
            if re.match(r"^https?://\S+$", clean_s):
                continue

            # --- Rule (a): LaTeX commands and formula delimiters ---
            # Drop any claim containing a backslash followed by letters, \[, \(, or $
            if re.search(r"\\[a-zA-Z]", clean_s) or re.search(r"\\\[|\\\(", clean_s) or "$" in clean_s:
                stats["dropped_latex"] += 1
                continue

            # --- Rule (b): Incomplete lead-ins & trailing function words ---
            # Drop if the last word before the final period is a function word or "roughly"
            last_word_match = re.search(r"(\b[a-zA-Z]+)\b[^\w]*$", clean_s)
            if last_word_match and last_word_match.group(1).lower() in TRAILING_INCOMPLETE_WORDS:
                stats["dropped_non_claims"] += 1
                continue
            # Drop if this sentence was an incomplete clause directly preceding a masked equation
            if next_is_formula and s_idx == len(raw_sentences) - 1 and seg_ends_with_colon:
                stats["dropped_non_claims"] += 1
                continue

            # --- Rule (c): Lowercase start or citation scraps ---
            stripped_start = re.sub(r"^[\s\"'“‘\(\[\{<]+", "", clean_s)
            if stripped_start and stripped_start[0].islower():
                stats["dropped_non_claims"] += 1
                continue
            if clean_s.strip().startswith("(") and (clean_s.strip().endswith(").") or clean_s.strip().endswith(")")):
                stats["dropped_non_claims"] += 1
                continue

            # --- Rule (d): Under 7 words ---
            words = [w for w in clean_s.split() if re.search(r"\w", w)]
            if len(words) < 7:
                stats["dropped_non_claims"] += 1
                continue

            # --- Rule (e): Bullet labels / fragments that are noun phrases with no verb ---
            if not has_verb(clean_s):
                stats["dropped_non_claims"] += 1
                continue

            claims.append(clean_s)

    stats["total_extracted"] = len(claims)
    return claims, stats


def extract_claims(report: str) -> List[str]:
    """Extract factual claim sentences from a report body, stopping before Sources.
    
    Thin wrapper over extract_claims_with_stats returning only the claims list.
    """
    claims, _ = extract_claims_with_stats(report)
    return claims


def split_evidence(search_results: str, scraped_content: str) -> List[str]:
    """Split raw search results and scraped content into fine-grained passages.
    
    Passages are roughly 1-3 sentences and <= 400 characters, suitable for
    sentence-level or passage-level NLI entailment scoring.
    """
    combined_raw: List[str] = []
    if search_results:
        combined_raw.append(search_results)
    if scraped_content:
        combined_raw.append(scraped_content)

    if not combined_raw:
        return []

    passages: List[str] = []
    for block in combined_raw:
        sentences = re.split(r"(?<=[.!?])\s+", block.strip())
        current_chunk: List[str] = []
        current_len = 0

        for sent in sentences:
            s_clean = sent.strip()
            if not s_clean:
                continue
            if len(s_clean) < 20:
                continue
            if s_clean.startswith("http://") or s_clean.startswith("https://"):
                continue

            if current_len + len(s_clean) > 400 and current_chunk:
                passages.append(" ".join(current_chunk))
                current_chunk = [s_clean]
                current_len = len(s_clean)
            else:
                current_chunk.append(s_clean)
                current_len += len(s_clean)

        if current_chunk:
            passages.append(" ".join(current_chunk))

    return passages
