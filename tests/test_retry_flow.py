import os
import sys
import copy
import json
import tempfile

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_MODEL", "test-pinned-model")
os.environ.setdefault("GROQ_API_KEY", "fake_key_for_test")

from agents import CriticFeedback
from graph import (
    DEPS,
    run_pipeline,
    route_after_critic,
    MAX_RETRIES,
)
from retrieval import parse_search_results, build_fallback_query, clean_scraped_text

class FakeWebSearch:
    def __init__(self):
        self.queries = []
    def invoke(self, inputs, config=None):
        query = inputs["query"] if isinstance(inputs, dict) else str(inputs)
        self.queries.append(query)
        if len(self.queries) == 1:
            return (
                "Title: Initial Paper 1\nURL: https://arxiv.org/url1\nSnippet: initial evidence 1\n"
                "-----\n"
                "Title: Initial Paper 2\nURL: https://arxiv.org/url2\nSnippet: initial evidence 2\n"
            )
        else:
            return (
                "Title: Retry Paper\nURL: https://arxiv.org/retry_url\nSnippet: retry evidence\n"
            )

class FakeScraper:
    def __init__(self, fail_first_n=0):
        self.scraped_urls = []
        self.fail_first_n = fail_first_n
        self.calls = 0
    def invoke(self, inputs, config=None):
        url = inputs["url"] if isinstance(inputs, dict) else str(inputs)
        self.scraped_urls.append(url)
        self.calls += 1
        if self.calls <= self.fail_first_n:
            return "Extraction failed: No content found at this URL."
        return (
            f"Detailed scraped content for {url} with rich technical facts, methodology, "
            f"and experimental results demonstrating high-fidelity academic findings. "
            f"This content provides deep domain insights and detailed citations that comprehensively "
            f"address all research gaps identified by the reviewers and evaluators during the pipeline run."
        )

class FakeQueryWriter:
    def __init__(self, should_raise=False):
        self.invoked_inputs = []
        self.should_raise = should_raise
    def invoke(self, inputs, config=None):
        self.invoked_inputs.append(inputs)
        if self.should_raise:
            raise RuntimeError("Simulated query writer failure")
        return f"refined academic search query for {inputs.get('topic')}"

class FakeWriter:
    def __init__(self):
        self.invoked_inputs = []
    def invoke(self, inputs, config=None):
        self.invoked_inputs.append(inputs)
        return "REPORT v1"

class FakeReviser:
    def __init__(self):
        self.invoked_inputs = []
    def invoke(self, inputs, config=None):
        self.invoked_inputs.append(inputs)
        critique = inputs.get("critique", "")
        return "REPORT v2 (revised: " + critique + ")"

class FakeCritic:
    def __init__(self, scores, needs_more_research_list):
        self.scores = list(scores)
        self.needs_more_research_list = list(needs_more_research_list)
        self.calls = 0
        self.invoked_inputs = []
    def invoke(self, inputs, config=None):
        self.invoked_inputs.append(inputs)
        idx = min(self.calls, len(self.scores) - 1)
        score = self.scores[idx]
        nmr = self.needs_more_research_list[min(self.calls, len(self.needs_more_research_list) - 1)]
        self.calls += 1
        return CriticFeedback(
            score=score,
            critique=f"Critique #{self.calls} for score {score}",
            needs_more_research=nmr
        )

def setup_fakes(critic_scores, needs_more_research_list, should_raise_query_writer=False, fail_first_n_scrapes=0):
    searcher = FakeWebSearch()
    scraper = FakeScraper(fail_first_n=fail_first_n_scrapes)
    q_writer = FakeQueryWriter(should_raise=should_raise_query_writer)
    writer = FakeWriter()
    reviser = FakeReviser()
    critic = FakeCritic(critic_scores, needs_more_research_list)

    DEPS.web_search = searcher
    DEPS.scrape_url = scraper
    DEPS.query_writer_chain = q_writer
    DEPS.writer_chain = writer
    DEPS.revise_chain = reviser
    DEPS.critic_chain = critic

    return searcher, scraper, q_writer, writer, reviser, critic

def test_all_scenarios():
    print("=" * 70)
    print("Running Fake LLM Deterministic Retrieval & Retry Verification Tests")
    print("=" * 70)

    # Unit Test: parse_search_results extracts correctly
    sample_raw = (
        "Title: Quantum Hall Effect\nURL: https://arxiv.org/abs/2101.00001\nSnippet: Sample discovery.\n"
        "-----\n"
        "Title: Casimir Dynamics\nURL: https://nature.com/articles/casimir-123\nSnippet: Vacuum force measurements.\n"
    )
    parsed = parse_search_results(sample_raw)
    assert len(parsed) == 2, f"Expected 2 parsed results, got {len(parsed)}"
    assert parsed[0]["url"] == "https://arxiv.org/abs/2101.00001"
    assert parsed[0]["title"] == "Quantum Hall Effect"
    assert "Sample discovery." in parsed[0]["snippet"]
    assert parsed[1]["url"] == "https://nature.com/articles/casimir-123"
    print("[PASS] Unit Test: parse_search_results correctly extracts URL, title, and snippet")

    # Unit Test: clean_scraped_text on MDPI sample
    mdpi_sample = (
        "[Skip to Content](#main)\n\n"
        "[![MDPI](https://mdpi-res.com/logo.svg)](/)\n\n"
        "[Download PDF\n PDF](/article.pdf)\n\n"
        "* Article\n"
        "* ![Open Access](https://mdpi-res.com/open-access.svg)\n\n"
        "23 May 2019\n\n"
        "18 Pages\n\n"
        "# Extraction of Zero-Point Energy from the Vacuum\n\n"
        "Department of Electrical Engineering, University of Colorado\n\n"
        "Author to whom correspondence should be addressed.\n\n"
        "[Atoms](/journal/atoms)\n"
        "**2019**, *7*(2), 51;\n\n"
        "Order Reprints\n\n"
        "## Abstract\n\n"
        "In research articles and patents several methods have been proposed for zero-point energy extraction."
    )
    cleaned_mdpi = clean_scraped_text(mdpi_sample)
    assert "![MDPI]" not in cleaned_mdpi
    assert "![Open Access]" not in cleaned_mdpi
    assert "[Skip to Content]" not in cleaned_mdpi
    assert "Skip to Content" not in cleaned_mdpi
    assert "Download PDF" not in cleaned_mdpi
    assert "Order Reprints" not in cleaned_mdpi
    assert "Extraction of Zero-Point Energy from the Vacuum" in cleaned_mdpi
    assert "In research articles and patents several methods have been proposed" in cleaned_mdpi
    print("[PASS] Unit Test: clean_scraped_text strips markdown images, links, and boilerplate on MDPI sample")

    # Unit Test: Narrowed article boilerplate regex does not strip content lines
    test_article_text = (
        "* Article\n"
        "Article 12 discusses Casimir forces in microcavities.\n"
        "Articles published in Nature describe similar findings."
    )
    cleaned_article = clean_scraped_text(test_article_text)
    assert "* Article" not in cleaned_article.split("\n")
    assert "Article 12 discusses Casimir forces in microcavities." in cleaned_article
    assert "Articles published in Nature describe similar findings." in cleaned_article
    print("[PASS] Unit Test: Narrowed article pattern strips exact boilerplate line while preserving content lines")

    # Unit Test: ChatGroq max_retries is 0 so rate-limit waiting is always visible
    from agents import llm
    assert llm.max_retries == 0, f"Expected llm.max_retries == 0, got {llm.max_retries}"
    print("[PASS] Unit Test: ChatGroq max_retries is 0")

    # Unit Test: extract_claims extracts factual sentences and excludes headings & Sources section
    from faithfulness.claims import extract_claims
    sample_report = (
        "# Research Report: Quantum Hall Effect\n\n"
        "## Introduction\n"
        "The Quantum Hall Effect is an extraordinary phenomenon observed in two-dimensional electron systems at low temperatures.\n"
        "Strong magnetic fields lead to quantized Hall conductance.\n\n"
        "## Key Findings\n"
        "* Graphene exhibits an anomalous integer quantum Hall effect at room temperature.\n"
        "* Fractional charges emerge from strongly correlated electron states in high magnetic fields.\n"
        "* Edge states remain topologically protected against backscattering from non-magnetic impurities.\n\n"
        "### Sources\n"
        "1. Klitzing et al., Physical Review Letters, 1980. https://doi.org/10.1103/PhysRevLett.45.494\n"
        "2. Novoselov et al., Nature, 2005. https://nature.com/articles/nature04233\n"
    )
    claims = extract_claims(sample_report)
    assert len(claims) == 5, f"Expected 5 claims, got {len(claims)}: {claims}"
    assert "The Quantum Hall Effect is an extraordinary phenomenon observed in two-dimensional electron systems at low temperatures." in claims
    assert "Strong magnetic fields lead to quantized Hall conductance." in claims
    assert "Graphene exhibits an anomalous integer quantum Hall effect at room temperature." in claims
    assert "Fractional charges emerge from strongly correlated electron states in high magnetic fields." in claims
    assert "Edge states remain topologically protected against backscattering from non-magnetic impurities." in claims
    # Ensure headings and sources are excluded
    for c in claims:
        assert not c.startswith("#")
        assert "Klitzing" not in c
        assert "Novoselov" not in c
    print("[PASS] Unit Test: extract_claims extracts clean factual sentences, excluding headings and Sources")

    # Unit Test: Markdown table body parsing with leading # cell dropped, headers excluded, and target claim extracted
    table_sample = (
        "# Research Report: Energy Extraction\n\n"
        "### Key Findings\n\n"
        "| # | Finding | Evidence & Reasoning |\n"
        "|---|---------|----------------------|\n"
        "| **1** | **Mechanical cycles violate conservation laws** | Forward proposed extracting energy until plates touch. A reset requires external work. |\n"
        "| **2** | **Switchable-mirror limits** | The energy cost of the transition has been measured in thin-film VO₂ and graphene devices and is ≥ 10⁻¹⁸ J per µm², which exceeds the Casimir energy gain for plate separations > 10 nm. <br>• Even if an ideal switch existed, net work is zero. |\n\n"
        "### Sources\n"
        "1. MDPI 2019 review.\n"
    )
    table_claims = extract_claims(table_sample)
    # Header cells must NOT be extracted
    assert not any(c == "Finding" or c == "Evidence & Reasoning" for c in table_claims), f"Header cells should not be extracted: {table_claims}"
    assert not any("#" in c for c in table_claims)
    # Target claim with >= 10^-18 J per um^2 IS extracted
    target_claims = [c for c in table_claims if "≥ 10⁻¹⁸ J per µm²" in c]
    assert len(target_claims) == 1, f"Expected target sentence with ≥ 10⁻¹⁸ J per µm² to be extracted, got: {table_claims}"
    assert "exceeds the Casimir energy gain" in target_claims[0]
    print("[PASS] Unit Test: Markdown table body extracted (≥ 10⁻¹⁸ J per µm² sentence included, headers excluded)")

    # Unit Test: Abbreviation protection (Fig., Eq., et al., e.g., i.e., vs., No.)
    abbr_sample = (
        "According to Pinto et al., the Casimir force follows Eq. 4 as demonstrated in Fig. 2. "
        "The experimental setup (i.e., gold mirrors vs. aluminum mirrors, e.g., in cavity No. 3) confirms zero net work."
    )
    abbr_claims = extract_claims(abbr_sample)
    assert len(abbr_claims) == 2, f"Expected 2 sentences, got {len(abbr_claims)}: {abbr_claims}"
    assert "According to Pinto et al., the Casimir force follows Eq. 4 as demonstrated in Fig. 2." in abbr_claims
    assert "The experimental setup (i.e., gold mirrors vs. aluminum mirrors, e.g., in cavity No. 3) confirms zero net work." in abbr_claims
    print("[PASS] Unit Test: Abbreviation protection preserves (Fig., Eq., et al., e.g., i.e., vs., No.) without premature splitting")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Scenario 1: fixed mode, critic score 4.0
        # exactly one pass, retry_count == 0, exactly 1 record in attempts.
        # First-pass search query is exactly topic, zero LLM calls in retrieval.
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes([4.0], [False])
        out1 = run_pipeline("Test Topic 1", mode="fixed", results_dir=tmpdir)
        assert out1["retry_count"] == 0, f"Expected retry_count 0, got {out1['retry_count']}"
        assert len(out1["attempts"]) == 1, f"Expected 1 attempt, got {len(out1['attempts'])}"
        assert out1["attempts"][0]["critic_score"] == 4.0
        assert out1["attempts"][0]["decision"] == "end"
        assert len(wr.invoked_inputs) == 1
        assert len(rev.invoked_inputs) == 0
        assert len(searcher.queries) == 1
        assert searcher.queries[0] == "Test Topic 1", f"First pass search query must equal topic, got {searcher.queries[0]}"
        assert len(q_writer.invoked_inputs) == 0, "Query writer must NOT be called on pass 0"
        # Evidence-aware critic checks:
        assert len(cr.invoked_inputs) == 1
        assert "research" in cr.invoked_inputs[0] and "report" in cr.invoked_inputs[0]
        assert "SEARCH RESULTS :" in cr.invoked_inputs[0]["research"]
        print("[PASS] Scenario 1: fixed mode on low score does exactly 1 pass (critic received research evidence)")

        # Scenario 2: adaptive_llm, score 4.0 then 8.0, needs_more_research=False
        # goes to the writer, retry_count == 1, final report differs from first,
        # reviser received the critique text, 2 attempts records, first-pass writer prompt unchanged
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes([4.0, 8.0], [False, False])
        out2 = run_pipeline("Test Topic 2", mode="adaptive_llm", results_dir=tmpdir)
        assert out2["retry_count"] == 1, f"Expected retry_count 1, got {out2['retry_count']}"
        assert len(out2["attempts"]) == 2, f"Expected 2 attempts, got {len(out2['attempts'])}"
        assert out2["attempts"][0]["decision"] == "writer"
        assert out2["attempts"][1]["decision"] == "end"
        assert out2["report"] != out2["attempts"][0]["report"], "Report must differ between passes"
        assert len(rev.invoked_inputs) == 1
        assert rev.invoked_inputs[0]["critique"] == out2["attempts"][0]["critique"]
        assert len(wr.invoked_inputs) == 1
        assert len(searcher.queries) == 1, "Researcher must not be re-invoked when needs_more_research=False"
        print("[PASS] Scenario 2: adaptive_llm writer retry passes critique to reviser and updates report")

        # Scenario 3: adaptive_llm, score 4.0 then 8.0, needs_more_research=True
        # goes to researcher retry:
        # - query_writer received critique text
        # - retry search query differs from first-pass query
        # - reader picks new URL and never scrapes already-used URL
        # - evidence accumulated (new first, old after) and capped at 8000 chars
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes([4.0, 8.0], [True, False])
        out3 = run_pipeline("Test Topic 3", mode="adaptive_llm", results_dir=tmpdir)
        assert out3["retry_count"] == 1
        assert len(out3["attempts"]) == 2
        assert out3["attempts"][0]["decision"] == "researcher"
        assert len(searcher.queries) == 2
        assert searcher.queries[0] == "Test Topic 3"
        assert searcher.queries[1] != searcher.queries[0], "Retry search query must differ from first pass"
        assert len(q_writer.invoked_inputs) == 1
        assert "Critique #1 for score 4.0" in q_writer.invoked_inputs[0]["critique"]
        # used_urls grows and scraper never re-scrapes an existing URL:
        assert len(out3["used_urls"]) == 2
        assert "https://arxiv.org/url1" in out3["used_urls"]
        assert "https://arxiv.org/retry_url" in out3["used_urls"]
        assert len(scraper.scraped_urls) == 2
        assert len(set(scraper.scraped_urls)) == 2, "Reader must never scrape an already-used URL"
        # Evidence accumulation: new first
        assert out3["search_results"].startswith("Title: Retry Paper")
        print("[PASS] Scenario 3: adaptive_llm researcher retry: query differs, reader avoids used URLs, evidence accumulates")

        # Scenario 4: adaptive_llm, critic always scores 3.0
        # stops after exactly MAX_RETRIES (3) retries, so 4 attempts total, then ends
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes(
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [False, False, False, False, False]
        )
        out4 = run_pipeline("Test Topic 4", mode="adaptive_llm", results_dir=tmpdir)
        assert out4["retry_count"] == MAX_RETRIES, f"Expected retry_count {MAX_RETRIES}, got {out4['retry_count']}"
        assert len(out4["attempts"]) == 4, f"Expected 4 attempts, got {len(out4['attempts'])}"
        assert out4["attempts"][-1]["decision"] == "end"
        print(f"[PASS] Scenario 4: adaptive_llm caps at MAX_RETRIES ({MAX_RETRIES}) with 4 total attempts")

        # Scenario 5: adaptive_llm, critic score 9.0 first time
        # no retry
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes([9.0], [False])
        out5 = run_pipeline("Test Topic 5", mode="adaptive_llm", results_dir=tmpdir)
        assert out5["retry_count"] == 0
        assert len(out5["attempts"]) == 1
        assert out5["attempts"][0]["decision"] == "end"
        assert len(wr.invoked_inputs) == 1
        assert len(rev.invoked_inputs) == 0
        print("[PASS] Scenario 5: adaptive_llm accepts high score (9.0) immediately with 0 retries")

        # Scenario 6: route_after_critic does not mutate its input state
        fake_state = {
            "decision": "researcher",
            "retry_count": 2,
            "attempt": 2,
            "score": 4.5,
            "feedback": "needs work",
            "attempts": [{"test": 1}],
        }
        state_before = copy.deepcopy(fake_state)
        route_result = route_after_critic(fake_state)
        assert route_result == "researcher"
        assert fake_state == state_before, "route_after_critic mutated input state!"
        print("[PASS] Scenario 6: route_after_critic is pure and does not mutate input state")

        # Scenario 7: A results/*.json file is written with all required fields in temp dir
        json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert len(json_files) >= 5, f"Expected at least 5 log files in {tmpdir}, found {len(json_files)}"
        sample_path = os.path.join(tmpdir, json_files[0])
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required_fields = [
            "run_id", "system", "topic", "model_name",
            "final_report", "contexts", "retry_count",
            "quality_threshold", "queries_used", "used_urls",
            "final_score", "attempts"
        ]
        for field in required_fields:
            assert field in data, f"Missing field '{field}' in logged JSON"
        assert isinstance(data["attempts"], list)
        if data["attempts"]:
            att0 = data["attempts"][0]
            att_fields = [
                "attempt", "report", "critic_score", "critique",
                "needs_more_research", "judge_accept", "judge_score",
                "judge_reason", "decision"
            ]
            for af in att_fields:
                assert af in att0, f"Missing attempt field '{af}' in attempt record"
        print("[PASS] Scenario 7: Run logs correctly written to JSON with all required fields in temp dir")

        # Scenario 8: Query writer raises exception -> fallback query is used seamlessly
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes(
            [4.0, 8.0], [True, False], should_raise_query_writer=True
        )
        out8 = run_pipeline("Casimir Cavity", mode="adaptive_llm", results_dir=tmpdir)
        assert out8["retry_count"] == 1
        assert len(searcher.queries) == 2
        # Second query should use fallback query containing topic and critique keywords:
        retry_query = searcher.queries[1]
        assert "casimir cavity" in retry_query.lower()
        print(f"[PASS] Scenario 8: Fallback query used seamlessly when query writer fails ('{retry_query}')")

        # Scenario 9: Scraper retry fallback when top URL returns error
        searcher, scraper, q_writer, wr, rev, cr = setup_fakes(
            [8.0], [False], fail_first_n_scrapes=1
        )
        out9 = run_pipeline("Test Fallback Scrape", mode="fixed", results_dir=tmpdir)
        # Should have attempted first URL (failed), then tried second URL (succeeded):
        assert len(scraper.scraped_urls) == 2
        assert scraper.scraped_urls[0] == "https://arxiv.org/url1"
        assert scraper.scraped_urls[1] == "https://arxiv.org/url2"
        assert "Detailed scraped content for https://arxiv.org/url2" in out9["scraped_content"]
        print("[PASS] Scenario 9: Reader fallback on failed scrape tries next URL successfully")

    print("=" * 70)
    print("ALL 9 SCENARIOS PASSED WITH ZERO API CALLS & ZERO TOKEN BURN!")
    print("=" * 70)

if __name__ == "__main__":
    test_all_scenarios()
