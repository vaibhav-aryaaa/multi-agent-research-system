from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def _canned_stream(initial_state, stream_mode="updates"):
    """Yield simulated node updates for a complete pipeline execution."""
    yield {"researcher": {"search_results": "Evidence chunk 1\n-----\nEvidence chunk 2 <script>alert(1)</script>"}}
    yield {"reader": {"scraped_content": "Extracted text summary <script>alert(1)</script>"}}
    yield {"writer": {"report": "# Intelligence Report\n\nKey findings on topic with <script>alert(1)</script>."}}
    yield {
        "critic": {
            "score": 8.0,
            "feedback": "Analysis is solid.",
            "decision": "end",
            "attempts": [
                {"attempt": 1, "critic_score": 8.0, "decision": "end"}
            ],
        }
    }


def test_app_initial_load():
    """Verify app loads cleanly in idle state with System selector present."""
    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    assert not at.exception
    
    # Assert system selector exists with the expected options
    system_boxes = [s for s in at.selectbox if s.label == "System"]
    assert len(system_boxes) == 1
    options = system_boxes[0].options
    assert "Adaptive, LLM judge (S3)" in options
    assert "Fixed pipeline (S2)" in options
    assert "Adaptive, classifier judge (S4)" in options
    assert system_boxes[0].value == "Adaptive, LLM judge (S3)"


def test_app_smoke_run_and_safe_rendering(tmp_path):
    """Test full pipeline run with canned stream, verifying report, attempts panel, and safe rendering."""
    with patch("graph.graph.stream", side_effect=_canned_stream), \
         patch("runlog.save_run") as mock_save:
        
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        assert not at.exception
        
        # Enter topic and submit
        at.text_input[0].input("Quantum Computing Advancements").run()
        at.button[0].click().run()
        assert not at.exception
        
        # Verify execution finished
        assert at.session_state.stage == "finished"
        mock_save.assert_called_once()
        assert mock_save.call_args[1].get("system") == "adaptive_llm"
        
        # Verify report was rendered
        markdown_values = [m.value for m in at.markdown]
        assert any("Intelligence Report" in val for val in markdown_values)
        
        # Verify attempts panel is present
        assert len(at.dataframe) >= 1
        assert at.session_state.results.get("retry_count") == 0
        assert len(at.session_state.results.get("attempts", [])) == 1
        
        # Verify XSS / safe rendering: <script>alert(1)</script> never in unsafe HTML
        for m in at.markdown:
            is_unsafe = getattr(m, "is_unsafe_allow_html", getattr(m, "unsafe_allow_html", False))
            if is_unsafe:
                assert "<script>alert(1)</script>" not in m.value, f"Raw script injected in unsafe HTML: {m.value}"


def test_app_fixed_pipeline_selection():
    """Verify selecting Fixed pipeline (S2) passes system='fixed' to save_run."""
    with patch("graph.graph.stream", side_effect=_canned_stream), \
         patch("runlog.save_run") as mock_save:
        
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        
        # Select Fixed pipeline
        at.selectbox[0].select("Fixed pipeline (S2)").run()
        assert not at.exception
        
        at.text_input[0].input("Microbiome Therapeutics").run()
        at.button[0].click().run()
        assert not at.exception
        
        mock_save.assert_called_once()
        assert mock_save.call_args[1].get("system") == "fixed"


def test_app_classifier_judge_failure_handled_cleanly():
    """Verify that if cross-encoder fails to load, S4 handles it with a note without crashing."""
    with patch("faithfulness.cross_encoder.load_cross_encoder", side_effect=RuntimeError("Model weights unavailable")):
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        
        # Switch to S4 which will invoke load_cross_encoder inside try/except
        at.selectbox[0].select("Adaptive, classifier judge (S4)").run()
        assert not at.exception
        
        # Verify warning was shown
        assert any("unavailable" in w.value.lower() for w in at.warning)
