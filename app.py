from typing import Optional
import os
import sys
import time
import random
import traceback
import streamlit as st

# 1. Ensure application directory is in sys.path for Streamlit Cloud
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# 2. Bridge Streamlit Cloud secrets to os.environ so backend modules receive them
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            if isinstance(_v, str) and _k not in os.environ:
                os.environ[_k] = _v
except Exception:
    pass

# Ensure GROQ_MODEL has a sensible default
if not os.getenv("GROQ_MODEL"):
    os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="AgentFlow · Multi-Agent Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 3. Defensive imports with clean user-facing diagnostics
try:
    from graph import graph
    from runlog import save_run
except Exception as exc:
    st.error(f"### Application Initialization Error: `{type(exc).__name__}`")
    st.error(f"**Error Details:** {exc}")
    st.markdown("---")
    st.markdown("**Full Traceback:**")
    st.code(traceback.format_exc(), language="python")
    st.info("💡 **Tip for Streamlit Cloud**: Ensure `GROQ_API_KEY` and `TAVILY_API_KEY` are configured in your Streamlit Cloud app settings under **Secrets**.")
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;700&display=swap');

/* ── Reset & base ── */
:root {
    --text-color: #2d241e !important;
    --background-color: #fdfcf0 !important;
    --secondary-background-color: #fffef5 !important;
}

html, body, .stApp, [class*="css"], [data-testid="stAppViewContainer"], .block-container {
    font-family: 'Lora', serif;
    color: #2d241e !important;
    background-color: #fdfcf0 !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #2d241e !important;
}

.stApp {
    background-color: #fdfcf0;
    background-image: 
        radial-gradient(at 0% 0%, rgba(91, 33, 182, 0.03) 0, transparent 50%), 
        radial-gradient(at 100% 100%, rgba(91, 33, 182, 0.05) 0, transparent 50%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 4rem 5rem; max-width: 1300px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 4rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: #5b21b6;
    margin-bottom: 1.2rem;
    opacity: 0.8;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(4rem, 9vw, 6rem);
    font-weight: 900;
    line-height: 1.0;
    color: #1e1b4b;
    margin: 0 auto 1.5rem;
    letter-spacing: -0.02em;
    text-align: center;
}
.hero h1 a {
    display: none !important;
}
.hero h1 span {
    color: #5b21b6;
    font-style: italic;
}
.hero-sub {
    font-family: 'Lora', serif;
    font-size: 1.35rem;
    font-weight: 400;
    color: #4b4540;
    max-width: 850px;
    margin: 0 auto;
    font-style: italic;
    line-height: 1.7;
    opacity: 0.9;
    text-align: center;
    display: block;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(91, 33, 182, 0.15), transparent);
    margin: 2rem 0 4rem;
}

/* ── Streamlit Selectbox overrides ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #fffef5 !important;
    border: 2px solid #ddd6fe !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
}
[data-testid="stSelectbox"] span {
    color: #1e1b4b !important;
}
[data-testid="stSelectbox"] label {
    color: #4c1d95 !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}

/* ── Streamlit Input overrides ── */
.stTextInput > div > div > input {
    background-color: #fffef5 !important;
    border: 2px solid #ddd6fe !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
    font-family: 'Lora', serif !important;
    font-size: 1.15rem !important;
    padding: 0.9rem 1.4rem !important;
    box-shadow: 0 4px 12px rgba(91, 33, 182, 0.03) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5b21b6 !important;
    box-shadow: 0 0 0 4px rgba(91, 33, 182, 0.08) !important;
    background-color: #ffffff !important;
}
.stTextInput label {
    color: #4c1d95 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.8rem !important;
    letter-spacing: 0.05em;
}

/* ── Primary Button ── */
.stButton > button {
    background: #5b21b6 !important;
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.2rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(91, 33, 182, 0.15) !important;
}
.stButton > button:hover {
    background: #4c1d95 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(91, 33, 182, 0.25) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: #5b21b6 !important;
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 0 4px 15px rgba(91, 33, 182, 0.15) !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #4c1d95 !important;
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(91, 33, 182, 0.25) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(91, 33, 182, 0.04) !important;
}
[data-testid="stMetricLabel"] * {
    color: #6b7280 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] * {
    color: #1e1b4b !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(91, 33, 182, 0.03) !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary * {
    color: #4c1d95 !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
    color: #374151 !important;
    font-family: 'Lora', serif !important;
}

/* ── Container Cards (Report & Feedback) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 20px !important;
    padding: 2.5rem 3rem !important;
    box-shadow: 0 6px 30px rgba(91, 33, 182, 0.04) !important;
    margin-top: 1rem !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] * {
    color: #2d241e !important;
}

/* ── Pipeline Step Cards ── */
.step-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.step-card.active {
    border-color: #5b21b6;
    background: #f5f3ff;
    box-shadow: 0 4px 20px rgba(91, 33, 182, 0.06);
}
.step-card.done {
    border-color: #10b981;
    background: #f0fdf4;
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: transparent;
}
.step-card.active::before { background: #5b21b6; }
.step-card.done::before   { background: #10b981; }

.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    letter-spacing: 0.15em;
    font-weight: 500;
}
.step-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-top: 0.2rem;
}
.status-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-waiting { color: #9ca3af; background: #f3f4f6; }
.badge-running { color: #5b21b6; background: #ede9fe; }
.badge-done    { color: #059669; background: #d1fae5; }

/* ── Results ── */
.report-container {
    background: #ffffff;
    border-radius: 24px;
    padding: 4.5rem;
    border: 1px solid #e5e7eb;
    margin-top: 3rem;
    box-shadow: 0 10px 50px rgba(0,0,0,0.03);
}
.report-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #5b21b6;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #ede9fe;
    padding-bottom: 1.2rem;
}
.report-content {
    color: #2d241e;
    line-height: 2.0;
    font-size: 1.15rem;
}
.feedback-container {
    background: #f5f3ff;
    border-left: 6px solid #5b21b6;
    padding: 3rem;
    margin-top: 3rem;
    border-radius: 0 16px 16px 0;
}

.try-chip {
    background: rgba(91, 33, 182, 0.06);
    border: 1px solid rgba(91, 33, 182, 0.12);
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #5b21b6;
    font-family: 'DM Sans', sans-serif;
    display: inline-block;
    margin: 0.3rem;
    cursor: default;
    transition: background 0.2s;
}
.try-chip:hover {
    background: rgba(91, 33, 182, 0.1);
}

.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 8rem;
    letter-spacing: 0.15em;
}
</style>
""", unsafe_allow_html=True)

def render_step(num: str, title: str, state: str, desc: str):
    labels = {"waiting": "Waiting", "running": "Running", "done": "Completed"}
    status_cls = f"badge-{state}"
    card_cls = "active" if state == "running" else ("done" if state == "done" else "")
    
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <span class="step-num">STAGE // {num}</span>
                <div class="step-title">{title}</div>
                <div style="font-size:0.9rem; color:#6b7280; margin-top:0.4rem;">{desc}</div>
            </div>
            <span class="status-badge {status_cls}">{labels[state]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = {}
if "stage" not in st.session_state:
    st.session_state.stage = "idle"

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI Intelligence</div>
    <h1>Agent<span>Flow</span></h1>
    <p class="hero-sub">
        Collaborative intelligence at work. Four specialized agents cross-reference, 
        scrape, and synthesize web data into a refined research manuscript.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Handle Suggestion Click (Before Widget Instantiation) ─────────────────────
if "pending_suggestion" in st.session_state:
    st.session_state.topic_field = st.session_state.pending_suggestion
    st.session_state.stage = "running"
    del st.session_state.pending_suggestion

# ── Placeholder & Suggestion Persistence ──────────────────────────────────────
PLACEHOLDERS = [
    "e.g. Advancements in CRISPR gene editing for 2025",
    "e.g. The impact of DeepSeek-V3 on the LLM ecosystem",
    "e.g. Future of Solid-State Batteries in EVs",
    "e.g. Quantum Computing breakthroughs in financial modeling",
    "e.g. Ethical implications of NeuroLink technology",
    "e.g. Vertical farming vs traditional agriculture in 2030",
    "e.g. The role of AI in drug discovery for rare diseases"
]

SUGGESTIONS = [
    "DeepSeek-V3 Architecture", "Sora's Impact on Video Production", 
    "Liquid AI World Models", "Grok-3 Training Clusters", 
    "Gemini 2.0 Agentic Capabilities", "Agentic Workflows in Enterprise",
    "Solid-State Battery Breakthroughs", "Quantum Supremacy 2025",
    "Mars Colonization Logistics", "Vertical Farming Scalability"
]

if "active_placeholder" not in st.session_state:
    st.session_state.active_placeholder = random.choice(PLACEHOLDERS)
if "active_suggestions" not in st.session_state:
    st.session_state.active_suggestions = random.sample(SUGGESTIONS, 3)

col_input, col_spacer, col_pipeline = st.columns([5, 0.6, 4])

with col_input:
    SYSTEM_OPTIONS = [
        "Adaptive, LLM judge (S3)",
        "Fixed pipeline (S2)",
        "Adaptive, classifier judge (S4)",
    ]

    system_choice = st.selectbox(
        "System",
        SYSTEM_OPTIONS,
        index=0,
        help="Select pipeline architecture: Fixed pipeline (S2), Adaptive with LLM judge (S3), or Adaptive with classifier judge (S4)."
    )

    s4_disabled = False
    if system_choice == "Adaptive, classifier judge (S4)":
        try:
            from faithfulness.cross_encoder import load_cross_encoder
            load_cross_encoder()
        except Exception as e:
            s4_disabled = True
            st.warning(f"⚠️ Classifier judge (S4) unavailable ({e}). Please select Fixed pipeline (S2) or Adaptive, LLM judge (S3).")

    topic_input = st.text_input(
        "RESEARCH OBJECTIVE",
        placeholder=st.session_state.active_placeholder,
        key="topic_field"
    )
    if st.button("Synthesize Data"):
        if s4_disabled:
            st.error("Classifier judge (S4) cannot be loaded in this environment. Please switch to Fixed pipeline (S2) or Adaptive, LLM judge (S3).")
        elif topic_input:
            st.session_state.selected_system = system_choice
            st.session_state.results = {}
            st.session_state.stage = "running"
            st.rerun()
        else:
            st.warning("Please define a research objective.")

    missing_keys = []
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "fake_key_for_offline":
        missing_keys.append("GROQ_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")
    if missing_keys:
        st.warning(f"⚠️ Missing API key(s): {', '.join(missing_keys)}. Configure them in Streamlit Cloud Secrets (or `.env`) to run live queries.")
    
    # Interactive Suggestion Chips
    st.markdown('<div class="suggestion-chip-container" style="margin-top:2rem; display:flex; align-items:center; flex-wrap:wrap; gap: 0.8rem;">', unsafe_allow_html=True)
    st.markdown('<span style="font-family:\'DM Mono\', monospace; font-size:0.8rem; color:#9ca3af; letter-spacing:0.1em; margin-right: 0.5rem;">SUGGESTIONS:</span>', unsafe_allow_html=True)
    
    # Target suggestion chips specifically
    st.markdown("""
        <style>
        .suggestion-chip-container div[data-testid="stButton"] button {
            background: #ede9fe !important;
            border: 1.5px solid #c4b5fd !important;
            border-radius: 12px !important;
            padding: 0.5rem 1.2rem !important;
            font-size: 0.88rem !important;
            color: #5b21b6 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            transition: all 0.2s !important;
            width: auto !important;
            height: auto !important;
            box-shadow: 0 2px 6px rgba(91, 33, 182, 0.08) !important;
        }
        .suggestion-chip-container div[data-testid="stButton"] button:hover {
            background: #5b21b6 !important;
            color: #ffffff !important;
            border-color: #5b21b6 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(91, 33, 182, 0.25) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    for suggestion in st.session_state.active_suggestions:
        if st.button(suggestion, key=f"btn_{suggestion}"):
            st.session_state.pending_suggestion = suggestion
            st.session_state.selected_system = system_choice
            st.session_state.results = {}
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div style="font-family:\'Playfair Display\', serif; font-size:1.6rem; font-weight:700; color:#1e1b4b; margin-bottom:1.8rem; letter-spacing:0.02em;">Pipeline Workflow</div>', unsafe_allow_html=True)
    
    step1_ph = st.empty()
    step2_ph = st.empty()
    step3_ph = st.empty()
    step4_ph = st.empty()

    def update_steps(active_step: str, retry_text: Optional[str] = None):
        step4_desc = retry_text if retry_text else "Evaluating for bias and accuracy"
        with step1_ph:
            render_step("01", "Researcher", "running" if active_step == "researcher" else ("done" if active_step in ["reader", "writer", "critic", "revising", "finished"] else "waiting"), "Scanning high-authority web sources")
        with step2_ph:
            render_step("02", "Deep Reader", "running" if active_step == "reader" else ("done" if active_step in ["writer", "critic", "revising", "finished"] else "waiting"), "Parsing & extracting structured content")
        with step3_ph:
            render_step("03", "Synthesis", "running" if active_step == "writer" else ("done" if active_step in ["critic", "revising", "finished"] else "waiting"), "Writing the final report")
        with step4_ph:
            render_step("04", "Quality Assurance", "running" if active_step in ["critic", "revising"] else ("done" if active_step == "finished" else "waiting"), step4_desc)

if st.session_state.stage == "idle":
    update_steps("idle")

elif st.session_state.stage == "running":
    topic = st.session_state.topic_field
    update_steps("researcher")
    
    selected_system = st.session_state.get("selected_system", "Adaptive, LLM judge (S3)")
    if selected_system == "Fixed pipeline (S2)":
        selected_mode = "fixed"
        judge_name = "llm"
        retry_enabled = False
    elif selected_system == "Adaptive, classifier judge (S4)":
        selected_mode = "adaptive_classifier"
        judge_name = "classifier"
        retry_enabled = True
    else:
        selected_mode = "adaptive_llm"
        judge_name = "llm"
        retry_enabled = True

    initial_state = {
        "topic": topic,
        "search_results": "",
        "latest_search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "score": None,
        "attempt": 0,
        "retry_count": 0,
        "decision": "",
        "judge": judge_name,
        "retry_enabled": retry_enabled,
        "used_urls": [],
        "queries_used": [],
        "attempts": [],
    }
    
    current_state = dict(initial_state)
    
    try:
        for step_output in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in step_output.items():
                if "attempts" in node_state:
                    current_state.setdefault("attempts", []).extend(node_state["attempts"])
                for k, v in node_state.items():
                    if k != "attempts":
                        current_state[k] = v
                
                if node_name == "researcher":
                    update_steps("reader")
                elif node_name == "reader":
                    update_steps("writer")
                elif node_name == "writer":
                    update_steps("critic")
                elif node_name == "critic":
                    decision = node_state.get("decision", "end")
                    if decision != "end":
                        retry_n = current_state.get("retry_count", 1)
                        update_steps("revising", retry_text=f"Revising (retry {retry_n} of 3)")

        update_steps("finished")
        save_run(current_state, system=selected_mode)
        critic_display = f"Score: {current_state.get('score')}/10\n\nCritique: {current_state.get('feedback', '')}" if current_state.get("score") is not None else current_state.get("feedback", "")
        
        st.session_state.results = {
            "system": selected_mode,
            "system_label": selected_system,
            "search": current_state.get("search_results", ""),
            "reader": current_state.get("scraped_content", ""),
            "writer": current_state.get("report", ""),
            "critic": critic_display,
            "score": current_state.get("score"),
            "feedback": current_state.get("feedback", ""),
            "retry_count": current_state.get("retry_count", 0),
            "attempts": current_state.get("attempts", []),
            "token_usage": current_state.get("token_usage"),
        }
        st.session_state.stage = "finished"
        st.rerun()
    except Exception as e:
        st.session_state.stage = "idle"
        st.error(f"Pipeline error: {str(e)}. Please verify API keys / rate limits and try again.")

elif st.session_state.stage == "finished":
    update_steps("finished")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_app1, col_app2 = st.columns(2)
    with col_app1:
        with st.expander("🔍 Search Engine Output"):
            st.markdown(st.session_state.results.get("search", ""))
    with col_app2:
        with st.expander("📄 Reader Agent Extraction"):
            st.markdown(st.session_state.results.get("reader", ""))

    with st.container(border=True):
        st.markdown('<div class="report-label">FORMAL INTELLIGENCE SYNTHESIS</div>', unsafe_allow_html=True)
        st.markdown(st.session_state.results.get("writer", ""))
    
    st.download_button(
        label="Download Research Manuscript (.md)",
        data=st.session_state.results.get('writer', ''),
        file_name=f"agentflow_report_{int(time.time())}.md",
        mime="text/markdown"
    )
    
    with st.container(border=True):
        st.markdown('<div style="font-family:\'Playfair Display\', serif; font-weight:700; font-size:1.3rem; color:#4c1d95; margin-bottom:1rem; letter-spacing:0.02em;">Critical Review Summary</div>', unsafe_allow_html=True)
        st.markdown(st.session_state.results.get("critic", ""))

    # Compact Loop Panel
    st.markdown('<div style="margin-top:2.5rem; margin-bottom:1rem; font-family:\'Playfair Display\', serif; font-size:1.3rem; font-weight:700; color:#1e1b4b;">Run Summary & Loop Behavior</div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("System Used", st.session_state.results.get("system_label", "N/A"))
    with col_s2:
        score_val = st.session_state.results.get("score")
        st.metric("Final Critic Score", f"{score_val}/10" if score_val is not None else "N/A")
    with col_s3:
        st.metric("Retry Count", st.session_state.results.get("retry_count", 0))

    attempts_data = st.session_state.results.get("attempts", [])
    if attempts_data:
        rows = []
        for i, att in enumerate(attempts_data):
            att_num = att.get("attempt", i)
            # Display human-friendly 1-indexed attempt number
            disp_attempt = att_num + 1 if isinstance(att_num, int) else i + 1
            rows.append({
                "Attempt": f"Pass {disp_attempt}",
                "Critic Score": att.get("critic_score", att.get("score")),
                "Decision": att.get("decision", "end"),
            })
        import pandas as pd
        st.caption("Attempts & Decisions")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    token_usage = st.session_state.results.get("token_usage")
    if token_usage and isinstance(token_usage, dict):
        total_tokens = token_usage.get("total_tokens") or token_usage.get("total")
        if total_tokens:
            st.caption(f"Total tokens used: {total_tokens:,}")

per_system_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "analysis", "per_system.csv")
if os.path.exists(per_system_path):
    with st.expander("📊 About This Study"):
        st.markdown("Adaptive loops raise RAGAS faithfulness over the fixed pipeline; the classifier judge's extra gain is not significant after correction.")
        import pandas as pd
        df_study = pd.read_csv(per_system_path)
        st.dataframe(df_study, use_container_width=True)

st.markdown("""
<div class="notice">
    AgentFlow · LangGraph · research build
</div>
""", unsafe_allow_html=True)