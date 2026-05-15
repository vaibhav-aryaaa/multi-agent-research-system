import streamlit as st
import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="AgentFlow · Multi-Agent Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Lora', serif;
    color: #2d241e;
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
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(91, 33, 182, 0.15), transparent);
    margin: 2rem 0 4rem;
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

col_input, col_spacer, col_pipeline = st.columns([5, 0.6, 4])

with col_input:
    topic_input = st.text_input(
        "RESEARCH OBJECTIVE",
        placeholder="e.g. Advancements in CRISPR gene editing for 2025",
        key="topic_field"
    )
    if st.button("Synthesize Data"):
        if topic_input:
            st.session_state.results = {}
            st.session_state.stage = "search"
            st.rerun()
        else:
            st.warning("Please define a research objective.")
    
    st.markdown('<div style="margin-top:2rem; display:flex; align-items:center; flex-wrap:wrap;">', unsafe_allow_html=True)
    st.markdown('<span style="font-family:\'DM Mono\', monospace; font-size:0.8rem; color:#9ca3af; margin-right:0.8rem; letter-spacing:0.1em;">SUGGESTIONS:</span>', unsafe_allow_html=True)
    examples = ["Web3 Security", "Quantum Supremacy", "Vertical Farming"]
    for ex in examples:
        st.markdown(f'<span class="try-chip">{ex}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div style="font-family:\'Playfair Display\', serif; font-size:1.6rem; font-weight:700; color:#1e1b4b; margin-bottom:1.8rem; letter-spacing:0.02em;">Pipeline Workflow</div>', unsafe_allow_html=True)
    
    s = st.session_state.stage
    render_step("01", "Researcher", "running" if s=="search" else ("done" if s in ["reader","writer","critic","finished"] else "waiting"), "Scanning high-authority web sources")
    render_step("02", "Deep Reader", "running" if s=="reader" else ("done" if s in ["writer","critic","finished"] else "waiting"), "Parsing & extracting structured content")
    render_step("03", "Synthesis", "running" if s=="writer" else ("done" if s in ["critic","finished"] else "waiting"), "Writing the final report")
    render_step("04", "Quality Assurance", "running" if s=="critic" else ("done" if s=="finished" else "waiting"), "Evaluating for bias and accuracy")

if st.session_state.stage != "idle" and st.session_state.stage != "finished":
    topic = st.session_state.topic_field
    
    if st.session_state.stage == "search":
        agent = build_search_agent()
        res = agent.invoke({"messages": [("user", f"Search for: {topic}")]})
        st.session_state.results["search"] = res["messages"][-1].content
        st.session_state.stage = "reader"
        st.rerun()

    elif st.session_state.stage == "reader":
        agent = build_reader_agent()
        res = agent.invoke({"messages": [("user", f"Read and extract deep info for: {topic}. Results: {st.session_state.results['search']}")]})
        st.session_state.results["reader"] = res["messages"][-1].content
        st.session_state.stage = "writer"
        st.rerun()

    elif st.session_state.stage == "writer":
        res = writer_chain.invoke({
            "topic": topic,
            "research": f"SEARCH:\n{st.session_state.results['search']}\n\nCONTENT:\n{st.session_state.results['reader']}"
        })
        st.session_state.results["writer"] = res
        st.session_state.stage = "critic"
        st.rerun()

    elif st.session_state.stage == "critic":
        res = critic_chain.invoke({"report": st.session_state.results["writer"]})
        st.session_state.results["critic"] = res
        st.session_state.stage = "finished"
        st.rerun()

if st.session_state.stage == "finished":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_app1, col_app2 = st.columns(2)
    with col_app1:
        with st.expander("🔍 Search Engine Output"):
            st.markdown(f'<div style="font-size:0.95rem; color:#4b5563; line-height:1.7;">{st.session_state.results.get("search", "")}</div>', unsafe_allow_html=True)
    with col_app2:
        with st.expander("📄 Reader Agent Extraction"):
            st.markdown(f'<div style="font-size:0.95rem; color:#4b5563; line-height:1.7;">{st.session_state.results.get("reader", "")}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-container">
        <div class="report-label">Formal Intelligence Synthesis</div>
        <div class="report-content">
            {st.session_state.results['writer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.download_button(
        label="Download Research Manuscript (.md)",
        data=st.session_state.results['writer'],
        file_name=f"agentflow_report_{int(time.time())}.md",
        mime="text/markdown"
    )
    
    st.markdown(f"""
    <div class="feedback-container">
        <div style="font-family:'Playfair Display', serif; font-weight:700; font-size:1.3rem; color:#4c1d95; margin-bottom:1rem; letter-spacing:0.02em;">Critical Review Summary</div>
        <div style="font-size:1.05rem; color:#374151; line-height:1.8;">
            {st.session_state.results['critic']}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="notice">
    AgentFlow · Powered by Groq & LangGraph · v1.1.0
</div>
""", unsafe_allow_html=True)