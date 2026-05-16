# 🔬 AgentFlow: Multi-Agent AI Research Intelligence

![AgentFlow Logo](https://raw.githubusercontent.com/vaibhav-aryaaa/multi-agent-research-system/main/assets/logo.png)

**AgentFlow** is a production-grade, multi-agent AI research platform designed to synthesize high-fidelity, academic-quality manuscripts. Powered by **LangGraph** and **Llama 3.3 70B**, it orchestrates a specialized team of AI agents to crawl the deep web, extract scholarly data, and produce refined research reports.

![Dashboard Preview](https://raw.githubusercontent.com/vaibhav-aryaaa/multi-agent-research-system/main/assets/dashboard.png)

## 🌟 Key Features

### 🎓 Scholar Mode (Advanced Filtering)
Unlike general AI search tools, AgentFlow is anchored to elite academic domains. It prioritizes sources from **ArXiv, Nature, ScienceDirect, ResearchGate, and JSTOR**, while filtering out social media and low-quality blogs.

### 🤖 Multi-Agent Orchestration
The system utilizes a sequential agentic pipeline:
- **The Researcher**: Scans high-authority academic repositories.
- **The Deep Reader**: Parses and extracts structured content from complex manuscripts.
- **The Synthesizer**: Writes a professional-grade research report.
- **The Critic**: Evaluates the report for bias, accuracy, and depth.

### 🎨 Premium "Bookish" UI
- **Glassmorphism Design**: High-density interactive suggestion chips with backdrop blurs.
- **Dynamic UX**: Randomized research prompts and trending topic "chips" on every reload.
- **Real-time Pipeline Tracking**: Visual stage-based cards to monitor the agents' progress.

---

## 🛠️ Tech Stack

- **LLM**: Groq (Llama 3.3 70B Versatile)
- **Orchestration**: LangGraph / LangChain
- **Search & Extraction**: Tavily AI (Scholar Mode enabled)
- **Frontend**: Streamlit (Custom CSS Glassmorphism)
- **Environment**: Python 3.10+

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vaibhav-aryaaa/multi-agent-research-system.git
cd multi-agent-research-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is optimized for **Streamlit Community Cloud**. 

1. Push your code to GitHub.
2. Connect your repo to Streamlit Cloud.
3. Add your `GROQ_API_KEY` and `TAVILY_API_KEY` to the **Advanced Settings > Secrets** section in TOML format.

---

## 📄 License
MIT License - Copyright (c) 2024 Vaibhav Arya

---

> [!TIP]
> **Pro Tip**: Use the suggestion chips to see "Scholar Mode" in action with technical topics like *Quantum Computing* or *CRISPR*.
