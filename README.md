# 🔬 AgentFlow: Multi-Agent AI Research Intelligence

![AgentFlow Logo](https://raw.githubusercontent.com/vaibhav-aryaaa/multi-agent-research-system/main/assets/logo.png)

**AgentFlow** is a production-grade, multi-agent AI research platform designed to synthesize high-fidelity, academic-quality manuscripts. Powered by **LangGraph** and **openai/gpt-oss-120b** (via Groq), it orchestrates a specialized team of AI agents to crawl the deep web, extract scholarly data, and produce refined research reports.

> [!NOTE]
> **Methodology Note**: All systems use `gpt-oss-120b` via Groq because the planned Llama model (`llama-3.3-70b-versatile`) was not available on our access tier. The original planned model is preserved as part of the methodology history.

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

- **LLM**: Groq (`openai/gpt-oss-120b`; originally planned `llama-3.3-70b-versatile`)
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

## ⚖️ Faithfulness Classifiers (System 4)

AgentFlow incorporates learned faithfulness classifiers to serve as independent, measurable retry judges:
- **Logistic Regression Baseline (`lr`)**: Fast baseline using frozen `all-MiniLM-L6-v2` embeddings and handcrafted sentence-pair features (`[claim, evidence, |diff|, prod, cos_sim]`).
- **DistilBERT Cross-Encoder (`cross_encoder`)**: Fine-tuned judge that reads claim and retrieved evidence jointly (`premise=passage, hypothesis=claim`, max pooling over top-$k$ passages) to detect hallucinations and ungrounded assertions.

### Training the Cross-Encoder on Google Colab GPU
1. Open Google Colab and set runtime to **GPU (T4)**.
2. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt transformers accelerate
   ```
3. Run the training script (trains on 100k balanced pairs from GLUE MNLI):
   ```bash
   python -m faithfulness.train_cross_encoder
   ```
   *(Optional: pass `--push_to_hub <username>/<model_name>` with `HF_TOKEN` set to upload directly to Hugging Face Hub).*
4. Download the trained model folder and place it in `faithfulness/models/cross_encoder/`.

### Environment Configuration & Hub Loading
Set `FAITHFULNESS_CE_PATH` to point to your local model directory or Hugging Face Hub repo id:
```bash
export FAITHFULNESS_CE_PATH="faithfulness/models/cross_encoder"
```
Heavy libraries (`torch`, `transformers`) are loaded lazily to ensure the web UI and non-classifier modes start instantly without PyTorch initialization overhead.

### In-Domain Evaluation Rule & Decision Gate
- Public MNLI serves only as pre-training data. Real system performance is decided on human-labeled in-domain evidence from actual pipeline runs:
  ```bash
  python -m faithfulness.evaluate --backend lr
  python -m faithfulness.evaluate --backend cross_encoder
  python scripts/compare_backends.py
  ```
- **Selection Rule**: The default backend (`CLASSIFIER_BACKEND`) is chosen strictly based on higher **DEV** macro-F1 (never heldout).
- **Decision Gate**: The chosen classifier must achieve $\ge 0.70$ heldout macro-F1 to serve as the System 4 retry judge.

---

## 📄 License
MIT License - Copyright (c) 2024 Vaibhav Arya

---

> [!TIP]
> **Pro Tip**: Use the suggestion chips to see "Scholar Mode" in action with technical topics like *Quantum Computing* or *CRISPR*.
