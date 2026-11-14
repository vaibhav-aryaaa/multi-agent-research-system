# Environment Specification and Reproduction Notes

This document details the software and hardware environment used to execute the evaluation experiments and reproduce all tables and figures.

## Environment Details
- **Python Version**: `Python 3.13` (tested on 3.13.2)
- **Operating System**: macOS (Darwin 24.3.0, Apple Silicon / ARM64). Standard Linux (Ubuntu 22.04 LTS / 24.04 LTS) is also supported with equivalent dependencies.
- **Virtual Environment**: Standard Python `venv` (`python3.13 -m venv .venv`).
- **Approximate Installation Size**: ~1.7 GB (including PyTorch CPU, Hugging Face Transformers, Sentence-Transformers, and Ragas dependencies).

## Machine Learning & Neural Network Checkpoints
- **Cross-Encoder Judge**: `vaibhavarya/distilbert-faithfulness-nli` (configured via `FAITHFULNESS_CE_PATH`).
  - Hugging Face model repository: `https://huggingface.co/vaibhavarya/distilbert-faithfulness-nli`
  - Base architecture: `distilbert-base-uncased` fine-tuned on sentence-level NLI entailment for research grounding.
  - Download behavior: Downloads automatically on first execution and caches in `~/.cache/huggingface/hub/`.
- **RAGAS Local Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
  - Used locally for question generation / answer relevancy semantic embeddings; runs offline with zero API calls.

## API Keys & Operational Modes
| Task | Keys Required | Operational Governor |
| :--- | :--- | :--- |
| **Full Offline Reproduction** | **None** | Fully offline; zero external calls. |
| **Pipeline Benchmark Generation** (`scripts/run_eval.py`) | `GROQ_API_KEY`, `TAVILY_API_KEY` | Daily token budget (195k tokens/day), exponential backoff, rate sleep. |
| **RAGAS Faithfulness Evaluation** (`scripts/run_ragas.py`) | `GOOGLE_API_KEY` | Free-tier governor: max 10 requests/min, daily budget 450 requests. |
