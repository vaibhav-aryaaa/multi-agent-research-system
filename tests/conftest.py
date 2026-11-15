import os

# Ensure tests run fully offline without hanging on remote Hugging Face checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
