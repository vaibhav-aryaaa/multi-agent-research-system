import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# If Hugging Face cache contains the model locally, stay offline to prevent slow remote checks.
# If not present (e.g. fresh CI runner), allow it to fetch without crashing.
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir) and any("all-MiniLM-L6-v2" in d for d in os.listdir(cache_dir)):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
