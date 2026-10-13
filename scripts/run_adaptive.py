import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import run_pipeline

topic = "Perpetual mechanical energy extraction from Casimir cavity forces"
threshold = os.getenv("QUALITY_THRESHOLD", "7.0")

print("=" * 70)
print(f"ADAPTIVE_LLM REAL RUN (QUALITY_THRESHOLD={threshold})")
print(f"TOPIC: {topic}")
print("=" * 70, flush=True)

start = time.time()
result = run_pipeline(topic=topic, mode="adaptive_llm")
duration = time.time() - start

print("\n" + "=" * 70)
print(f"COMPLETED in {duration:.1f}s")
print(f"Retry count: {result.get('retry_count')}")
print(f"Final Score: {result.get('score')}")
print(f"Tokens: {result.get('token_usage')}")
print(f"Attempts: {len(result.get('attempts', []))}")
print("=" * 70)
