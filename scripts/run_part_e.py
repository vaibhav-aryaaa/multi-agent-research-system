import sys
import os
import time
import signal

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def timeout_handler(signum, frame):
    print("\n" + "!" * 70, flush=True)
    print("EXPERIMENT TIMED OUT: Exceeded hard 10-minute (600s) timeout limit.", flush=True)
    print("!" * 70, flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(600)

from agents import model_name
from graph import run_pipeline

topic = "Perpetual mechanical energy extraction from Casimir cavity forces"

print("=" * 70)
print(f"PHASE 2 PART E: FIXED-MODEL REAL RUN")
print(f"PINNED MODEL: {model_name}")
print(f"MODE: fixed (System 2: single pass)")
print(f"TOPIC: {topic}")
print("=" * 70, flush=True)

start_time = time.time()

try:
    result = run_pipeline(topic=topic, mode="fixed")
    duration = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETED in {duration:.1f}s")
    print(f"Topic: {result.get('topic')}")
    print(f"Model: {result.get('model_name')}")
    print(f"Retry count: {result.get('retry_count')}")
    print(f"Critic score: {result.get('score')}")
    token_usage = result.get('token_usage', {})
    print(f"Final Token Usage: {token_usage}")
    print(f"Report length: {len(result.get('report', ''))} chars")
    print("=" * 70, flush=True)
except Exception as e:
    duration = time.time() - start_time
    print("\n" + "!" * 70)
    print(f"EXPERIMENT STOPPED / FAILED after {duration:.1f}s")
    print(f"Error: {e}")
    print("!" * 70, flush=True)
    sys.exit(1)
