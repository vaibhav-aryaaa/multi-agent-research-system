import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from graph import run_pipeline
from config import FAITHFULNESS_THRESHOLD

topic = "Perpetual mechanical energy extraction from Casimir cavity forces"

print("=" * 70)
print(f"ADAPTIVE_CLASSIFIER REAL RUN (FAITHFULNESS_THRESHOLD={FAITHFULNESS_THRESHOLD})")
print(f"TOPIC: {topic}")
print("=" * 70, flush=True)

start = time.time()
result = run_pipeline(topic=topic, mode="adaptive_classifier")
duration = time.time() - start

print("\n" + "=" * 70)
print(f"COMPLETED in {duration:.1f}s")
print(f"Retry count: {result.get('retry_count')}")
print(f"Final Score: {result.get('score')}")
print(f"Tokens: {result.get('token_usage')}")
attempts = result.get("attempts", [])
print(f"Attempts count: {len(attempts)}")

for idx, att in enumerate(attempts):
    print(f"\n--- Attempt {idx} ---")
    print(f"  Decision: {att.get('decision')}")
    print(f"  Critic score: {att.get('critic_score')}")
    print(f"  Judge accept: {att.get('judge_accept')}")
    print(f"  Judge score (supported_fraction): {att.get('judge_score')}")
    print(f"  Judge reason: {att.get('judge_reason')}")
    details = att.get("judge_details") or {}
    print(f"  Claims count: {details.get('n_claims')}")
    print(f"  Supported count: {details.get('n_supported')}")
    unsupp = details.get("unsupported_claims", [])
    print(f"  Unsupported claims (showing up to 5 of {len(unsupp)}):")
    for u_idx, u in enumerate(unsupp[:5], 1):
        c_text = u.get("claim", "")[:120]
        p_val = u.get("p_supported", 0.0)
        print(f"    {u_idx}. [p_supp={p_val:.4f}] {c_text}...")

print("=" * 70)
