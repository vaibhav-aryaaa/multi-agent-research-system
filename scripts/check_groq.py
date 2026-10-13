import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY is not set in environment or .env")
    sys.exit(1)

client = Groq(api_key=api_key)

print("Fetching available models from Groq API...")
try:
    models_response = client.models.list()
    available_model_ids = {m.id for m in models_response.data}
    print("\nAll models available to this key:")
    for m_id in sorted(available_model_ids):
        print(f"  - {m_id}")
except Exception as e:
    print(f"Failed to list models: {e}")
    sys.exit(1)

candidates = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

print("\nTesting candidate models with a tiny completion (prompt: 'say ok', max_tokens: 5)...")
results = []

for model in candidates:
    is_avail = model in available_model_ids
    if not is_avail:
        results.append({
            "model": model,
            "available": "No",
            "tpm_limit": "N/A",
            "rpd_limit": "N/A",
            "remaining_tokens": "N/A",
            "reset_tokens": "N/A",
        })
        continue

    try:
        raw_resp = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
        )
        headers = raw_resp.headers
        tpm_limit = headers.get("x-ratelimit-limit-tokens", "unknown")
        remaining_tokens = headers.get("x-ratelimit-remaining-tokens", "unknown")
        rpd_limit = headers.get("x-ratelimit-limit-requests", "unknown")
        rem_requests = headers.get("x-ratelimit-remaining-requests", "unknown")
        reset_tokens = headers.get("x-ratelimit-reset-tokens", "unknown")
        
        results.append({
            "model": model,
            "available": "Yes",
            "tpm_limit": tpm_limit,
            "rpd_limit": f"{rpd_limit} (rem: {rem_requests})",
            "remaining_tokens": remaining_tokens,
            "reset_tokens": reset_tokens,
        })
    except Exception as e:
        results.append({
            "model": model,
            "available": "Yes (Error)",
            "tpm_limit": "Error",
            "rpd_limit": str(e),
            "remaining_tokens": "Error",
            "reset_tokens": "Error",
        })

print("\n" + "=" * 95)
print(f"{'Model':<28} | {'Available':<10} | {'TPM Limit':<12} | {'RPD Limit / Remaining':<22} | {'Remaining Tokens':<15}")
print("=" * 95)
for r in results:
    print(f"{r['model']:<28} | {r['available']:<10} | {r['tpm_limit']:<12} | {r['rpd_limit']:<22} | {r['remaining_tokens']:<15}")
print("=" * 95)
