"""Collect human validation claims from diverse research topics (Phase 3 Part D).

Features:
- Runs each topic in fixed mode (one run per topic, no retries).
- Loads existing reports from docs/examples/*.json (Casimir baseline reports).
- Resume support: reuses existing result JSONs from results/ if topic already completed.
- Clean stop on daily rate-limit error (saves CSV collected so far, prints missing topics).
- Extracts clean claims with extract_claims_with_stats (excludes formulas and meta-sentences).
- Maps top-2 evidence passages by cosine similarity using all-MiniLM-L6-v2.
- Samples at most 12 claims per report, spread evenly across the report (seed 42).
- Writes data/validation_claims.csv with empty label column.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from faithfulness.claims import extract_claims_with_stats, split_evidence
from faithfulness.embedder import embed
from graph import run_pipeline

DEV_TOPICS_PATH = os.path.join(PROJECT_ROOT, "data", "dev_topics.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOCS_EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "docs", "examples")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "validation_claims.csv")


def load_dev_topics(path: str = DEV_TOPICS_PATH) -> List[str]:
    """Load research topics from text file, skipping blank lines and comments."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Topics file not found at: {path}")
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                topics.append(t)
    return topics


def find_existing_result(topic: str) -> Optional[Dict[str, Any]]:
    """Scan results/ directory to find an existing run matching the topic."""
    if not os.path.exists(RESULTS_DIR):
        return None
    for fpath in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = data.get("topic", "").strip().lower()
            if t == topic.strip().lower() and data.get("final_report"):
                data["_source_file"] = fpath
                return data
        except Exception:
            continue
    return None


def sample_claims_spread(claims: List[str], max_k: int = 12, seed: int = 42) -> List[int]:
    """Sample up to max_k claim indices evenly distributed across the report."""
    n = len(claims)
    if n <= max_k:
        return list(range(n))

    # Spread indices uniformly across the entire report body
    raw_steps = np.linspace(0, n - 1, max_k)
    selected = sorted(list(set(int(round(s)) for s in raw_steps)))

    if len(selected) < max_k:
        remaining = [i for i in range(n) if i not in selected]
        rng = np.random.RandomState(seed)
        extra = rng.choice(remaining, size=max_k - len(selected), replace=False)
        selected = sorted(selected + list(extra))

    return selected


def process_report_for_claims(
    report_id: str,
    topic: str,
    report_text: str,
    contexts: List[str],
) -> List[Dict[str, Any]]:
    """Extract claims, pair with top-2 evidence passages, and sample spread across report."""
    claims, stats = extract_claims_with_stats(report_text)
    if not claims:
        print(f"  [WARN] No claims extracted for report {report_id}")
        return []

    # Assemble passages from accumulated contexts
    raw_evidence = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)
    passages = split_evidence("", raw_evidence)
    if not passages and contexts:
        passages = [str(c).strip() for c in contexts if str(c).strip()]

    # Embed claims and passages
    c_embs = embed(claims)
    if passages:
        p_embs = embed(passages)
        # Cosine similarity matrix (normalized dot product)
        sim_matrix = np.dot(c_embs, p_embs.T)
    else:
        sim_matrix = np.zeros((len(claims), 0), dtype=np.float32)

    claim_pairs = []
    for i, claim in enumerate(claims):
        if passages and sim_matrix.shape[1] > 0:
            top_idx = np.argsort(sim_matrix[i])[::-1][:min(2, len(passages))]
            ev1 = passages[top_idx[0]] if len(top_idx) > 0 else ""
            ev2 = passages[top_idx[1]] if len(top_idx) > 1 else ""
        else:
            ev1, ev2 = "", ""

        claim_pairs.append((claim, ev1, ev2))

    # Sample up to 12 claims spread across report
    selected_indices = sample_claims_spread(claims, max_k=12, seed=42)

    rows = []
    for idx in selected_indices:
        claim, ev1, ev2 = claim_pairs[idx]
        rows.append({
            "report_id": report_id,
            "topic": topic,
            "claim_index": idx + 1,
            "claim": claim,
            "evidence_top1": ev1,
            "evidence_top2": ev2,
            "label": "",  # Empty for human annotation
        })

    return rows


def main():
    print("=" * 80)
    print("PHASE 3 PART D: CLAIM COLLECTION PIPELINE")
    print("=" * 80)

    all_rows: List[Dict[str, Any]] = []
    processed_topics = set()
    rows_per_topic: Dict[str, int] = {}

    # 1. Load existing Casimir baseline reports from docs/examples/
    print("\n--- Step 1: Ingesting existing baseline reports from docs/examples/ ---")
    example_files = sorted(glob.glob(os.path.join(DOCS_EXAMPLES_DIR, "*.json")))
    for fpath in example_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            rep_id = data.get("run_id") or os.path.splitext(os.path.basename(fpath))[0]
            top = data.get("topic", "Casimir Energy Extraction")
            rep_text = data.get("final_report") or data.get("report") or ""
            ctx = data.get("contexts", [])

            rep_rows = process_report_for_claims(rep_id, top, rep_text, ctx)
            all_rows.extend(rep_rows)
            rows_per_topic[f"Casimir ({rep_id[:15]})"] = len(rep_rows)
            print(f"  Ingested {len(rep_rows)} claims from {os.path.basename(fpath)}")
        except Exception as exc:
            print(f"  [ERROR] Failed to ingest {fpath}: {exc}")

    # 2. Process dev topics
    print("\n--- Step 2: Processing dev topics from data/dev_topics.txt ---")
    topics = load_dev_topics()
    print(f"Loaded {len(topics)} topics to process.")

    missing_topics: List[str] = []
    daily_limit_stopped = False

    for idx, topic in enumerate(topics, 1):
        if daily_limit_stopped:
            missing_topics.append(topic)
            continue

        print(f"\n[{idx}/{len(topics)}] Topic: {topic}")

        # Check resume cache in results/
        cached_result = find_existing_result(topic)
        if cached_result is not None:
            source_file = os.path.basename(cached_result.get("_source_file", "results/"))
            rep_id = cached_result.get("run_id") or source_file.replace(".json", "")
            print(f"  [RESUME] Found existing run log ({source_file}). Reusing.")
            rep_text = cached_result.get("final_report") or cached_result.get("report") or ""
            ctx = cached_result.get("contexts", [])

            rep_rows = process_report_for_claims(rep_id, topic, rep_text, ctx)
            all_rows.extend(rep_rows)
            rows_per_topic[topic[:40] + "..."] = len(rep_rows)
            processed_topics.add(topic)
            continue

        # Execute real pipeline in fixed mode (no retries)
        print("  Running fixed-mode pipeline (one pass, no retries)...")
        try:
            state = run_pipeline(topic, mode="fixed")
            rep_id = state.get("run_id", f"run_{idx}")
            rep_text = state.get("report", "")
            # Reconstruct contexts from final state
            from runlog import build_contexts
            ctx = build_contexts(state, system="fixed")

            rep_rows = process_report_for_claims(rep_id, topic, rep_text, ctx)
            all_rows.extend(rep_rows)
            rows_per_topic[topic[:40] + "..."] = len(rep_rows)
            processed_topics.add(topic)
            print(f"  Extracted {len(rep_rows)} claims for topic.")

            # Pace successive calls to respect 8,000 TPM limit
            time.sleep(35)

        except Exception as exc:
            err_str = str(exc).lower()
            if any(term in err_str for term in ["daily limit", "day limit", "tpd", "quota exceeded", "rate limit", "429"]):
                print(f"\n[RATE LIMIT / DAILY CAP] Encountered limit: {exc}")
                print("Stopping collection cleanly. Proceeding to save data collected so far.")
                daily_limit_stopped = True
                missing_topics.append(topic)
            else:
                print(f"\n[ERROR] Pipeline run failed for topic '{topic}': {exc}")
                missing_topics.append(topic)

    # 3. Write CSV output
    print("\n--- Step 3: Saving validation claims dataset ---")
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(df_out)} total claims to: {OUTPUT_CSV_PATH}")

    # 4. Summary reporting
    print("\n" + "=" * 80)
    print(f"{'Topic':<48} | {'Claims Extracted'}")
    print("-" * 80)
    for t_name, count in rows_per_topic.items():
        print(f"{t_name:<48} | {count}")
    print("=" * 80)
    print(f"Total Rows Collected: {len(df_out)}")

    if missing_topics:
        print(f"\n[NOTE] {len(missing_topics)} topics could not be processed due to rate limits or errors:")
        for mt in missing_topics:
            print(f"  - {mt}")

    # 5. Print 5 sample rows
    print("\n--- 5 Sample Rows from validation_claims.csv ---")
    for i, r in df_out.head(5).iterrows():
        print(f"\nRow {i+1}:")
        print(f"  Report ID: {r['report_id']}")
        print(f"  Topic:     {r['topic']}")
        print(f"  Claim:     {r['claim']}")
        print(f"  Ev 1:      {r['evidence_top1'][:120]}...")
        print(f"  Ev 2:      {r['evidence_top2'][:120]}...")
        print(f"  Label:     '{r['label']}'")


if __name__ == "__main__":
    main()
