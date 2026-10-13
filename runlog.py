import os
import json
import uuid
from datetime import datetime
from typing import Optional
from agents import model_name as default_model_name
from config import QUALITY_THRESHOLD

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def build_contexts(result: dict, system: str) -> list[str]:
    """Build contexts for RAGAS evaluation from the final accumulated evidence.
    
    Splits search_results into one string per source (splitting on the '-----' separator
    the search tool uses) and appends scraped_content as one additional string.
    Baseline runs return an empty list.
    """
    if system == "baseline":
        return []
    
    contexts: list[str] = []
    search_results = result.get("search_results") or ""
    if search_results:
        parts = search_results.split("-----")
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                contexts.append(cleaned)
                
    scraped = result.get("scraped_content") or ""
    if scraped and scraped.strip():
        contexts.append(scraped.strip())
        
    return contexts

def save_run(result: dict, system: str, results_dir: Optional[str] = None) -> str:
    """Save an execution run log as a JSON file in the results/ directory.
    
    Returns the absolute path to the generated JSON log.
    """
    target_dir = results_dir or RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    run_id = f"{timestamp_str}_{short_hash}"
    
    contexts = build_contexts(result, system)
    
    run_data = {
        "run_id": run_id,
        "system": system,
        "topic": result.get("topic", ""),
        "model_name": result.get("model_name") or default_model_name,
        "timestamp": now.isoformat(),
        "final_report": result.get("report", ""),
        "contexts": contexts,
        "retry_count": result.get("retry_count", 0),
        "quality_threshold": result.get("quality_threshold", float(os.getenv("QUALITY_THRESHOLD", str(QUALITY_THRESHOLD)))),
        "final_score": result.get("score"),
        "attempts": result.get("attempts", []),
    }
    if "token_usage" in result:
        run_data["token_usage"] = result["token_usage"]
    
    filename = f"{run_id}_{system}.json"
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)
        
    return filepath
