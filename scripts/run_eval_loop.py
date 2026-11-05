#!/usr/bin/env python3
"""Evaluation Runner Loop (Task E0).

Repeatedly runs scripts/run_eval.py for Q01-Q20 until all 20 questions have
all four system result files, or until daily budget halts, pausing until budget
refreshes. Exits non-zero after 3 consecutive runs with zero progress.
Logs all output to results/eval/run_loop.log.
"""

import os
import sys
import time
import re
import subprocess
from datetime import datetime

TARGET_QIDS = [f"Q{i:02d}" for i in range(1, 21)]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eval")
LOG_PATH = os.path.join(RESULTS_DIR, "run_loop.log")
SYSTEMS = ["baseline", "fixed", "adaptive_llm", "adaptive_classifier"]


def count_finished_systems(results_dir: str, qids: list) -> int:
    """Count how many total system result files exist for the target questions."""
    count = 0
    for qid in qids:
        for sys_name in SYSTEMS:
            if os.path.exists(os.path.join(results_dir, f"{qid}_{sys_name}.json")):
                count += 1
    return count


def is_all_complete(results_dir: str, qids: list) -> bool:
    """Check if all target questions have all four system files."""
    for qid in qids:
        for sys_name in SYSTEMS:
            if not os.path.exists(os.path.join(results_dir, f"{qid}_{sys_name}.json")):
                return False
    return True


def log_message(msg: str, log_file):
    """Write timestamped message to stdout and run_loop.log."""
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    log_file.write(formatted + "\n")
    log_file.flush()


def parse_wait_minutes(output: str) -> int:
    """Parse 're-run the same command in hh:mm' from runner output. Fallback: 60 mins."""
    match = re.search(r"re-run (?:the same command )?in (\d+):(\d+)", output, re.IGNORECASE)
    if match:
        hours = int(match.group(1))
        mins = int(match.group(2))
        return (hours * 60) + mins
    return 60


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    qids_arg = ",".join(TARGET_QIDS)
    python_bin = sys.executable

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_message("Starting evaluation runner loop for Q01-Q20...", log_file)

        consecutive_stalls = 0
        iteration = 0

        while True:
            iteration += 1
            if is_all_complete(RESULTS_DIR, TARGET_QIDS):
                log_message(f"All {len(TARGET_QIDS)} questions are fully complete! Exiting loop cleanly.", log_file)
                sys.exit(0)

            count_before = count_finished_systems(RESULTS_DIR, TARGET_QIDS)
            log_message(f"Iteration {iteration}: {count_before}/{len(TARGET_QIDS)*4} system outputs present.", log_file)

            cmd = [
                python_bin,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_eval.py"),
                "--only", qids_arg,
                "--sleep", "20",
                "--daily-token-budget", "195000",
            ]

            log_message(f"Executing: {' '.join(cmd)}", log_file)

            # Stream output to both terminal and log file
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            captured_output = []
            if proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                    captured_output.append(line)

            proc.wait()
            ret_code = proc.returncode
            full_out = "".join(captured_output)

            count_after = count_finished_systems(RESULTS_DIR, TARGET_QIDS)
            new_systems = count_after - count_before
            log_message(f"Run completed with return code {ret_code}. New systems completed: {new_systems}.", log_file)

            if new_systems == 0:
                consecutive_stalls += 1
                log_message(f"Warning: No new systems finished (consecutive stalls: {consecutive_stalls}/3).", log_file)
                if consecutive_stalls >= 3:
                    log_message("Error: 3 consecutive runs completed no new system files. Exiting non-zero.", log_file)
                    sys.exit(1)
            else:
                consecutive_stalls = 0

            if is_all_complete(RESULTS_DIR, TARGET_QIDS):
                log_message("All target questions complete! Finished.", log_file)
                sys.exit(0)

            # Check if stopped by daily budget
            if "daily budget reached" in full_out.lower():
                wait_mins = parse_wait_minutes(full_out)
                sleep_secs = (wait_mins * 60) + 120  # wait time + 2 minutes
                log_message(f"Daily budget reached. Sleeping for {wait_mins}m + 2m ({sleep_secs}s)...", log_file)
                time.sleep(sleep_secs)
            else:
                # If exited cleanly without daily budget message and not complete, sleep 20s before retry
                log_message("Runner exited. Waiting 20s before next iteration...", log_file)
                time.sleep(20)


if __name__ == "__main__":
    main()
