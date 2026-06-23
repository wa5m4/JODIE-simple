"""Monitor Data Parallel experiment progress."""

import re
import time
from pathlib import Path

def check_progress(log_file="data_parallel_tbatch_rerun.log"):
    """Check experiment progress from log file."""
    if not Path(log_file).exists():
        print(f"Log file {log_file} not found")
        return None

    with open(log_file) as f:
        lines = f.readlines()

    # Find all completed trials
    trials = []
    for line in lines:
        match = re.search(r'\[DataParallel (\d+)/(\d+)\].*val_score=([\d.]+)', line)
        if match:
            trial_num = int(match.group(1))
            total_trials = int(match.group(2))
            score = float(match.group(3))
            trials.append((trial_num, score))

    if not trials:
        print("No completed trials found yet")
        return None

    current_trial, _ = trials[-1]
    total_trials = 27
    progress_pct = (current_trial / total_trials) * 100

    print(f"Progress: {current_trial}/{total_trials} trials ({progress_pct:.1f}%)")
    print(f"\nCompleted trials:")
    for trial_num, score in trials:
        print(f"  Trial {trial_num:2d}: MRR = {score:.4f}")

    best_trial = max(trials, key=lambda x: x[1])
    print(f"\nBest so far: Trial {best_trial[0]} with MRR = {best_trial[1]:.4f}")

    return current_trial, total_trials

if __name__ == "__main__":
    check_progress()
