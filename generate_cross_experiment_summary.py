"""
Generate comprehensive cross-experiment summary comparing all 4 execution modes.
"""

import json
import pandas as pd
from pathlib import Path

def load_results(base_dir: str):
    """Load results from all execution modes."""
    base_path = Path(base_dir)

    modes = {
        "serial_tbatch": "Serial (T-Batch)",
        "data_parallel_tbatch": "Data Parallel (T-Batch)",
        "pipeline_naive_tbatch": "Pipeline Naive (T-Batch)",
        "pipeline_smart_tbatch": "Pipeline Smart (T-Batch)",
    }

    results = {}

    for mode_dir, mode_name in modes.items():
        mode_path = base_path / mode_dir

        # Load best architecture
        best_arch_file = mode_path / "best_arch.json"
        leaderboard_file = mode_path / "leaderboard.csv"

        if not best_arch_file.exists():
            print(f"Warning: {best_arch_file} not found, skipping {mode_name}")
            continue

        with open(best_arch_file) as f:
            best_arch = json.load(f)

        # Load leaderboard for all trials
        if leaderboard_file.exists():
            leaderboard = pd.read_csv(leaderboard_file)
        else:
            leaderboard = None

        results[mode_name] = {
            "best_arch": best_arch,
            "leaderboard": leaderboard,
            "mode_dir": mode_dir,
        }

    return results

def print_summary(results):
    """Print comprehensive summary."""
    print("=" * 80)
    print("CROSS-EXPERIMENT SUMMARY: All Execution Modes")
    print("=" * 80)
    print()

    # Table 1: Best Architecture Comparison
    print("## 1. Best Architecture Found by Each Mode")
    print("-" * 80)

    rows = []
    for mode_name, data in results.items():
        arch = data["best_arch"]
        config = arch.get("config", {})

        row = {
            "Mode": mode_name,
            "Model": config.get("model", "N/A"),
            "Embedding": config.get("embedding_dim", "N/A"),
            "Memory": config.get("memory_updater", "N/A"),
            "Aggregator": config.get("aggregator", "N/A"),
            "MRR": f"{arch.get('mrr', 0):.4f}",
            "Recall@10": f"{arch.get('recall_at_k', 0):.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Table 2: Performance Ranking
    print("## 2. Performance Ranking")
    print("-" * 80)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["best_arch"].get("mrr", 0),
        reverse=True
    )

    best_mrr = sorted_results[0][1]["best_arch"].get("mrr", 0)

    rank_rows = []
    for rank, (mode_name, data) in enumerate(sorted_results, 1):
        mrr = data["best_arch"].get("mrr", 0)
        gap = ((best_mrr - mrr) / best_mrr * 100) if best_mrr > 0 else 0

        rank_rows.append({
            "Rank": rank,
            "Mode": mode_name,
            "MRR": f"{mrr:.4f}",
            "Gap vs Best": f"{gap:.1f}%",
        })

    df_rank = pd.DataFrame(rank_rows)
    print(df_rank.to_string(index=False))
    print()

    # Table 3: Search Space Exploration
    print("## 3. Search Space Exploration")
    print("-" * 80)

    exploration_rows = []
    for mode_name, data in results.items():
        lb = data["leaderboard"]
        if lb is not None and len(lb) > 0:
            exploration_rows.append({
                "Mode": mode_name,
                "Trials": len(lb),
                "Unique Models": lb["model"].nunique() if "model" in lb.columns else "N/A",
                "Avg MRR": f"{lb['mrr'].mean():.4f}" if "mrr" in lb.columns else "N/A",
                "Std MRR": f"{lb['mrr'].std():.4f}" if "mrr" in lb.columns else "N/A",
            })

    if exploration_rows:
        df_explore = pd.DataFrame(exploration_rows)
        print(df_explore.to_string(index=False))
        print()

    # Table 4: Architecture Diversity
    print("## 4. Architecture Diversity Analysis")
    print("-" * 80)

    for mode_name, data in results.items():
        lb = data["leaderboard"]
        if lb is not None and len(lb) > 0:
            print(f"\n{mode_name}:")
            if "embedding_dim" in lb.columns:
                print(f"  Embedding dims explored: {sorted(lb['embedding_dim'].unique())}")
            if "memory_updater" in lb.columns:
                print(f"  Memory updaters: {lb['memory_updater'].value_counts().to_dict()}")
            if "model" in lb.columns:
                print(f"  Models: {lb['model'].value_counts().to_dict()}")

    print()
    print("=" * 80)


def main():
    base_dir = "outputs/full_cross_experiment_fixed"
    results = load_results(base_dir)

    if not results:
        print("No results found!")
        return

    print_summary(results)

    # Save summary to file
    output_file = Path(base_dir) / "cross_experiment_summary.txt"
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
