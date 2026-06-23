"""
从四个执行模式的 best_arch.json 中提取架构配置。
"""

import json
from pathlib import Path

def extract_architectures():
    base_dir = Path("outputs/full_cross_experiment_fixed")

    modes = {
        "serial": "serial_tbatch",
        "data_parallel": "data_parallel_tbatch",
        "pipeline_naive": "pipeline_naive_tbatch",
        "pipeline_smart": "pipeline_smart_tbatch",
    }

    architectures = {}

    for mode_name, dir_name in modes.items():
        best_arch_file = base_dir / dir_name / "best_arch.json"

        if not best_arch_file.exists():
            print(f"Warning: {best_arch_file} not found")
            continue

        with open(best_arch_file) as f:
            data = json.load(f)

        # 提取架构参数（不包括训练参数）
        config = data.get("config", {})
        arch_params = {
            "model": config.get("model"),
            "embedding_dim": config.get("embedding_dim"),
            "memory_cell": config.get("memory_cell"),
            "time_proj": config.get("time_proj"),
            "use_static_embeddings": config.get("use_static_embeddings"),
            "normalize_state": config.get("normalize_state"),
            "enable_graph_update": config.get("enable_graph_update"),
            "enable_event_agg": config.get("enable_event_agg"),
            "event_agg": config.get("event_agg"),
            "max_neighbors": config.get("max_neighbors"),
        }

        architectures[mode_name] = {
            "arch_params": arch_params,
            "original_mrr": data.get("mrr"),
            "original_phase": data.get("phase"),
        }

        print(f"{mode_name}: {arch_params['model']}, {arch_params['embedding_dim']}-dim, MRR={data.get('mrr'):.4f}")

    # 保存提取的架构
    output_file = "unified_retrain_architectures.json"
    with open(output_file, "w") as f:
        json.dump(architectures, f, indent=2)

    print(f"\n提取的架构已保存到: {output_file}")
    return architectures

if __name__ == "__main__":
    extract_architectures()
