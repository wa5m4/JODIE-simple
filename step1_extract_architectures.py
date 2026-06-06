#!/usr/bin/env python3
"""
步骤1: 提取四组最优架构
"""
import json
from pathlib import Path

base_dir = Path("outputs/full_cross_experiment_fixed")
experiments = ["serial_tbatch", "data_parallel_tbatch", "pipeline_naive_tbatch", "pipeline_smart_tbatch"]

print("=" * 120)
print("步骤1: 提取四组最优架构")
print("=" * 120)
print()

architectures = {}

for exp_name in experiments:
    best_arch_path = base_dir / exp_name / "best_arch.json"
    if best_arch_path.exists():
        with open(best_arch_path) as f:
            data = json.load(f)
            config = data.get("config", {})

            # Extract architecture parameters (not training parameters)
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
                "agg_activation": config.get("agg_activation"),
                "attn_type": config.get("attn_type"),
                "time_decay": config.get("time_decay"),
                "hidden_dim": config.get("hidden_dim"),
                "memory_gate": config.get("memory_gate"),
                "message_mode": config.get("message_mode"),
                "msg_linear": config.get("msg_linear"),
            }

            architectures[exp_name.replace("_tbatch", "")] = {
                "params": arch_params,
                "original_val_mrr": data.get("val_mrr", 0),
                "original_test_mrr": data.get("test_mrr", 0),
                "original_test_recall": data.get("test_recall_at_k", 0),
            }

# Print comparison table
print(f"{'搜索方法':<20} | {'Model':<25} | {'Emb':<5} | {'Cell':<6} | {'原始Test MRR':<15}")
print("-" * 120)

for name, arch in architectures.items():
    params = arch["params"]
    print(f"{name:<20} | {params['model']:<25} | {params['embedding_dim']:<5} | {params['memory_cell']:<6} | {arch['original_test_mrr']:<15.4f}")

print()
print("=" * 120)
print("架构参数详细对比")
print("=" * 120)
print()

# Find differences
all_params = set()
for arch in architectures.values():
    all_params.update(arch["params"].keys())

print(f"{'参数':<30} | {'Serial':<15} | {'Data Parallel':<15} | {'Pipeline Naive':<15} | {'Pipeline Smart':<15}")
print("-" * 120)

for param in sorted(all_params):
    values = [str(architectures[name]["params"].get(param, "N/A")) for name in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]]
    # Check if all same
    all_same = len(set(values)) == 1
    marker = "" if all_same else " ⚠️"
    print(f"{param:<30} | {values[0]:<15} | {values[1]:<15} | {values[2]:<15} | {values[3]:<15}{marker}")

print()
print("=" * 120)
print("关键差异:")
print("=" * 120)

# Identify key differences
for param in sorted(all_params):
    values = [architectures[name]["params"].get(param) for name in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]]
    if len(set(str(v) for v in values)) > 1:
        print(f"  {param}:")
        for name in ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]:
            print(f"    {name}: {architectures[name]['params'].get(param)}")

# Save architectures for next step
output_file = "arch_reeval_configs.json"
with open(output_file, "w") as f:
    json.dump(architectures, f, indent=2)

print()
print(f"✅ 架构配置已保存到: {output_file}")
