"""
统一重训练：用 Serial + T-Batch 模式重新训练所有架构。
"""

import json
import subprocess
import sys
from pathlib import Path

def retrain_architecture(mode_name, arch_params, output_dir):
    """用 Serial + T-Batch 重训练单个架构"""

    # 构建命令
    cmd = [
        "python", "-u", "train_single_arch.py",
        "--model", str(arch_params["model"]),
        "--embedding-dim", str(arch_params["embedding_dim"]),
        "--batch-mode", "tbatch",
        "--train-batch-size", "32",
        "--dataset", "public_csv",
        "--local-data-path", "data/public/mooc.csv",
        "--max-events", "20000",
        "--epochs", "3",
        "--seed", "42",
        "--output-dir", output_dir,
    ]

    # 添加可选参数
    if arch_params.get("memory_cell"):
        cmd.extend(["--memory-cell", str(arch_params["memory_cell"])])
    if arch_params.get("time_proj"):
        cmd.extend(["--time-proj", str(arch_params["time_proj"])])
    if arch_params.get("use_static_embeddings"):
        cmd.extend(["--use-static-embeddings", str(arch_params["use_static_embeddings"])])
    if arch_params.get("event_agg"):
        cmd.extend(["--event-agg", str(arch_params["event_agg"])])
    if arch_params.get("max_neighbors") is not None:
        cmd.extend(["--max-neighbors", str(arch_params["max_neighbors"])])

    print(f"\n{'='*80}")
    print(f"重训练 {mode_name} 的最优架构")
    print(f"{'='*80}")
    print(f"架构: {arch_params['model']}, {arch_params['embedding_dim']}-dim")
    print(f"命令: {' '.join(cmd)}")
    print()

    # 执行训练
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Error: 重训练 {mode_name} 失败")
        return None

    # 读取结果
    result_file = Path(output_dir) / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    else:
        print(f"Warning: 结果文件 {result_file} 不存在")
        return None

def main():
    # 读取提取的架构
    with open("unified_retrain_architectures.json") as f:
        architectures = json.load(f)

    # 创建输出目录
    base_output_dir = Path("outputs/unified_retrain")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 重训练所有架构
    results = {}

    for mode_name, arch_data in architectures.items():
        output_dir = base_output_dir / mode_name
        output_dir.mkdir(exist_ok=True)

        result = retrain_architecture(
            mode_name,
            arch_data["arch_params"],
            str(output_dir)
        )

        if result:
            results[mode_name] = {
                "arch_params": arch_data["arch_params"],
                "original_mrr": arch_data["original_mrr"],
                "retrain_test_mrr": result.get("test_mrr"),
                "retrain_test_recall": result.get("test_recall_at_k"),
                "retrain_time_sec": result.get("time_sec"),
            }

    # 保存结果
    output_file = base_output_dir / "unified_retrain_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("统一重训练完成")
    print(f"{'='*80}")
    print(f"结果已保存到: {output_file}")

    # 打印对比
    print("\n## 重训练结果对比\n")
    print(f"{'模式':<20} {'原始MRR':<12} {'重训MRR':<12} {'差异':<10}")
    print("-" * 60)
    for mode_name, data in results.items():
        orig = data["original_mrr"]
        retrain = data["retrain_test_mrr"]
        diff = retrain - orig if (orig and retrain) else None
        print(f"{mode_name:<20} {orig:<12.4f} {retrain:<12.4f} {diff:+.4f}" if diff else f"{mode_name:<20} {orig:<12} {retrain:<12}")

if __name__ == "__main__":
    main()
