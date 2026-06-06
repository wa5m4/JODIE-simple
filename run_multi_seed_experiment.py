#!/usr/bin/env python3
"""
多种子全面实验：对比Serial、Data Parallel、Pipeline Naive、Pipeline Smart
- 数据量: 20000
- Trials: 27
- Epochs: 3
- GPUs: 0,1,2
- 评估模式: 在线(frozen=False)
"""

import subprocess
import json
import os
from pathlib import Path
import time

# 实验配置
SEEDS = [20042, 12345, 67890]  # 3个不同的种子
DATA_SIZE = 20000
NUM_TRIALS = 27
EPOCHS = 3
GPUS = "0,1,2"
EVAL_MODE = "online"  # frozen=False
DATA_PATH = "data/public/mooc.csv"
DATASET = "public_csv"

MODES = ["serial", "data_parallel", "pipeline_naive", "pipeline_smart"]

BASE_OUTPUT_DIR = "outputs/multi_seed_experiment"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def run_nas_search(mode, seed, output_dir):
    """运行NAS搜索"""
    print(f"\n{'='*80}")
    print(f"Running NAS Search: mode={mode}, seed={seed}")
    print(f"{'='*80}\n")

    # 根据模式设置execution-mode和其他参数
    if mode == "serial":
        execution_mode = "serial"
        extra_args = ["--gpu-list", GPUS]
    elif mode == "data_parallel":
        execution_mode = "data_parallel"
        extra_args = ["--data-parallel-visible-gpus", GPUS]
    elif mode == "pipeline_naive":
        execution_mode = "ray_pipeline"
        extra_args = ["--gpu-list", GPUS, "--pipeline-mode", "naive"]
    elif mode == "pipeline_smart":
        execution_mode = "ray_pipeline"
        extra_args = ["--gpu-list", GPUS, "--pipeline-mode", "smart"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cmd = [
        "python", "search.py",
        "--execution-mode", execution_mode,
        "--dataset", DATASET,
        "--local-data-path", DATA_PATH,
        "--max-events", str(DATA_SIZE),
        "--trials", str(NUM_TRIALS),
        "--epochs-per-trial", str(EPOCHS),
        "--seed", str(seed),
        "--eval-frozen", "false",
        "--output-dir", output_dir
    ] + extra_args

    # 保存命令到日志
    log_file = os.path.join(output_dir, "nas_search.log")
    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")

    # 实时显示输出到终端和日志文件
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    with open(log_file, "a") as f:
        f.write(f"\nElapsed: {elapsed:.1f}s\n")

    print(f"\n✅ NAS搜索完成，耗时: {elapsed:.1f}s")
    return result.returncode == 0, elapsed

def extract_best_arch(mode, seed, output_dir):
    """提取最佳架构"""
    print(f"Extracting best architecture for {mode} seed={seed}")

    results_file = os.path.join(output_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"  ❌ Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        results = json.load(f)

    if not results:
        print(f"  ❌ No results found")
        return None

    # 找到最佳架构
    best = max(results, key=lambda x: x.get("test_mrr", 0))

    arch_file = os.path.join(output_dir, "best_architecture.json")
    with open(arch_file, "w") as f:
        json.dump(best, f, indent=2)

    print(f"  ✅ Best MRR: {best.get('test_mrr', 0):.4f}")
    return best

def retrain_arch(mode, seed, output_dir, arch):
    """重训最佳架构"""
    print(f"\n{'='*80}")
    print(f"Retraining: mode={mode}, seed={seed}")
    print(f"{'='*80}\n")

    retrain_dir = os.path.join(output_dir, "retrain")
    os.makedirs(retrain_dir, exist_ok=True)

    cmd = [
        "python", "train_single_arch.py",
        "--dataset", DATASET,
        "--local-data-path", DATA_PATH,
        "--max-events", str(DATA_SIZE),
        "--epochs", str(EPOCHS),
        "--seed", str(seed),
        "--eval-frozen", "false",
        "--output-dir", retrain_dir,
        "--model", arch["config"]["model"],
        "--embedding-dim", str(arch["config"]["embedding_dim"]),
        "--memory-cell", arch["config"]["memory_cell"],
        "--time-proj", arch["config"]["time_proj"]
    ]

    # 保存命令到日志
    log_file = os.path.join(retrain_dir, "retrain.log")
    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")

    # 实时显示输出到终端
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    with open(log_file, "a") as f:
        f.write(f"\nElapsed: {elapsed:.1f}s\n")

    # 从输出目录读取结果
    result_file = os.path.join(retrain_dir, "result.json")
    retrain_result = None
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            retrain_result = json.load(f)
        print(f"  ✅ Retrain MRR: {retrain_result.get('test_mrr', 0):.4f}")
    else:
        print(f"  ❌ Failed to find retrain result")

    return result.returncode == 0, elapsed, retrain_result

def run_experiment():
    """运行完整实验"""
    all_results = {}

    for seed in SEEDS:
        print(f"\n\n{'#'*80}")
        print(f"# SEED: {seed}")
        print(f"{'#'*80}\n")

        seed_results = {}

        for mode in MODES:
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}", mode)
            os.makedirs(output_dir, exist_ok=True)

            # 1. NAS搜索
            nas_success, nas_time = run_nas_search(mode, seed, output_dir)
            if not nas_success:
                print(f"❌ NAS search failed for {mode}")
                continue

            # 2. 提取最佳架构
            best_arch = extract_best_arch(mode, seed, output_dir)
            if not best_arch:
                print(f"❌ Failed to extract best architecture for {mode}")
                continue

            # 3. 重训
            retrain_success, retrain_time, retrain_result = retrain_arch(mode, seed, output_dir, best_arch)

            # 保存结果
            seed_results[mode] = {
                "nas_mrr": best_arch.get("test_mrr", 0),
                "nas_time": nas_time,
                "retrain_mrr": retrain_result.get("test_mrr", 0) if retrain_result else 0,
                "retrain_time": retrain_time,
                "architecture": best_arch["config"]
            }

        all_results[f"seed_{seed}"] = seed_results

    # 保存所有结果
    results_file = os.path.join(BASE_OUTPUT_DIR, "all_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"All results saved to: {results_file}")
    print(f"{'='*80}\n")

    return all_results

if __name__ == "__main__":
    results = run_experiment()
    print("\n✅ Experiment completed!")
    print(f"Results directory: {BASE_OUTPUT_DIR}")
