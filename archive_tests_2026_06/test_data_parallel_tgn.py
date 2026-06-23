#!/usr/bin/env python
"""验证Data Parallel在TGN模式下真正做数据并行"""
import subprocess
import json
import sys

# 共用参数
DATASET = "public_csv"
LOCAL_DATA_PATH = "data/public/mooc.csv"
MAX_EVENTS = 2000
TRIALS = 3
EPOCHS = 1
GPU_LIST = "0,1,2"
SEED = 42
BATCH_MODE = "tgn"
TGN_LOSS_MODE = "last"
TGN_WINDOW_SIZE = 10.0

print("="*70)
print("验证Data Parallel在TGN模式下是否真正做数据并行")
print("="*70)
print(f"数据集: {DATASET} ({LOCAL_DATA_PATH})")
print(f"事件数: {MAX_EVENTS}, Trials: {TRIALS}, Epochs: {EPOCHS}")
print(f"GPU: {GPU_LIST}, Seed: {SEED}")
print(f"Batch mode: {BATCH_MODE}, TGN loss mode: {TGN_LOSS_MODE}")
print("="*70)
print()

# 测试1: Serial + TGN
print("[Test 1/2] Serial + TGN")
print("-"*70)
cmd1 = [
    "python", "search.py",
    "--execution-mode", "serial",
    "--batch-mode", BATCH_MODE,
    "--tgn-loss-mode", TGN_LOSS_MODE,
    "--tgn-window-size", str(TGN_WINDOW_SIZE),
    "--dataset", DATASET,
    "--local-data-path", LOCAL_DATA_PATH,
    "--max-events", str(MAX_EVENTS),
    "--trials", str(TRIALS),
    "--epochs-per-trial", str(EPOCHS),
    "--gpu-list", GPU_LIST,
    "--seed", str(SEED),
    "--space", "mixed",
    "--output-dir", "outputs/verify_dp_serial_tgn",
]
result1 = subprocess.run(cmd1, capture_output=True, text=True)
if result1.returncode != 0:
    print(f"❌ Serial + TGN 失败:")
    print(result1.stderr)
    sys.exit(1)

# 读取结果
with open("outputs/verify_dp_serial_tgn/best_arch.json") as f:
    serial_result = json.load(f)
serial_mrr = serial_result["test_mrr"]
serial_recall = serial_result["test_recall_at_k"]
print(f"✓ Serial + TGN: MRR={serial_mrr:.6f}, Recall@10={serial_recall:.6f}")
print()

# 测试2: Data Parallel + TGN
print("[Test 2/2] Data Parallel + TGN")
print("-"*70)
cmd2 = [
    "python", "search.py",
    "--execution-mode", "data_parallel",
    "--data-parallel-workers", "3",
    "--batch-mode", BATCH_MODE,
    "--tgn-loss-mode", TGN_LOSS_MODE,
    "--tgn-window-size", str(TGN_WINDOW_SIZE),
    "--dataset", DATASET,
    "--local-data-path", LOCAL_DATA_PATH,
    "--max-events", str(MAX_EVENTS),
    "--trials", str(TRIALS),
    "--epochs-per-trial", str(EPOCHS),
    "--gpu-list", GPU_LIST,
    "--seed", str(SEED),
    "--space", "mixed",
    "--output-dir", "outputs/verify_dp_parallel_tgn",
]
result2 = subprocess.run(cmd2, capture_output=True, text=True)
if result2.returncode != 0:
    print(f"❌ Data Parallel + TGN 失败:")
    print(result2.stderr)
    sys.exit(1)

# 读取结果
with open("outputs/verify_dp_parallel_tgn/best_arch.json") as f:
    dp_result = json.load(f)
dp_mrr = dp_result["test_mrr"]
dp_recall = dp_result["test_recall_at_k"]
print(f"✓ Data Parallel + TGN: MRR={dp_mrr:.6f}, Recall@10={dp_recall:.6f}")
print()

# 比较结果
print("="*70)
print("结果对比")
print("="*70)
print(f"Serial + TGN:        MRR={serial_mrr:.6f}, Recall@10={serial_recall:.6f}")
print(f"Data Parallel + TGN: MRR={dp_mrr:.6f}, Recall@10={dp_recall:.6f}")
print(f"MRR差异: {abs(serial_mrr - dp_mrr):.6f}")
print()

if abs(serial_mrr - dp_mrr) < 1e-10:
    print("❌ 失败: 两个模式的MRR完全相同（精确到10位小数）")
    print("   Data Parallel可能仍在回退到Serial模式")
    sys.exit(1)
else:
    print("✓ 成功: 两个模式的MRR不同，Data Parallel正在做真正的数据并行")
    print(f"  差异: {abs(serial_mrr - dp_mrr):.6f} ({abs(serial_mrr - dp_mrr)/serial_mrr*100:.2f}%)")
