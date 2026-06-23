#!/usr/bin/env python
"""快速测试data_parallel + tbatch是否正常工作"""
import subprocess
import sys

cmd = [
    "python", "search.py",
    "--execution-mode", "data_parallel",
    "--data-parallel-workers", "2",
    "--batch-mode", "tbatch",
    "--train-batch-size", "32",
    "--dataset", "public_csv",
    "--local-data-path", "data/public/mooc.csv",
    "--max-events", "500",
    "--trials", "1",
    "--epochs-per-trial", "1",
    "--gpu-list", "0,1",
    "--seed", "42",
    "--space", "mixed",
    "--output-dir", "outputs/test_dp_tbatch",
]

print("测试 data_parallel + tbatch...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("❌ 失败:")
    print(result.stderr[-2000:])  # 只打印最后2000字符
    sys.exit(1)
else:
    print("✓ 成功: data_parallel + tbatch 正常工作")
