#!/bin/bash
# 运行多种子实验并生成总结

set -e

echo "=========================================="
echo "多种子实验：Serial vs Data Parallel vs Pipeline"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 种子: 20042, 12345, 67890"
echo "  - 数据量: 20000"
echo "  - Trials: 27"
echo "  - Epochs: 3"
echo "  - GPUs: 0,1,2"
echo "  - 评估: 在线模式 (frozen=False)"
echo ""

# 运行实验
echo "Step 1: 运行NAS搜索和重训..."
python run_multi_seed_experiment.py

# 分析结果
echo ""
echo "Step 2: 分析结果并生成报告..."
python analyze_multi_seed_results.py

# 显示报告
echo ""
echo "=========================================="
echo "实验完成！查看报告:"
echo "  outputs/multi_seed_experiment/MULTI_SEED_SUMMARY.md"
echo "=========================================="
