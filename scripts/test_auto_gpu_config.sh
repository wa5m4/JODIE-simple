#!/bin/bash
# ================================================================
# 演示：自动化 GPU worker 分配与配置
# ================================================================
# 用法示例：
#   bash test_auto_gpu_config.sh 0,1,2           # 指定 GPU 0,1,2
#   bash test_auto_gpu_config.sh 0,1,2,3,4,5,6,7 # 指定 GPU 0-7（8GPU）
#   bash test_auto_gpu_config.sh                  # 自动检测所有 GPU
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# 获取 GPU 列表
if [ -n "$1" ]; then
    GPU_LIST="$1"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
    GPU_LIST=$(python -c "print(','.join(str(i) for i in range($NUM_GPUS)))")
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     自动化 GPU Worker 配置测试                             ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  GPU 列表: $GPU_LIST"
echo "║  GPU 数量: $NUM_GPUS"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 小数据集测试（快速验证）
echo "[1] 小数据集自动化配置测试 (synthetic, 快速)"
echo "    - 启用自动化配置"
echo "    - GPU 列表: $GPU_LIST"
echo ""

python search.py \
    --dataset synthetic \
    --num-interactions 1000 \
    --space small \
    --search-mode random \
    --execution-mode ray_pipeline \
    --trials 3 \
    --epochs-per-trial 1 \
    --architectures-per-step 2 \
    --time-budget-sec 120 \
    --gpu-list "$GPU_LIST" \
    --enable-auto-pipeline-config \
    --pipeline-trace \
    --enable-efficiency-monitor \
    --output-dir "outputs/test_auto_config_small"

echo ""
echo "✅ 小数据集测试完成"
echo ""

# 中等数据集测试（演示扩展性）
echo "[2] 中等数据集自动化配置测试 (synthetic, 中等规模)"
echo "    - 启用自动化配置"
echo "    - GPU 列表: $GPU_LIST"
echo ""

python search.py \
    --dataset synthetic \
    --num-interactions 5000 \
    --space small \
    --search-mode random \
    --execution-mode ray_pipeline \
    --trials 3 \
    --epochs-per-trial 1 \
    --architectures-per-step 2 \
    --time-budget-sec 120 \
    --gpu-list "$GPU_LIST" \
    --enable-auto-pipeline-config \
    --pipeline-trace \
    --enable-efficiency-monitor \
    --output-dir "outputs/test_auto_config_medium"

echo ""
echo "✅ 中等数据集测试完成"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║             所有测试完成！                                 ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  输出目录:"
echo "║    - outputs/test_auto_config_small/    (小数据集)"
echo "║    - outputs/test_auto_config_medium/   (中等数据集)"
echo "║"
echo "║  查看效率日志："
echo "║    - efficiency_log_*.csv 文件"
echo "║"
echo "║  查看配置信息："
echo "║    - 搜索时的 [Auto-Config] 日志输出"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
