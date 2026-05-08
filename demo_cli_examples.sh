#!/bin/bash
# 演示脚本：展示 run_comparison_3way.sh 的 CLI 参数功能

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║   📋 run_comparison_3way.sh CLI 参数演示                             ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# 创建演示函数
show_config() {
    local title="$1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "示例 $title"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ════════════════════════════════════════════════════════════════════════════
# 示例 1: 快速测试
# ════════════════════════════════════════════════════════════════════════════
show_config "1: 快速测试（5 分钟）"
echo ""
echo "命令："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --max-events 5000 \\"
echo "    --time-budget 60 \\"
echo "    --trials 10 \\"
echo "    --epochs 1 \\"
echo "    --seeds 42"
echo ""
echo "✓ 配置参数："
cat << 'CONFIG'
  - 数据量：5000 个事件（小规模）
  - 时间预算：60 秒
  - 试验上限：10 个
  - 每 trial 的 epochs：1（快速训练）
  - Seeds：42（单次实验）
  - GPU：自动检测
  - 搜索空间：rnn_only（默认）
  - 输出目录：outputs/comparison_TIMESTAMP
CONFIG

echo ""
echo "✓ 适用场景："
echo "  - 快速验证系统是否正常运行"
echo "  - 测试新参数组合"
echo "  - 开发和调试阶段"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# 示例 2: 小规模对比
# ════════════════════════════════════════════════════════════════════════════
show_config "2: 小规模对比（10-15 分钟）"
echo ""
echo "命令："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --max-events 10000 \\"
echo "    --time-budget 300 \\"
echo "    --trials 20 \\"
echo "    --epochs 2 \\"
echo "    --seeds 42,43"
echo ""
echo "✓ 配置参数："
cat << 'CONFIG'
  - 数据量：10000 个事件
  - 时间预算：300 秒（5 分钟 × 4 个方法）
  - 试验上限：20 个
  - 每 trial 的 epochs：2
  - Seeds：42,43（2 次重复实验）
  - GPU：自动检测
  - 预期总耗时：约 30 分钟（包括初始化和报告生成）
CONFIG

echo ""
echo "✓ 适用场景："
echo "  - 初步验证 Pipeline-Smart 相对于 Data-Parallel 的改进"
echo "  - 小型工作站或 GPU 不足的情况"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# 示例 3: 标准对比（发表级）
# ════════════════════════════════════════════════════════════════════════════
show_config "3: 标准对比（1-2 小时）"
echo ""
echo "命令："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --gpu-list 0,1,2,3,4,5,6,7 \\"
echo "    --max-events 20000 \\"
echo "    --time-budget 600 \\"
echo "    --trials 50 \\"
echo "    --epochs 3 \\"
echo "    --seeds 42,43,44 \\"
echo "    --output-dir outputs/final_results"
echo ""
echo "✓ 配置参数："
cat << 'CONFIG'
  - GPU 列表：0,1,2,3,4,5,6,7（显式指定 8 个 GPU）
  - 数据量：20000 个事件（标准规模）
  - 时间预算：600 秒（10 分钟 × 4 个方法）
  - 试验上限：50 个
  - 每 trial 的 epochs：3
  - Seeds：42,43,44（3 次重复，提高可信度）
  - 输出目录：outputs/final_results
  - 预期总耗时：约 2 小时
CONFIG

echo ""
echo "✓ 适用场景："
echo "  - 严格的科学实验和发表准备"
echo "  - 充分利用所有可用 GPU"
echo "  - 多次种子实验以增加统计可靠性"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# 示例 4: 对比不同搜索空间
# ════════════════════════════════════════════════════════════════════════════
show_config "4: 对比不同搜索空间"
echo ""
echo "命令 1 - RNN-only 搜索空间："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --space rnn_only \\"
echo "    --output-dir outputs/exp_rnn_only"
echo ""
echo "命令 2 - 小搜索空间："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --space small \\"
echo "    --output-dir outputs/exp_small"
echo ""
echo "✓ 特点："
echo "  - 对比不同搜索空间的 NAS 效率"
echo "  - 每个实验输出到独立目录"
echo "  - 便于后续对比分析"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# 示例 5: 部分 GPU 可用
# ════════════════════════════════════════════════════════════════════════════
show_config "5: 部分 GPU 可用"
echo ""
echo "命令 1 - 使用 GPU 0, 2, 4（跳过 1, 3, 5）："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --gpu-list 0,2,4 \\"
echo "    --time-budget 300"
echo ""
echo "命令 2 - 仅使用 GPU 0："
echo "  bash scripts/run_comparison_3way.sh \\"
echo "    --gpu-list 0 \\"
echo "    --time-budget 200"
echo ""
echo "✓ 说明："
echo "  - 灵活支持部分 GPU 场景"
echo "  - 自动检测 GPU 数量，调整 pipeline 阶段数"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# 参数快速参考
# ════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 参数快速参考"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat << 'PARAMS'
参数                  说明                        默认值                示例
────────────────────────────────────────────────────────────────────────────
--gpu-list           GPU ID 列表（逗号分隔）    自动检测            0,1,2,3,4,5,6,7
--space              搜索空间类型               rnn_only            small, large
--dataset            数据集类型                 public_csv          synthetic
--data-file          数据文件路径               data/public/mooc.csv  data/synthetic.csv
--max-events         最大事件数                 20000               5000, 50000
--time-budget        每个方法的时间预算（秒）  1200                60, 300, 600
--epochs             每个 trial 的 epoch 数    3                   1, 5
--trials             试验上限                   999*                10, 20, 50
--seeds              种子列表（逗号分隔）      42,43               42, 42,43,44
--output-dir         输出目录                   outputs/comparison_TS  outputs/my_exp
--help               显示帮助信息               -                   -

*实际由 --time-budget 控制
PARAMS

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CLI 参数功能演示完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 提示："
echo "  1. 查看详细指南：cat docs/CLI_PARAMETERS_GUIDE.md"
echo "  2. 查看脚本帮助：bash scripts/run_comparison_3way.sh --help"
echo "  3. 创建实验脚本记录参数便于复现"
echo ""
