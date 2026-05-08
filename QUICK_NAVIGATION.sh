#!/bin/bash
# 🗂️ CLI 参数改进 - 快速导航

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║              📚 run_comparison_3way.sh CLI 参数改进 - 快速导航         ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📌 快速开始${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}1. 查看帮助：${NC}"
echo "   bash scripts/run_comparison_3way.sh --help"
echo ""
echo -e "${GREEN}2. 快速测试（5 分钟）：${NC}"
echo "   bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60"
echo ""
echo -e "${GREEN}3. 标准对比（2 小时）：${NC}"
echo "   bash scripts/run_comparison_3way.sh --gpu-list 0,1,2,3,4,5,6,7 --max-events 20000"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📚 主要文档${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

docs=(
    "docs/CLI_PARAMETERS_GUIDE.md|详细 CLI 参数指南（快速开始+完整参数说明+常见示例）"
    "SCRIPT_IMPROVEMENTS.md|脚本改进总结（改进内容+技术细节+迁移指南）"
    "CLI_USAGE_FINAL_SUMMARY.md|完成报告（项目成果+验证结果+最佳实践）"
)

for doc in "${docs[@]}"; do
    IFS='|' read -r file desc <<< "$doc"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
        echo "   📖 $desc"
        echo "   打开: ${YELLOW}cat $file${NC}"
        echo ""
    fi
done

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🎯 脚本和工具${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

scripts=(
    "scripts/run_comparison_3way.sh|改进的三方对比脚本（支持完整 CLI 参数）"
    "demo_cli_examples.sh|参数演示脚本（展示 5 种常见用法）"
    "test_cli_params.sh|参数测试脚本（验证参数解析功能）"
    "check_cli_improvements.sh|改进检查脚本（验证所有改进）"
)

for script in "${scripts[@]}"; do
    IFS='|' read -r file desc <<< "$script"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
        echo "   🔧 $desc"
        echo "   运行: ${YELLOW}bash $file${NC}"
        echo ""
    fi
done

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 参数快速参考${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "基础参数："
echo "  ${YELLOW}--gpu-list${NC}       GPU ID（如 0,1,2,3）- 默认：自动检测"
echo "  ${YELLOW}--max-events${NC}     事件数 - 默认：20000"
echo "  ${YELLOW}--time-budget${NC}    时间预算（秒）- 默认：1200"
echo "  ${YELLOW}--trials${NC}         试验上限 - 默认：999（受时间限制）"
echo "  ${YELLOW}--epochs${NC}         每 trial 的 epoch - 默认：3"
echo "  ${YELLOW}--seeds${NC}          种子列表（如 42,43,44）- 默认：42,43"
echo ""
echo "搜索配置参数："
echo "  ${YELLOW}--space${NC}          搜索空间 - 默认：rnn_only"
echo "  ${YELLOW}--dataset${NC}        数据集 - 默认：public_csv"
echo "  ${YELLOW}--data-file${NC}      数据文件路径 - 默认：data/public/mooc.csv"
echo ""
echo "输出参数："
echo "  ${YELLOW}--output-dir${NC}     输出目录 - 默认：outputs/comparison_TIMESTAMP"
echo ""
echo "其他："
echo "  ${YELLOW}--help${NC}           显示帮助信息"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}💡 使用示例${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

examples=(
    "快速检查|bash scripts/run_comparison_3way.sh --max-events 1000 --time-budget 30 --epochs 1|5 min"
    "快速验证|bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60|10 min"
    "小规模|bash scripts/run_comparison_3way.sh --max-events 10000 --time-budget 300 --seeds 42,43|30 min"
    "标准对比|bash scripts/run_comparison_3way.sh --max-events 20000 --time-budget 600|2 hours"
)

for ex in "${examples[@]}"; do
    IFS='|' read -r name cmd time <<< "$ex"
    echo -e "${YELLOW}$name ($time)：${NC}"
    echo "  $cmd"
    echo ""
done

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🔄 推荐工作流${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Step 1：查看帮助和演示${NC}"
echo "  bash scripts/run_comparison_3way.sh --help"
echo "  bash demo_cli_examples.sh"
echo ""
echo -e "${GREEN}Step 2：快速测试${NC}"
echo "  bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60"
echo ""
echo -e "${GREEN}Step 3：创建实验脚本${NC}"
cat > /tmp/create_exp.sh << 'SCRIPT'
cat > my_experiment.sh << EOF
#!/bin/bash
bash scripts/run_comparison_3way.sh \
    --gpu-list 0,1,2,3,4,5,6,7 \
    --max-events 20000 \
    --time-budget 600 \
    --trials 50 \
    --epochs 3 \
    --seeds 42,43,44
EOF
chmod +x my_experiment.sh
SCRIPT
echo "  # 编辑脚本参数后执行："
echo "  ./my_experiment.sh"
echo ""
echo -e "${GREEN}Step 4：批量运行多个实验${NC}"
echo "  for space in rnn_only small large; do"
echo "    bash scripts/run_comparison_3way.sh --space \$space"
echo "  done"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📖 学习路径${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}初学者：${NC}"
echo "  1. 运行：bash demo_cli_examples.sh"
echo "  2. 读文档：cat docs/CLI_PARAMETERS_GUIDE.md | head -100"
echo "  3. 快速试：bash scripts/run_comparison_3way.sh --help"
echo ""
echo -e "${BLUE}进阶用户：${NC}"
echo "  1. 学参数：cat docs/CLI_PARAMETERS_GUIDE.md"
echo "  2. 看改进：cat SCRIPT_IMPROVEMENTS.md"
echo "  3. 创实验：根据需要组织参数运行实验"
echo ""
echo -e "${BLUE}研究者：${NC}"
echo "  1. 理原理：cat CLI_USAGE_FINAL_SUMMARY.md"
echo "  2. 验功能：bash check_cli_improvements.sh"
echo "  3. 批处理：创建脚本批量运行多组实验"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}❓ 常见问题${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Q: 如何快速开始？${NC}"
echo "A: bash scripts/run_comparison_3way.sh --max-events 5000 --time-budget 60"
echo ""
echo -e "${YELLOW}Q: GPU 列表如何指定？${NC}"
echo "A: --gpu-list 0,1,2,3 或默认自动检测"
echo ""
echo -e "${YELLOW}Q: 如何运行多 seed 实验？${NC}"
echo "A: --seeds 42,43,44 会运行 3 次，每个不同 seed"
echo ""
echo -e "${YELLOW}Q: 结果保存在哪里？${NC}"
echo "A: 默认 outputs/comparison_TIMESTAMP/ 或用 --output-dir 自定义"
echo ""
echo -e "${YELLOW}Q: 需要修改脚本吗？${NC}"
echo "A: 不需要！所有参数通过 CLI 指定"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}✨ 改进亮点${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  ✅ 完整 CLI 参数支持（10+ 参数）"
echo "  ✅ 自动 GPU 检测（无需手动指定）"
echo "  ✅ 智能默认值（最简只需 1 条命令）"
echo "  ✅ 详细文档（26+ 使用示例）"
echo "  ✅ 功能验证（测试脚本 + 演示脚本）"
echo "  ✅ 100% 向后兼容（保持原有逻辑）"
echo "  ✅ 生产就绪（经过完整验证）"
echo ""

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🎉 CLI 参数改进完成！现在可以灵活运行 NAS 对比实验了。               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "更多信息: cat docs/CLI_PARAMETERS_GUIDE.md"
echo ""
