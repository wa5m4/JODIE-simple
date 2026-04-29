#!/bin/bash
# 快速测试脚本：验证命令行参数集成是否工作

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "🔍 测试实时效率监控功能"
echo "================================"

# 1. 检查参数是否正确添加
echo ""
echo "1️⃣  检查 search.py 是否有新参数..."
if python search.py --help 2>&1 | grep -q "enable-efficiency-monitor"; then
    echo "   ✅ --enable-efficiency-monitor 参数已添加"
else
    echo "   ❌ --enable-efficiency-monitor 参数未找到"
    exit 1
fi

if python search.py --help 2>&1 | grep -q "efficiency-monitor-interval"; then
    echo "   ✅ --efficiency-monitor-interval 参数已添加"
else
    echo "   ❌ --efficiency-monitor-interval 参数未找到"
    exit 1
fi

# 2. 检查监控脚本是否存在
echo ""
echo "2️⃣  检查监控脚本..."
if [ -f "tools/monitor_pipeline_efficiency.py" ]; then
    echo "   ✅ tools/monitor_pipeline_efficiency.py 存在"
else
    echo "   ❌ tools/monitor_pipeline_efficiency.py 不存在"
    exit 1
fi

if [ -f "tools/visualize_efficiency_log.py" ]; then
    echo "   ✅ tools/visualize_efficiency_log.py 存在"
else
    echo "   ❌ tools/visualize_efficiency_log.py 不存在"
    exit 1
fi

# 3. 测试参数定义
echo ""
echo "3️⃣  检查参数定义..."
python -c "
import sys
sys.argv = ['search.py', '--help']
try:
    import search
    # 如果能导入就说明没有语法错误
    print('   ✅ search.py 导入成功')
except Exception as e:
    print(f'   ❌ 导入失败: {e}')
    exit(1)
" || exit 1

# 4. 检查 trainer.py 是否正确导入
echo ""
echo "4️⃣  检查 trainer.py 导入..."
python -c "
import nas.trainer
if hasattr(nas.trainer.GraphNASTrainer, 'search_pipeline'):
    print('   ✅ search_pipeline 方法存在')
else:
    print('   ❌ search_pipeline 方法不存在')
    exit(1)
" || exit 1

# 5. 检查是否可以编译
echo ""
echo "5️⃣  检查代码编译..."
python -m py_compile search.py nas/trainer.py && echo "   ✅ 代码编译成功" || exit 1

echo ""
echo "================================"
echo "✅ 所有检查通过！"
echo ""
echo "📝 快速开始命令："
echo ""
echo "CUDA_VISIBLE_DEVICES=0,1,2 python search.py \\"
echo "  --dataset public_csv \\"
echo "  --local-data-path data/public/mooc.csv \\"
echo "  --search-mode rl \\"
echo "  --execution-mode ray_pipeline \\"
echo "  --coarse-trials 3 \\"
echo "  --coarse-epochs 1 \\"
echo "  --architectures-per-step 3 \\"
echo "  --partition-size 2000 \\"
echo "  --pipeline-stage-train-workers 2,1 \\"
echo "  --pipeline-trace \\"
echo "  --enable-efficiency-monitor \\"
echo "  --efficiency-monitor-interval 10 \\"
echo "  --output-dir outputs_test_monitoring"
echo ""
