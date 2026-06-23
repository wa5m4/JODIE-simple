#!/bin/bash
# 启动三因素实验 - 后台运行包装脚本

nohup bash run_missing_experiments.sh > run_experiments.out 2>&1 &

echo "实验已在后台启动!"
echo "进程ID: $!"
echo ""
echo "监控进度:"
echo "  tail -f run_experiments.out"
echo ""
echo "查看状态:"
echo "  ps aux | grep run_missing_experiments"
echo ""
echo "完成后查看结果:"
echo "  python analyze_three_factors.py"
