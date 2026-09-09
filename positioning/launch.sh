#!/usr/bin/env bash
# 单个 run 启动器:apply 配置 → nohup 启动 → 90s 自动预检 → 失败则杀进程。
# 用法(在仓库根目录): bash positioning/launch.sh <cell: D|B> <strategy: smart|naive|dp|serial> [gpu-list]
#   第 3 个参数可选:手动指定 GPU(如 cell D 是 auto,必须给,例如 "5,6,7")。
# 注意:本脚本不做 GPU 空闲检查(chain_all.sh 会做),手动启动前请确认所需 GPU 空闲。
set -u
cd "$(dirname "$0")/.."

CELL="${1:?用法: launch.sh <cell> <strategy> [gpu-list]}"
STRAT="${2:?}"
GPU_LIST="${3:-}"

echo "== [launch] 应用配置 cell $CELL / $STRAT =="
if [ -n "$GPU_LIST" ]; then
    python positioning/apply_config.py --cell "$CELL" --strategy "$STRAT" \
        --gpu-list "$GPU_LIST" --apply || exit 1
else
    python positioning/apply_config.py --cell "$CELL" --strategy "$STRAT" --apply || exit 1
    GPU_LIST=$(python - "$CELL" "$STRAT" <<'PY'
import sys
sys.path.insert(0, "positioning")
from configs import get_config
print(get_config(sys.argv[1], sys.argv[2])["GPU_LIST"])
PY
)
    if [ "$GPU_LIST" = "auto" ]; then
        echo "== [launch] cell D 需动态选卡,请加第 3 个参数或改用 chain_all.sh =="
        exit 1
    fi
fi

LOG="run_pos_cell${CELL}_${STRAT}.log"
echo "== [launch] nohup 启动: CUDA_VISIBLE_DEVICES=$GPU_LIST PYTHONIOENCODING=utf-8 python run_all.py > $LOG =="
# 关键修复:ray_pipeline.py 不自行设置 CUDA_VISIBLE_DEVICES,必须在此限定可见卡
nohup env PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    python run_all.py > "$LOG" 2>&1 &
PID=$!
echo "== [launch] PID=$PID =="

sleep 90

echo "== [launch] 90s 自动预检 =="
if python positioning/precheck.py --cell "$CELL" --strategy "$STRAT" \
    --log "$LOG" --gpu-list "$GPU_LIST"; then
    echo "== [launch] 预检通过,运行继续。查看: tail -f $LOG =="
else
    echo "== [launch] 预检失败,终止 PID=$PID(完整输出见 $LOG) =="
    kill "$PID" 2>/dev/null
    exit 1
fi
