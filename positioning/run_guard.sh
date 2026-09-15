#!/usr/bin/env bash
# 手动启动 run 的守护进程:复刻 chain_all.sh 的 pmon 盯梢纪律(2026-09-15 并行方案)。
# 用法: bash positioning/run_guard.sh <CELL> <STRAT> <GPU_LIST> <RUN_PID> <LOG>
set -u
CELL=$1; STRAT=$2; GPU_LIST=$3; RUN_PID=$4; LOG=$5
ACTIVE_SM_THRESH=15

active_foreign_on_gpus() {
    local gpus="$1" pmon_file="$2" idx pid sm user
    awk -v gpus="$gpus" -v thresh="$ACTIVE_SM_THRESH" '
        $1 ~ /^[0-9]+$/ && $3 == "C" && $4 ~ /^[0-9]+$/ && $4 + 0 > thresh {
            n = split(gpus, arr, ",")
            for (k = 1; k <= n; k++) if (arr[k] == $1) { print $1, $2, $4; break }
        }' "$pmon_file" 2>/dev/null | while read -r idx pid sm; do
        user=$(ps -o user= -p "$pid" 2>/dev/null)
        [ -n "$user" ] && [ "$user" != "wanghaoyu" ] && echo "$pid"
    done | sort -u
}

PMON_LOG=$(mktemp /tmp/pmon_${CELL}_${STRAT}_XXXXXX.log)
nvidia-smi pmon -d 2 > "$PMON_LOG" 2>&1 &
PMON_PID=$!

POLLUTED=0
ALERTED=""
while kill -0 "$RUN_PID" 2>/dev/null; do
    sleep 60
    ACTIVE=$(active_foreign_on_gpus "$GPU_LIST" "$PMON_LOG")
    if [ -n "$ACTIVE" ]; then
        NEW=$(comm -13 <(echo "$ALERTED" | tr ' ' '\n' | sort -u) \
                      <(echo "$ACTIVE" | tr ' ' '\n' | sort -u) | tr -d '\n')
        if [ -n "$NEW" ]; then
            echo "[$(date '+%m-%d %H:%M:%S')] ⚠⚠ 污染警报:本 run 卡上出现活跃计算的他人进程 PID=$NEW (sm>${ACTIVE_SM_THRESH}%),本 run 计时可能作废" | tee -a "$LOG"
            POLLUTED=1
            ALERTED="$ALERTED $NEW"
        fi
    fi
done
kill "$PMON_PID" 2>/dev/null || true
rm -f "$PMON_LOG"

# 结束负载快照(与链同款)
{
    echo ""
    echo "── 负载快照 (run 结束, $(date '+%m-%d %H:%M:%S')) ──"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo "-- compute-apps --"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
} >> "$LOG"

if grep -q "失败策略:" "$LOG"; then
    echo "[guard] ✗✗ 本 run 存在失败策略(代码级崩溃) — 看 $LOG"
elif grep -q "全部完成" "$LOG"; then
    echo "[guard] ✓ 本 run 正常结束"
    [ "$POLLUTED" -eq 1 ] && echo "[guard] ⚠ 但有污染警报(重跑裁决需人工确认)"
else
    echo "[guard] ⚠ 进程退出但 log 中未见'全部完成',检查 $LOG"
fi
[ "$POLLUTED" -eq 1 ] && echo "GUARD_VERDICT=POLLUTED" || echo "GUARD_VERDICT=CLEAN"
