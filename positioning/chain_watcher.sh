#!/usr/bin/env bash
# chain_watcher.sh — phase5 夜间自动重启守护(独立于 Claude 会话,退出会话仍存活)
#
# 启动方式:
#   nohup bash /home/wanghaoyu/JODIE-simple/positioning/chain_watcher.sh \
#     > /home/wanghaoyu/JODIE-simple/positioning/chain_watcher.log 2>&1 &
#
# 行为:
#   1. 已有链条进程则直接退出(防双启动)。
#   2. 每 10 分钟检查启动条件:1 分钟负载 < 96 且 ≥2 张 GPU 各 ≥4GB 空闲。
#   3. 条件满足即启动 chain_all.sh phase5(B-dp 复测为第 1 个 run)。
#   4. 启动后继续守护:链条异常退出最多自动重启 3 次(链内「全部完成」跳过判定
#      保证续跑安全);正常结束(日志含「链条全部结束」)则退出。
#
# 监控/停止: tail -f positioning/chain_watcher.log;kill $(pgrep -f chain_watcher.sh)

set -u

ROOT=/home/wanghaoyu/JODIE-simple
CHAIN_LOG="$ROOT/positioning/chain_all_p5_restart.log"
LOAD_MAX=96
MIN_FREE_GPUS=2
MIN_GPU_FREE_MB=4096
MAX_RESTARTS=3

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

chain_running() {
    pgrep -f "chain_all.sh phase5|run_all.py" >/dev/null 2>&1
}

gpu_free_count() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -v t="$MIN_GPU_FREE_MB" '$1 >= t {c++} END {print c+0}'
}

conditions_met() {
    local load1
    load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null | cut -d. -f1)
    [ -z "$load1" ] && return 1
    [ "$load1" -ge "$LOAD_MAX" ] && return 1
    local frees
    frees=$(gpu_free_count)
    [ "$frees" -ge "$MIN_FREE_GPUS" ]
}

start_chain() {
    log "启动链条: bash positioning/chain_all.sh phase5(日志 $CHAIN_LOG)"
    cd "$ROOT" || exit 1
    nohup bash positioning/chain_all.sh phase5 >"$CHAIN_LOG" 2>&1 &
    CHAIN_PID=$!
    log "链条 PID=$CHAIN_PID"
}

# ── 阶段 1:防双启动 + 等条件 ──
if chain_running; then
    log "已有链条进程在跑,watcher 退出(防双启动)"
    exit 0
fi
log "watcher 启动:等负载 < $LOAD_MAX 且 ≥$MIN_FREE_GPUS 张 ≥${MIN_GPU_FREE_MB}MB 空闲 GPU"

while ! conditions_met; do
    log "条件未满足(load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null), 空闲GPU=$(gpu_free_count)/8),10 分钟后重查"
    sleep 600
done
log "条件满足(load1=$(awk '{print $1}' /proc/loadavg), 空闲GPU=$(gpu_free_count)/8),启动链条"

# ── 阶段 2:启动 + 守护 ──
restarts=0
start_chain

while true; do
    if kill -0 "$CHAIN_PID" 2>/dev/null; then
        sleep 600
        continue
    fi
    if grep -q "链条全部结束" "$CHAIN_LOG" 2>/dev/null; then
        log "链条正常结束($(grep -c '全部完成' "$CHAIN_LOG" 2>/dev/null || echo 0) 个 run 有完成标志),watcher 退出"
        exit 0
    fi
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        log "链条异常退出超过 $MAX_RESTARTS 次,停止守护,请人工检查 $CHAIN_LOG"
        exit 1
    fi
    log "链条异常退出(第 $restarts/$MAX_RESTARTS 次),10 分钟后重启(链内跳过判定保证续跑)"
    sleep 600
    start_chain
done
