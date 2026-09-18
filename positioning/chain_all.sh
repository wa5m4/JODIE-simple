#!/usr/bin/env bash
# 定位实验 run 的顺序执行链条(Phase 1 默认;`phase2` 跑 E/F;`phase3` 跑五臂合并)。
# Phase 1 顺序(2026-09-07 调整,与 configs.py RUNS 一致):B smart→naive→DP → D smart→naive→DP→serial。
# Phase 2(2026-09-09,configs.py RUNS_P2):E smart→naive→DP → F smart→naive→DP→serial。
# Phase 3(2026-09-09 五臂合并,configs.py RUNS_P3):DP 修复复测 + smart_sync + naive_alloc + F/C 全臂。
#  GPU 分配:全部 cell 均为 auto——启动前动态选所需张数的空闲卡(不绑定具体卡号)。
#
# 每个 run 启动前:
#   1) GPU 空闲检查(2026-09-15 用户裁定放宽):利用率 < 20% 且空闲显存 ≥ 4GB 即视为可用;
#      他人进程只占显存不计算(0% 利用率)不影响计时,不等待;不满足则每 5 分钟轮询
#   2) 磁盘检查:/home 剩余 > 5GB,否则等待
#   3) apply 配置 → nohup 启动 → 90s 自动预检;预检失败即杀进程并中止整条链
#   4) 等待本 run 进程退出;期间 pmon(2s 采样)盯本 run 卡上的他人进程:
#      只有 sm 利用率 > 15% 的「真实计算」才警报 + 在日志标注(⚠)
#      (2026-09-18 用户裁定:取消污染自动重跑;全部 run 只跑一遍+标记,终检统一补跑)
#   5) 已完成的干净 run(日志含"全部完成")自动跳过,续跑无需手算起始序号
#
# 用法(在仓库根目录):
#   nohup bash positioning/chain_all.sh > positioning/chain_all.log 2>&1 &
#   Phase 2:  nohup bash positioning/chain_all.sh phase2 > positioning/chain_all_p2.log 2>&1 &
#   Phase 3:  nohup bash positioning/chain_all.sh phase3 > positioning/chain_all_p3.log 2>&1 &
#   从中途续跑: bash positioning/chain_all.sh 3          (Phase 1 从第 3 个 run 开始)
#                bash positioning/chain_all.sh phase2 4  (Phase 2 从第 4 个 run 开始)
#   Phase 4/5(2026-09-15 保真修复后全量重跑):
#   phase4 = B 三臂金丝雀(serial/naive/naive_async,位级验证门);phase5 = 全矩阵其余 29 run
set -u
cd "$(dirname "$0")/.."

# 空闲判定(2026-09-10 用户裁定,放宽):0% 利用率只占显存的作业不影响计时,
# 不视为忙;只有「真实计算」才算污染(见 active_foreign_on_gpus)。
UTIL_LIMIT=20          # 利用率 ≥20% 视为忙
MEM_FREE_NEED=4096     # 空闲显存不足 4GB 视为忙(2026-09-15 用户批准放宽)
ACTIVE_SM_THRESH=15    # 他人进程 sm 利用率 >15% 判为活跃污染

PHASE=1
START_IDX="${1:-1}"
if [ "${1:-}" = "phase2" ]; then
    PHASE=2
    START_IDX="${2:-1}"
elif [ "${1:-}" = "phase3" ]; then
    PHASE=3
    START_IDX="${2:-1}"
elif [ "${1:-}" = "phase4" ]; then
    PHASE=4
    START_IDX="${2:-1}"
elif [ "${1:-}" = "phase5" ]; then
    PHASE=5
    START_IDX="${2:-1}"
fi

# 运行清单以 configs.py 为准(RUNS / RUNS_P2 / RUNS_P3 / RUNS_P4 / RUNS_P5),避免脚本与配置表漂移
RUN_SPECS=()
while IFS= read -r line; do
    RUN_SPECS+=("$line")
done < <(POS_PHASE=$PHASE python - <<'PY'
import os, sys
sys.path.insert(0, "positioning")
from configs import RUNS, RUNS_P2, RUNS_P3, RUNS_P4, RUNS_P5
RUN_LISTS = {"1": RUNS, "2": RUNS_P2, "3": RUNS_P3, "4": RUNS_P4, "5": RUNS_P5}
for r in RUN_LISTS[os.environ["POS_PHASE"]]:
    print(r["cell"], r["strategy"], r["log"])
PY
)
N_RUNS=${#RUN_SPECS[@]}
if ! [[ "$START_IDX" =~ ^[0-9]+$ ]] || [ "$START_IDX" -lt 1 ] || [ "$START_IDX" -gt "$N_RUNS" ]; then
    echo "用法: chain_all.sh [phase2] [起始序号 1-$N_RUNS]"
    exit 1
fi

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

gpu_busy_info() {
    # 入参: 逗号分隔 GPU id。全部可用返回 0;否则打印忙卡信息并返回 1。
    # 判定(2026-09-10 放宽):利用率 < 20% 且空闲显存 ≥ 10GB 即视为可用——
    # 他人进程只占显存不计算(0% 利用率)不影响计时,不算忙。
    local gpus="$1" g line mem total util free
    IFS=',' read -ra ids <<< "$gpus"
    local busy=""
    for g in "${ids[@]}"; do
        line=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | awk -F, -v id="$g" '$1==id{print $2, $3, $4}')
        if [ -z "$line" ]; then
            say "⚠ GPU $g 不存在或 nvidia-smi 不可用"
            return 1
        fi
        read -r mem total util <<< "$line"
        free=$((total - mem))
        if [ "$util" -ge "$UTIL_LIMIT" ] || [ "$free" -lt "$MEM_FREE_NEED" ]; then
            busy="$busy GPU$g(利用率${util}%/空闲显存${free}MiB)"
        fi
    done
    if [ -n "$busy" ]; then
        say "⚠ 以下 GPU 忙,等待中:$busy"
        return 1
    fi
    return 0
}

wait_gpus_free() {
    local gpus="$1"
    until gpu_busy_info "$gpus"; do sleep 300; done
    say "✓ 所需 GPU [$gpus] 已空闲"
}

pick_free_gpus() {
    # 入参: 需要的卡数。返回: 逗号分隔的按卡号升序的前 N 张可用卡
    # (利用率 < 20% 且空闲显存 ≥ 10GB;不足 N 张则返回已有的)。
    local need="$1" free_list=() line g mem total util free
    while IFS= read -r line; do
        g=$(echo "$line" | cut -d, -f1)
        mem=$(echo "$line" | cut -d, -f2)
        total=$(echo "$line" | cut -d, -f3)
        util=$(echo "$line" | cut -d, -f4)
        free=$((total - mem))
        if [ "$util" -lt "$UTIL_LIMIT" ] && [ "$free" -ge "$MEM_FREE_NEED" ]; then
            free_list+=("$g")
            [ "${#free_list[@]}" -ge "$need" ] && break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')
    local IFS=,
    echo "${free_list[*]}"
}

wait_disk_free() {
    until [ "$(df -P -B1M /home | awk 'NR==2{print $4}')" -ge 5000 ]; do
        say "⚠ /home 剩余不足 5GB,等待中..."
        sleep 300
    done
    say "✓ 磁盘剩余 $(df -h /home | awk 'NR==2{print $4}')"
}

cfg_value() {
    python - "$CELL" "$STRAT" "$1" <<'PY'
import sys
sys.path.insert(0, "positioning")
from configs import get_config
cfg = get_config(sys.argv[1], sys.argv[2])
print(cfg[sys.argv[3]])
PY
}

gpu_count_value() {
    python - "$CELL" <<'PY'
import sys
sys.path.insert(0, "positioning")
from configs import gpu_count_needed
print(gpu_count_needed(sys.argv[1]))
PY
}

active_foreign_on_gpus() {
    # 入参: 逗号分隔 GPU id、pmon 输出文件。
    # 输出: 本 run 卡上「真实计算」(sm 利用率 > 阈值)且属主非 wanghaoyu 的 PID。
    # 仅占显存不计算的进程(0% 利用率)不算污染(2026-09-10 用户裁定)。
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

say "===== 定位实验链条启动 (Phase $PHASE, 从第 $START_IDX 个 run 开始) ====="

DONE_LOGS=()
for ((i = START_IDX - 1; i < ${#RUN_SPECS[@]}; i++)); do
    read -r CELL STRAT LOGNAME <<< "${RUN_SPECS[$i]}"
    DONE_LOGS+=("$LOGNAME")
    say ""
    say "===== [$((i + 1))/$N_RUNS] cell $CELL / $STRAT ====="

    # 已完成的干净 run 自动跳过(2026-09-10):续跑不用手算起始序号;
    # 受污染的 run 会被改名 *_polluted 移走,不会误跳过。
    # 2026-09-12:run_all.py 结尾横幅「全部完成」在策略失败时也打印,
    # 跳过判定必须同时要求无「失败策略:」行(C cell 四个 pipeline 策略崩溃教训)。
    if grep -q "全部完成" "$LOGNAME" 2>/dev/null && ! grep -q "失败策略:" "$LOGNAME" 2>/dev/null; then
        say "✓ 跳过已完成的干净 run($LOGNAME 有完成标志且无失败策略)"
        continue
    fi

    GPU_LIST=$(cfg_value "GPU_LIST")
    GPU_OVERRIDE=""
    if [ "$GPU_LIST" = "auto" ]; then
        NEED=$(gpu_count_value)
        say "动态选卡:等待任意 $NEED 张空闲 GPU..."
        GPU_LIST=$(pick_free_gpus "$NEED")
        while [ "$(echo "$GPU_LIST" | tr ',' '\n' | grep -c .)" -lt "$NEED" ]; do
            say "⚠ 空闲 GPU: [$GPU_LIST] (不足 $NEED 张),继续等待..."
            sleep 300
            GPU_LIST=$(pick_free_gpus "$NEED")
        done
        GPU_OVERRIDE="$GPU_LIST"
        say "✓ 动态选中 GPU [$GPU_LIST]"
    else
        wait_gpus_free "$GPU_LIST"
    fi
    wait_disk_free

    if [ -n "$GPU_OVERRIDE" ]; then
        python positioning/apply_config.py --cell "$CELL" --strategy "$STRAT" \
            --gpu-list "$GPU_LIST" --apply || { say "✗ 配置写入失败,中止链条"; exit 1; }
    else
        python positioning/apply_config.py --cell "$CELL" --strategy "$STRAT" \
            --apply || { say "✗ 配置写入失败,中止链条"; exit 1; }
    fi

    LOG="$LOGNAME"
    # 关键修复:ray_pipeline.py 不自行设置 CUDA_VISIBLE_DEVICES(只有 data_parallel 会),
    # 必须在此限定可见卡,否则 pipeline worker 直连物理 cuda:0/1(2026-09-07 B-smart 首跑事故)。
    nohup env PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        python run_all.py > "$LOG" 2>&1 &
    PID=$!
    say "已启动 PID=$PID, CUDA_VISIBLE_DEVICES=$GPU_LIST, log=$LOG"
    # 启动负载快照(2026-09-09 起):run 刚启动、GPU 尚未加载时的机器状态
    {
        echo ""
        echo "── 负载快照 (run 启动, $(date '+%m-%d %H:%M:%S')) ──"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    } >> "$LOG"

    sleep 90
    if [ -n "$GPU_OVERRIDE" ]; then
        PRE_ARGS="--gpu-list $GPU_LIST"
    else
        PRE_ARGS=""
    fi
    if python positioning/precheck.py --cell "$CELL" --strategy "$STRAT" --log "$LOG" $PRE_ARGS; then
        say "✓ 预检通过"
    else
        say "✗ 预检失败,杀进程 PID=$PID 并中止链条(看 $LOG)"
        kill "$PID" 2>/dev/null
        exit 1
    fi

    # pmon 全程盯梢(2026-09-10):每 2s 采样本机全部卡的进程 sm 利用率,供污染裁决
    PMON_LOG=$(mktemp /tmp/pmon_${CELL}_${STRAT}_XXXXXX.log)
    nvidia-smi pmon -d 2 > "$PMON_LOG" 2>&1 &
    PMON_PID=$!

    # 等待本 run 结束;全程盯本 run 卡上「真实计算」的他人进程
    # (2026-09-10 用户裁定:0% 利用率只占显存的不算污染、不警报;sm>阈值 才算)
    POLLUTED=0
    ALERTED=""
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60
        ACTIVE=$(active_foreign_on_gpus "$GPU_LIST" "$PMON_LOG")
        if [ -n "$ACTIVE" ]; then
            NEW=$(comm -13 <(echo "$ALERTED" | tr ' ' '\n' | sort -u) \
                          <(echo "$ACTIVE" | tr ' ' '\n' | sort -u) | tr -d '\n')
            if [ -n "$NEW" ]; then
                say "⚠⚠ 污染警报:本 run 卡上出现活跃计算的他人进程 PID=$NEW (sm>${ACTIVE_SM_THRESH}%),本 run 计时可能作废"
                POLLUTED=1
                ALERTED="$ALERTED $NEW"
            fi
        fi
    done
    kill "$PMON_PID" 2>/dev/null || true
    rm -f "$PMON_LOG"
    if grep -q "失败策略:" "$LOG"; then
        say "✗✗ 本 run 存在失败策略(代码级崩溃,不是污染),中止链条 — 看 $LOG"
        exit 1
    elif grep -q "全部完成" "$LOG"; then
        say "✓ 本 run 正常结束"
    else
        say "⚠ 进程退出但 log 中未见'全部完成',检查 $LOG"
    fi

    # 结束负载快照(2026-09-09 起):记录本 run 结束时机器负载与在场作业,供异常判读
    {
        echo ""
        echo "── 负载快照 (run 结束, $(date '+%m-%d %H:%M:%S')) ──"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        echo "-- compute-apps --"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
    } >> "$LOG"

    ray stop --force > /dev/null 2>&1 || true
    sleep 10

    # 污染裁决(2026-09-18 用户裁定改):不再自动重跑。有污染只在日志标注(⚠),
    # 数据/计时照记照填,终检时对误差不可容忍的 run 统一补跑。
    if [ "$POLLUTED" -eq 1 ]; then
        say "⚠ 污染标注:本 run 期间有活跃他人进程(按新政策不重跑,标记 ⚠;终检时定夺是否补跑)"
        echo "[污染标注 $(date '+%m-%d %H:%M:%S')] 本 run 期间检测到活跃他人进程(sm>15%),按 2026-09-18 政策不重跑;计时可能偏差,终检时校验。" >> "$LOG"
    else
        say "✓ 本 run 干净(仅占显存不计算的作业不计污染)"
    fi
done

say ""
say "===== 链条全部结束 (Phase $PHASE) ====="
for L in "${DONE_LOGS[@]}"; do
    DONE=$(grep -c "全部完成" "$L" 2>/dev/null || true)
    say "$L (完成标志: ${DONE:-0})"
done
