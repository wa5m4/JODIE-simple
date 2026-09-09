#!/usr/bin/env bash
# 定位实验 7 个 run 的顺序执行链条。
# 顺序(2026-09-07 调整,与 configs.py RUNS 一致):B smart→naive→DP → D smart→naive→DP→serial。
#  GPU 分配:B 固定 5,6;D 为 auto——启动前动态选任意 3 张空闲卡(不再绑定 0,1,2)。
#
# 每个 run 启动前:
#   1) GPU 空闲检查:所需 GPU 显存 < 2000MiB 且利用率 < 20%,否则每 5 分钟轮询等待
#      (指南铁律:计时实验不共享 GPU,否则吞吐数字作废)
#   2) 磁盘检查:/home 剩余 > 5GB,否则等待
#   3) apply 配置 → nohup 启动 → 90s 自动预检;预检失败即杀进程并中止整条链
#   4) 等待本 run 进程退出 → ray stop --force → 下一个
#
# 用法(在仓库根目录):
#   nohup bash positioning/chain_all.sh > positioning/chain_all.log 2>&1 &
#   从中途续跑: bash positioning/chain_all.sh 3   (从第 3 个 run 开始)
set -u
cd "$(dirname "$0")/.."

# POS_ALLOW_IDLE=1:接受「0%利用率但持有显存」的卡(与他人空闲作业共存)。
# 计时风险自负:链条会在运行期间监控是否有新作业进入本 run 的卡。
IDLE_MODE="${POS_ALLOW_IDLE:-0}"
MEM_LIMIT=2000
[ "$IDLE_MODE" = "1" ] && MEM_LIMIT=8000

RUN_SPECS=("B smart" "B naive" "B dp" "D smart" "D naive" "D dp" "D serial")
START_IDX="${1:-1}"
if ! [[ "$START_IDX" =~ ^[1-7]$ ]]; then
    echo "用法: chain_all.sh [起始序号 1-7]"
    exit 1
fi

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

gpu_busy_info() {
    # 入参: 逗号分隔 GPU id。全部空闲返回 0;否则打印忙卡信息并返回 1。
    local gpus="$1" g line mem util
    IFS=',' read -ra ids <<< "$gpus"
    local busy=""
    for g in "${ids[@]}"; do
        line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | awk -F, -v id="$g" '$1==id{print $2, $3}')
        if [ -z "$line" ]; then
            say "⚠ GPU $g 不存在或 nvidia-smi 不可用"
            return 1
        fi
        read -r mem util <<< "$line"
        if [ "$mem" -ge "$MEM_LIMIT" ] || [ "$util" -ge 20 ]; then
            busy="$busy GPU$g(显存${mem}MiB/利用率${util}%)"
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
    # 入参: 需要的卡数。返回: 逗号分隔的按卡号升序的前 N 张空闲卡
    # (显存 < 2000MiB 且利用率 < 20%;不足 N 张则返回已有的)。
    local need="$1" free_list=() line g mem util
    while IFS= read -r line; do
        g=$(echo "$line" | cut -d, -f1)
        mem=$(echo "$line" | cut -d, -f2)
        util=$(echo "$line" | cut -d, -f3)
        if [ "$mem" -lt "$MEM_LIMIT" ] && [ "$util" -lt 20 ]; then
            free_list+=("$g")
            [ "${#free_list[@]}" -ge "$need" ] && break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
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

foreign_pids_on_gpus() {
    # 入参: 逗号分隔 GPU id。输出这些卡上属主非 wanghaoyu 的 PID(他人作业,含 root 守护)。
    local gpus="$1" uuid pid idx user
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=',' read -r uuid pid; do
        idx=$(nvidia-smi -L | grep "$uuid" | grep -oP 'GPU \K\d')
        case ",$gpus," in
            *",$idx,"*)
                user=$(ps -o user= -p "$pid" 2>/dev/null)
                [ -n "$user" ] && [ "$user" != "wanghaoyu" ] && echo "$pid"
                ;;
        esac
    done | sort -u
}

say "===== 定位实验链条启动 (从第 $START_IDX 个 run 开始) ====="

for ((i = START_IDX - 1; i < ${#RUN_SPECS[@]}; i++)); do
    read -r CELL STRAT <<< "${RUN_SPECS[$i]}"
    say ""
    say "===== [$((i + 1))/7] cell $CELL / $STRAT ====="

    GPU_LIST=$(cfg_value "GPU_LIST")
    GPU_OVERRIDE=""
    if [ "$GPU_LIST" = "auto" ]; then
        say "动态选卡:等待任意 3 张空闲 GPU..."
        GPU_LIST=$(pick_free_gpus 3)
        while [ "$(echo "$GPU_LIST" | tr ',' '\n' | grep -c .)" -lt 3 ]; do
            say "⚠ 空闲 GPU: [$GPU_LIST] (不足 3 张),继续等待..."
            sleep 300
            GPU_LIST=$(pick_free_gpus 3)
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

    LOG="run_pos_cell${CELL}_${STRAT}.log"
    # 关键修复:ray_pipeline.py 不自行设置 CUDA_VISIBLE_DEVICES(只有 data_parallel 会),
    # 必须在此限定可见卡,否则 pipeline worker 直连物理 cuda:0/1(2026-09-07 B-smart 首跑事故)。
    nohup env PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        python run_all.py > "$LOG" 2>&1 &
    PID=$!
    say "已启动 PID=$PID, CUDA_VISIBLE_DEVICES=$GPU_LIST, log=$LOG"

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

    # 等待本 run 结束;IDLE 模式下同时监控是否有新作业进入本 run 的卡
    BASE_FOREIGN=$(foreign_pids_on_gpus "$GPU_LIST" | tr '\n' ' ')
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60
        if [ "$IDLE_MODE" = "1" ]; then
            NOW_FOREIGN=$(foreign_pids_on_gpus "$GPU_LIST" | tr '\n' ' ')
            NEW=$(comm -13 <(echo "$BASE_FOREIGN" | tr ' ' '\n' | sort -u) \
                          <(echo "$NOW_FOREIGN" | tr ' ' '\n' | sort -u) | tr -d '\n')
            if [ -n "$NEW" ]; then
                say "⚠⚠ 污染警报:检测到新作业进入本 run 的 GPU (PID: $NEW),本 run 计时可能作废"
            fi
        fi
    done
    if grep -q "全部完成" "$LOG"; then
        say "✓ 本 run 正常结束"
    else
        say "⚠ 进程退出但 log 中未见'全部完成',检查 $LOG"
    fi

    ray stop --force > /dev/null 2>&1 || true
    sleep 10
done

say ""
say "===== 链条全部结束 ====="
for spec in "${RUN_SPECS[@]}"; do
    read -r C S <<< "$spec"
    L="run_pos_cell${C}_${S}.log"
    DONE=$(grep -c "全部完成" "$L" 2>/dev/null || true)
    say "$spec → $L (完成标志: ${DONE:-0})"
done
