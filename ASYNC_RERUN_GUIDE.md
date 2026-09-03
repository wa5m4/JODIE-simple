# 异步机制复证 · 服务器重跑说明(pipeline_smart 持久化池)

**实验名称**:异步架构生成+训练机制(pipeline_smart)在"修复全开 + 保真协议"下的复证
**目的**:把 2026-08-13 的成功数据(FINAL_REPORT:四策略架构一致、test 0.8561、Smart 加速 2.3×、总耗时 11,112s)在**当前代码 + 保真修复全开**的条件下重新跑出来,证明三点:

1. **质量不降**:与串行/naive 基线同样收敛到 133K 架构、test MRR ≈ 0.8561(噪声地板内)
2. **加速成立**:总耗时 ≈ 3.1h vs naive ≈ 4.6h(粗搜阶段参考 2.6×)
3. **RL 路径稳定**:历史上 RL controller 在异步路径曾崩溃(inplace 梯度冲突),flush 修复后 08-13 全量 50-trial 成功;本次确认最新代码上依然稳定

**日期**:2026-08-26

---

## 0. 唯一变量

- **唯一变量 = 搜索策略**:`pipeline_smart`(异步持久化池,1 stage × 3 workers)
- 对比基准(已有,不用重跑):`pipeline_naive`(同步批,3 stage × 1 worker)的两次修复全开运行 → `results/20260811_204240` / `results/20260825_133519`(133K / 0.8561,约 4.6h)
- **BATCH_MODE 保持 `serial`**(与基线一致):本次不测批处理模式,只测策略差异
- 其余与五档消融协议一致:rnn_only、50×2 + 8×5、MOOC(public_csv / 20000 事件)、seed 42、partition 2000、GPU 0,1,2、FEATURE_DIM=4、NEG_SAMPLE_SIZE=5、SELECTION_METRIC=mrr、SEARCH_MODE=rl
- Smart 配置与 08-13 成功运行一致:**手动**分配(`SMART_ENABLE_AUTO_PIPELINE_CONFIG=False`,1 stage × 3 workers)——智慧分配(自动分配)已证伪,不开

## 1. 分支与改动

- **分支**:`ablation/smart-async`(基于 ablation/stale-batch,含全部保真修复与负样本同源修复)
- 相对基线的 run_all.py 改动仅 2 处:`ENABLE_STRATEGIES = ["pipeline_smart"]`、`BATCH_MODE = "serial"`

## 2. 服务器步骤

```bash
cd <项目目录>
git fetch origin
git checkout ablation/smart-async
git pull

# 第 1 步:预检(约 12 分钟)——验证 RL+异步 flush 路径在最新代码上不崩溃
python pipeline_analysis/test_smart_flush.py
#   期望:5 trials × 2 arch/step 跑完,best≈0.79,无崩溃/无 inplace 报错
#   若报错:停下来,把报错贴给 Claude,先别跑全量

# 第 2 步:全量 50-trial 搜索(约 3.1 小时)
nohup PYTHONIOENCODING=utf-8 python run_all.py > run_smart_async_rerun.log 2>&1 &
```

- 预检(启动后 1 分钟内看 `run_smart_async_rerun.log`):应显示 `策略 1/1: Pipeline Smart`、`搜索模式: rl`、`BATCH_MODE=serial`、rnn_only、seed 42 等。
- 参考时长:08-13 全量 50-trial = 11,112s(3.1h)。

## 3. 跑完带回来的东西

1. `results/<时间戳>/pipeline_smart/pipeline_smart/summary.txt`
2. `results/<时间戳>/pipeline_smart/pipeline_smart/best_arch.json`
3. `results/<时间戳>/pipeline_smart/pipeline_smart/leaderboard.csv`
4. `results/<时间戳>/pipeline_smart/timing_log.csv` ← **关键**:每个 trial 的开始/结束时间,异步重叠的直接证据
5. `run_smart_async_rerun.log`(尾部)
6. 一句话:**总耗时 / test_score / 最终参数量 / 结果目录时间戳**

## 4. 判读(Claude 拿到数据后计算)

- **第一层·终点**:best arch 是否还是 133K 家族、test MRR 是否在 0.8561 ± 噪声地板(mean|d|=0.0069 / max|d|=0.0887)内 → 异步不伤搜索质量
- **第二层·轨迹**:leaderboard 分布 vs 基线目录(20260811_204240 / 20260825_133519)
- **第三层·计时**:timing_log.csv 中相邻 trial 的 start/end 时间重叠 → 证明"架构生成与训练在时间上重叠"(异步机制的实锤);总耗时 vs naive 约 4.6h
- 顺带确认:RL controller 路径全程无 inplace 崩溃

## 5. 跑完恢复

```bash
git checkout ablation/stale-batch   # 回双臂消融分支
# 或 git checkout refactored        # 回主线
```

## 6. 时间线备忘

- 08-13:FINAL_REPORT 四策略一致(PipeSmart 11,112s / 0.8561 / 2.3×),但当时修复协议与现在不同(CE 损失、修复未全开)
- 08-24 起:五档消融 + 双臂消融均以 pipeline_naive 为唯一载体
- 本次(2026-08-26):smart 异步机制在保真协议下复证 → 三点全立;2026-08-27 已定稿为独立贡献 C3(异步架构生成+训练,与 C1 保真度系统、C2 pipeline 并列)
