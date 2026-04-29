"""
Pipeline Efficiency Metrics - Detailed Explanation

本文档解释 analyze_pipeline_efficiency.py 中所有指标的计算方法
"""

# ============================================================================
# 1. GPU 利用率 (GPU Utilization Ratio)
# ============================================================================

"""
基础概念：
- 每个 dispatch/complete 事件对代表一个 GPU actor 的活动
- 关键是：在整个 pipeline 执行期间，有多少个 GPU 同时在工作？

计算步骤：
1. 从 trace 日志中提取所有 dispatch/complete 事件对
2. 对每个 (trial, stage) 对，记录 start_time (dispatch 时刻) 和 end_time (complete 时刻)
3. 在整个时间范围内均匀采样（每 0.1s 一次）
4. 统计每个采样点有多少个 GPU actor 在活动（start_time ≤ sample_time < end_time）
5. 计算平均 concurrent actors 和最大 concurrent actors

举例（从你的 trace）：
   dispatch trial=0 stage=1 elapsed=5.202s
   ...处理 2000 个事件，约 23.224s...
   complete trial=0 stage=1 elapsed=28.426s
   
   dispatch trial=1 stage=1 elapsed=5.206s
   
   这意味着 t=5.2s 时，有 1 个 actor (trial_0_stage_1) 开始
           t=5.2s 时，又有 1 个 actor (trial_1_stage_1) 开始 → 2 个同时活动
           但随后 trial_1_stage_1 被阻塞，只有 trial_0_stage_1 继续
           t=28.4s，trial_0_stage_1 完成 → 只剩 trial_1_stage_1 在工作 → 1 个

最终：
- 平均 concurrent actors = 1.00（大部分时间只有 1 个在工作）
- 理想值 = 3（你有 3 个 GPU）
- 利用率 = 1.00 / 3 = 33.33%
"""

# ============================================================================
# 2. GPU 效率 (GPU Efficiency)
# ============================================================================

"""
基础概念：
- 衡量实际使用的 GPU 工作秒数 vs 理想情况下应该使用的秒数
- 理想情况 = 3 个 GPU 全满负荷运行

计算步骤：
1. 总实际 GPU 秒数 = 所有 GPU tasks 的 duration 之和
2. 总理想 GPU 秒数 = total_wall_time × num_gpus
3. GPU 效率 = 总实际 GPU 秒数 / 总理想 GPU 秒数

具体数字（从你的 trace）：
   总 wall 时间 = 66.30s（从第一个 dispatch 到最后一个 complete）
   
   GPU tasks:
   - trial_0_stage_1: 23.224s
   - trial_3_stage_1: 15.049s
   - trial_4_stage_1: 22.820s
   
   总实际 GPU 秒数 = 23.224 + 15.049 + 22.820 = 61.093s
   总理想 GPU 秒数 = 66.30s × 3 GPUs = 198.90s
   
   GPU 效率 = 61.093 / 198.90 = 30.72%
   
解释：
- 理想情况：3 个 GPU 在 66.30s 内应该完成 3×66.30 = 198.90s 的工作
- 实际情况：只完成了 61.09s 的工作
- 浪费了 198.90 - 61.09 = 137.81s 的 GPU 潜力
"""

# ============================================================================
# 3. Stage 利用率 (Per-Stage Utilization)
# ============================================================================

"""
基础概念：
- 衡量每个 stage 在其存续期间是否被充分利用
- 即：stage 的工作时间 / stage 的总持续时间

计算步骤（以 Stage 1 为例）：
1. 找所有 stage=1 的 dispatch 事件，记录时间 → min_dispatch_time
2. 找所有 stage=1 的 complete 事件，记录时间 → max_complete_time
3. Stage 总持续时间 = max_complete_time - min_dispatch_time
4. Stage 总工作时间 = 所有 stage=1 tasks 的 stage_duration_sec 之和
5. Stage 利用率 = Stage 总工作时间 / Stage 总持续时间

具体数字（从你的 trace）：
   Stage 1:
   - first dispatch: 5.202s
   - last complete: 66.295s
   - 总持续时间 = 66.295 - 5.202 = 61.093s
   
   所有 stage=1 的工作时间：
   - trial_0: 23.224s
   - trial_1: ?
   - trial_2: ?
   - trial_3: 15.049s
   - trial_4: 22.820s
   - trial_5: ?
   
   如果总工作时间 ≈ 61.093s，那么利用率 = 61.093 / 61.093 = 100%
   
意义：
- 100% 表示 stage 没有空闲时间，一直在处理某个 trial
- 但这**不代表效率高**！因为虽然 stage 一直在工作，但其他 stage 可能在等待
"""

# ============================================================================
# 4. Trial 吞吐量 (Trial Throughput)
# ============================================================================

"""
基础概念：
- 衡量每秒能完成多少个 trial（从头到尾通过所有 stage）

计算步骤：
1. 找最后一个 stage（stage=num_stages）的所有 complete 事件
2. 记录这些事件的时间戳，从小到大排序
3. 计算时间范围 = last_complete_time - first_complete_time
4. 完成的 trial 数 = 这些 complete 事件的数量
5. 吞吐量 = num_completed_trials / time_range

具体数字（从你的 trace）：
   如果 stage=2/2 的 complete 事件中：
   - 最早的 trial 完成于 t1
   - 最晚的 trial 完成于 t2
   - 完成了 N 个 trial
   
   吞吐量 = N / (t2 - t1) trials/sec
   你的结果: 0.079 trials/sec = 4.8 trials/min
   
解释：
- 这个吞吐量很慢！平均每 12-13 秒才完成 1 个 trial
- 对比来说，如果 1 个 trial 需要 20-30s，吞吐量应该是 0.03-0.05 trials/sec
- 所以流水线确实有 pipeline 效果，但不够明显
"""

# ============================================================================
# 5. Pipeline Speedup（流水线加速比）
# ============================================================================

"""
基础概念：
- 衡量流水线设计带来的加速效果
- 比较：N 个 trial 顺序执行 vs 流水线执行的时间

计算步骤：
1. 计算单个 trial 顺序通过所有 stage 的总时间（使用 trial_0 作为基准）
   baseline_time = stage_1_duration + stage_2_duration + ... + stage_N_duration
   
2. 计算完成所有 trial 的实际总时间
   actual_time = max(所有 complete 事件的时间)
   
3. 如果顺序执行所有 trial：
   sequential_time = baseline_time × num_completed_trials
   
4. Pipeline Speedup = sequential_time / actual_time

具体数字（从你的 trace）：
   trial_0 的执行时间：
   - stage_1: 23.224s
   - baseline_time = 23.224s（只有 1 个 stage）
   
   3 个 trial 顺序执行：
   sequential_time = 23.224 × 3 = 69.67s
   
   实际流水线时间：
   actual_time = 66.30s
   
   Pipeline Speedup = 69.67 / 66.30 = 1.05x
   
解释：
- 流水线提供了 1.05 倍的加速 → 几乎没有加速！
- 理想情况（完美流水线）：可以接近 3x 的加速（不受限于单 stage 最慢的时间）
- 你的情况：stage 1 完成后立即进入 stage 2，但 stage 2 似乎很快完成，所以很难形成流水线重叠

这说明你的流水线**缺乏足够的并行机会**。
"""

# ============================================================================
# 6. Speedup 效率（Pipeline Efficiency）
# ============================================================================

"""
基础概念：
- 衡量流水线的加速效率（相对于 trial 数量）
- 理想情况：N 个 trial 通过 M 个 stage 的流水线应该获得接近 M 倍的加速

计算步骤：
1. Speedup 效率 = Pipeline Speedup / num_completed_trials

具体数字（从你的 trace）：
   Pipeline Speedup = 1.05x
   num_completed_trials = 3
   
   Speedup 效率 = 1.05 / 3 = 0.35 = 35%
   
解释：
- 理想的流水线应该接近 100%（或更高）
- 35% 意味着 3 个 trial 只获得了理想加速的 35%
- 原因：stage 太少（只有 1 个 stage 实际工作），没有形成有效的流水线重叠

示例对比：
- 2-stage 流水线，trial 1 在 stage 1，trial 2 已在 stage 2 → 可以获得 ~2x 加速 → 100% 效率
- 你的情况：所有 stage 基本同步执行，没有时间错开 → 只能获得 1.05x → 35% 效率
"""

# ============================================================================
# 总结：哪些指标最重要？
# ============================================================================

"""
优先级排序：

1. **GPU 利用率** ⭐⭐⭐⭐⭐
   - 直接反映你是否充分利用硬件
   - 你的 33.33% 说明有很大的优化空间
   
2. **Pipeline Speedup** ⭐⭐⭐⭐⭐
   - 最终的效果指标
   - 你的 1.05x 说明流水线效果非常差
   
3. **GPU 效率** ⭐⭐⭐⭐
   - GPU 利用率的另一种表达方式
   - 30.72% 对应了 33.33% 的利用率
   
4. **Stage 利用率** ⭐⭐⭐
   - 帮助诊断哪个 stage 是瓶颈
   - 100% 说明 stage 一直在工作，但可能其他 stage 在等
   
5. **Trial 吞吐量** ⭐⭐⭐
   - 绝对速度指标
   - 用来衡量改进前后的对比
"""

# ============================================================================
# 如何用这些指标调试？
# ============================================================================

"""
场景 1：GPU 利用率低（你的情况）
→ 问题：没有足够的 trial 并发，或 trial 之间的依赖阻塞了调度
→ 解决：
  - 增加 `--architectures-per-step`（更多 trial 同时进行）
  - 增加 `--pipeline-stage-train-workers`（更多 worker 竞争 GPU）
  - 增加 `--num-pipeline-stages`（更长的流水线）

场景 2：Pipeline Speedup 低
→ 问题：stage 之间没有时间重叠
→ 解决：
  - 确保 stage 时间不均（这样前一个 trial 的慢 stage 可以和后一个 trial 的快 stage 重叠）
  - 增加流水线深度

场景 3：某个 Stage 利用率低
→ 问题：这个 stage 是瓶颈，后续 stage 在等待它
→ 解决：
  - 分析这个 stage 的计算，看是否可以优化
  - 或增加这个 stage 的 worker 数量

场景 4：Trial 吞吐量低
→ 问题：基础速度太慢
→ 解决：
  - 减少 partition 大小（让每个任务更快完成）
  - 减少 `--coarse-epochs`（减少每个 trial 的工作量）
"""
