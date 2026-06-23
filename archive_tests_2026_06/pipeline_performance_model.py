#!/usr/bin/env python3
"""
Pipeline vs 架构并行的性能模型
找出临界点：什么时候pipeline更优？
"""

def architecture_parallel_time(num_trials, num_gpus, time_per_trial):
    """架构并行：批次处理"""
    num_batches = (num_trials + num_gpus - 1) // num_gpus
    return num_batches * time_per_trial

def pipeline_time(num_trials, num_stages, time_per_stage, startup_overhead=0):
    """
    Pipeline并行：流水线处理

    假设：
    - 每个stage处理时间相等
    - 第一个trial需要走完所有stages (startup latency)
    - 后续trials以最慢stage的速度流出
    """
    if num_trials == 0:
        return 0

    # 第一个trial的启动延迟
    first_trial_time = num_stages * time_per_stage + startup_overhead

    # 后续trials以pipeline速度流出
    # 理想情况：每完成一个stage，就可以输出一个trial
    remaining_trials = num_trials - 1
    if remaining_trials > 0:
        # 后续trials按最慢stage的速度完成
        return first_trial_time + remaining_trials * time_per_stage
    else:
        return first_trial_time

def analyze_scenarios():
    """分析不同场景下的性能"""

    print("=" * 80)
    print("Pipeline vs 架构并行性能分析")
    print("=" * 80)

    scenarios = [
        # (num_trials, num_gpus, time_per_trial, description)
        (50, 3, 400, "当前场景: 50 trials, 3 GPUs, 400s/trial"),
        (100, 3, 400, "更多trials: 100 trials, 3 GPUs"),
        (200, 3, 400, "大量trials: 200 trials, 3 GPUs"),
        (50, 8, 400, "更多GPUs: 50 trials, 8 GPUs"),
    ]

    for num_trials, num_gpus, time_per_trial, desc in scenarios:
        print(f"\n{desc}")
        print("-" * 80)

        # 架构并行
        arch_time = architecture_parallel_time(num_trials, num_gpus, time_per_trial)
        print(f"架构并行 (1 stage, {num_gpus} workers): {arch_time:.0f}s")

        # Pipeline: 假设3个stages，每个stage时间 = time_per_trial / 3
        num_stages = 3
        time_per_stage = time_per_trial / num_stages

        # 理想pipeline（无开销）
        ideal_pipeline_time = pipeline_time(num_trials, num_stages, time_per_stage, 0)
        speedup_ideal = arch_time / ideal_pipeline_time
        print(f"Pipeline (3 stages, 理想): {ideal_pipeline_time:.0f}s (加速 {speedup_ideal:.2f}×)")

        # 实际pipeline（10%开销）
        overhead = time_per_trial * 0.1  # 10% stage切换开销
        real_pipeline_time = pipeline_time(num_trials, num_stages, time_per_stage, overhead)
        speedup_real = arch_time / real_pipeline_time
        print(f"Pipeline (3 stages, +10%开销): {real_pipeline_time:.0f}s (加速 {speedup_real:.2f}×)")

        # 判断
        if speedup_real > 1.0:
            print(f"✅ Pipeline更优 (快 {(speedup_real-1)*100:.1f}%)")
        else:
            print(f"❌ 架构并行更优 (Pipeline慢 {(1-speedup_real)*100:.1f}%)")

def find_crossover_point():
    """找到pipeline开始优于架构并行的临界点"""
    print("\n" + "=" * 80)
    print("临界点分析：多少trials时pipeline开始有优势？")
    print("=" * 80)

    num_gpus = 3
    time_per_trial = 400
    num_stages = 3
    time_per_stage = time_per_trial / num_stages

    print(f"\n配置: {num_gpus} GPUs, {num_stages} stages, {time_per_trial}s/trial")
    print(f"\nOverhead | 临界trials数 | 说明")
    print("-" * 80)

    for overhead_pct in [0, 5, 10, 20, 30]:
        overhead = time_per_trial * (overhead_pct / 100)

        # 二分查找临界点
        left, right = 1, 1000
        crossover = None

        for _ in range(20):  # 二分迭代
            mid = (left + right) // 2
            arch_time = architecture_parallel_time(mid, num_gpus, time_per_trial)
            pipe_time = pipeline_time(mid, num_stages, time_per_stage, overhead)

            if pipe_time < arch_time:
                crossover = mid
                right = mid - 1
            else:
                left = mid + 1

        if crossover and crossover < 1000:
            print(f"{overhead_pct:3d}%    | {crossover:4d} trials | Pipeline在{crossover}个trials时开始更快")
        else:
            print(f"{overhead_pct:3d}%    | >1000 trials | Pipeline即使有很多trials也不划算")

if __name__ == "__main__":
    analyze_scenarios()
    find_crossover_point()
