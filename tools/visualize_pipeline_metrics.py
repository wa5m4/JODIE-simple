#!/usr/bin/env python3
"""
Pipeline Efficiency Visualization Tool

为每个指标生成 ASCII 图表和详细的计算步骤
"""

import re
from collections import defaultdict
from typing import List, Dict


def parse_dispatch_complete(log_file: str) -> List[Dict]:
    """提取 dispatch/complete 事件对"""
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            if '[pipeline-trace]' not in line or 'event=' not in line:
                continue
            
            match = re.search(r'elapsed=([\d.]+)s', line)
            elapsed = float(match.group(1)) if match else None
            
            match = re.search(r'event=(\w+)', line)
            event_type = match.group(1) if match else None
            
            match = re.search(r'trial=(\d+)', line)
            trial = int(match.group(1)) if match else None
            
            match = re.search(r'stage=(\d+)/(\d+)', line)
            if match:
                stage = int(match.group(1))
            else:
                stage = None
            
            match = re.search(r'stage_duration_sec=([\d.]+)', line)
            duration = float(match.group(1)) if match else None
            
            if elapsed is not None and event_type is not None and trial is not None and stage is not None:
                events.append({
                    'elapsed': elapsed,
                    'event': event_type,
                    'trial': trial,
                    'stage': stage,
                    'duration': duration,
                })
    
    return sorted(events, key=lambda x: x['elapsed'])


def visualize_gpu_utilization(events: List[Dict]):
    """可视化 GPU 利用率"""
    
    print("\n" + "="*80)
    print("指标 1: GPU 利用率计算步骤")
    print("="*80)
    
    # 提取 dispatch/complete 对
    pairs = {}
    for event in events:
        if event['event'] in ['dispatch', 'complete']:
            key = (event['trial'], event['stage'])
            if key not in pairs:
                pairs[key] = {}
            
            if event['event'] == 'dispatch':
                pairs[key]['start'] = event['elapsed']
            elif event['event'] == 'complete':
                pairs[key]['end'] = event['elapsed']
                pairs[key]['duration'] = event['duration']
    
    # 转换为任务列表
    tasks = []
    for (trial, stage), times in pairs.items():
        if 'start' in times and 'end' in times:
            tasks.append({
                'trial': trial,
                'stage': stage,
                'start': times['start'],
                'end': times['end'],
                'duration': times.get('duration', times['end'] - times['start'])
            })
    
    tasks = sorted(tasks, key=lambda x: x['start'])
    
    print("\n📊 Step 1: 提取所有 GPU task（dispatch → complete）")
    print("-" * 80)
    print(f"{'Trial':<8} {'Stage':<8} {'Start(s)':<12} {'End(s)':<12} {'Duration(s)':<12}")
    print("-" * 80)
    for i, task in enumerate(tasks[:10]):
        print(f"{task['trial']:<8} {task['stage']:<8} {task['start']:>10.2f}  {task['end']:>10.2f}  {task['duration']:>10.2f}")
    if len(tasks) > 10:
        print(f"... 还有 {len(tasks) - 10} 个 task")
    
    # 计算并发度
    print("\n📊 Step 2: 采样时间线，统计每个时刻的并发 GPU actors")
    print("-" * 80)
    
    if not tasks:
        print("No tasks found")
        return
    
    start_time = tasks[0]['start']
    end_time = tasks[-1]['end']
    
    # 采样
    sample_interval = 1.0  # 每 1 秒采样一次
    samples = {}
    t = start_time
    while t <= end_time:
        count = 0
        active_tasks = []
        for task in tasks:
            if task['start'] <= t < task['end']:
                count += 1
                active_tasks.append(f"T{task['trial']}S{task['stage']}")
        samples[t] = (count, active_tasks)
        t += sample_interval
    
    print(f"采样间隔: {sample_interval}s")
    print(f"{'时刻(s)':<12} {'并发数':<8} {'活跃 GPU actors':<40}")
    print("-" * 80)
    
    sorted_samples = sorted(samples.items())
    for t, (count, active) in sorted_samples[:15]:
        active_str = ','.join(active[:3])  # 只显示前 3 个
        if len(active) > 3:
            active_str += f"+{len(active)-3}"
        print(f"{t:>10.2f}s  {count:<8} {active_str:<40}")
    if len(sorted_samples) > 15:
        print(f"... 还有 {len(sorted_samples) - 15} 个采样点")
    
    # 计算平均并发度
    print("\n📊 Step 3: 计算平均并发度")
    print("-" * 80)
    
    concurrent_counts = [count for count, _ in samples.values()]
    avg_concurrent = sum(concurrent_counts) / len(concurrent_counts)
    max_concurrent = max(concurrent_counts)
    
    print(f"平均并发 GPU actors: {avg_concurrent:.2f}")
    print(f"最大并发 GPU actors: {max_concurrent}")
    print(f"理想值 (3 个 GPU): 3")
    print(f"\n✅ GPU 利用率 = {avg_concurrent:.2f} / 3 = {avg_concurrent/3:.2%}")
    
    # ASCII 图表
    print("\n📈 ASCII 图表：GPU 并发度随时间变化")
    print("-" * 80)
    
    max_count = max(concurrent_counts) if concurrent_counts else 1
    
    for t, (count, _) in sorted_samples:
        bar = "█" * count
        print(f"{t:>8.1f}s | {bar:<30} {count} GPU actors")


def visualize_gpu_efficiency(events: List[Dict]):
    """可视化 GPU 效率"""
    
    print("\n" + "="*80)
    print("指标 2: GPU 效率计算步骤")
    print("="*80)
    
    # 提取 task 信息
    tasks = {}
    for event in events:
        if event['event'] in ['dispatch', 'complete']:
            key = (event['trial'], event['stage'])
            if key not in tasks:
                tasks[key] = {}
            
            if event['event'] == 'dispatch':
                tasks[key]['start'] = event['elapsed']
            elif event['event'] == 'complete':
                tasks[key]['end'] = event['elapsed']
                tasks[key]['duration'] = event['duration']
    
    # 转换为任务列表
    task_list = []
    for (trial, stage), times in tasks.items():
        if 'start' in times and 'end' in times:
            task_list.append(times.get('duration', times['end'] - times['start']))
    
    if not task_list:
        print("No tasks found")
        return
    
    # 计算总 wall 时间
    all_events = [e for e in events if e['event'] in ['dispatch', 'complete']]
    if all_events:
        min_time = min(e['elapsed'] for e in all_events if e['event'] == 'dispatch')
        max_time = max(e['elapsed'] for e in all_events if e['event'] == 'complete')
        wall_time = max_time - min_time
    else:
        wall_time = 0
    
    print("\n📊 Step 1: 总结 GPU 工作时间")
    print("-" * 80)
    print(f"总 wall 时间 (第一个 dispatch 到最后一个 complete): {wall_time:.2f}s")
    print(f"总任务数: {len(task_list)}")
    print(f"总 GPU 工作秒数: {sum(task_list):.2f}s")
    
    print("\n📊 Step 2: 计算理想情况下的 GPU 秒数")
    print("-" * 80)
    num_gpus = 3
    ideal_gpu_seconds = wall_time * num_gpus
    
    print(f"理想情况: {num_gpus} 个 GPU 满负荷运行 {wall_time:.2f}s")
    print(f"理想 GPU 秒数 = {wall_time:.2f}s × {num_gpus} = {ideal_gpu_seconds:.2f}s")
    
    print("\n📊 Step 3: 计算 GPU 效率")
    print("-" * 80)
    actual_gpu_seconds = sum(task_list)
    efficiency = actual_gpu_seconds / ideal_gpu_seconds if ideal_gpu_seconds > 0 else 0
    
    print(f"GPU 效率 = 实际 GPU 秒数 / 理想 GPU 秒数")
    print(f"         = {actual_gpu_seconds:.2f} / {ideal_gpu_seconds:.2f}")
    print(f"         = {efficiency:.2%}")
    
    print("\n📊 图表：GPU 利用情况")
    print("-" * 80)
    print(f"理想情况:  {'█'*30}  {ideal_gpu_seconds:.0f}s")
    print(f"实际情况:  {'█'*int(30*efficiency):<30}  {actual_gpu_seconds:.0f}s")
    print(f"浪费部分:  {' '*int(30*efficiency)}{'░'*int(30*(1-efficiency))}")
    print(f"\n浪费的 GPU 秒数: {ideal_gpu_seconds - actual_gpu_seconds:.2f}s")


def visualize_stage_utilization(events: List[Dict]):
    """可视化 Stage 利用率"""
    
    print("\n" + "="*80)
    print("指标 3: Stage 利用率计算步骤")
    print("="*80)
    
    # 按 stage 分组
    by_stage = defaultdict(list)
    
    for event in events:
        if event['event'] in ['dispatch', 'complete'] and event['stage'] is not None:
            key = event['stage']
            by_stage[key].append({
                'event': event['event'],
                'elapsed': event['elapsed'],
                'duration': event['duration'],
                'trial': event['trial']
            })
    
    if not by_stage:
        print("No stage events found")
        return
    
    print("\n📊 为每个 stage 计算利用率")
    print("-" * 80)
    
    for stage in sorted(by_stage.keys()):
        events_in_stage = by_stage[stage]
        
        dispatch_times = [e['elapsed'] for e in events_in_stage if e['event'] == 'dispatch']
        complete_times = [e['elapsed'] for e in events_in_stage if e['event'] == 'complete']
        durations = [e['duration'] for e in events_in_stage if e['event'] == 'complete' and e['duration']]
        
        if dispatch_times and complete_times:
            first_dispatch = min(dispatch_times)
            last_complete = max(complete_times)
            stage_duration = last_complete - first_dispatch
            total_work = sum(durations) if durations else 0
            utilization = total_work / stage_duration if stage_duration > 0 else 0
            
            print(f"\nStage {stage}:")
            print(f"  第一个 dispatch 时刻: {first_dispatch:.2f}s")
            print(f"  最后一个 complete 时刻: {last_complete:.2f}s")
            print(f"  Stage 存续时间: {stage_duration:.2f}s")
            print(f"  所有 tasks 的总工作时间: {total_work:.2f}s")
            print(f"  处理的 trial 数: {len(dispatch_times)}")
            print(f"  Stage 利用率 = {total_work:.2f}s / {stage_duration:.2f}s = {utilization:.2%}")
            
            # ASCII bar
            bar_length = 30
            bar = "█" * int(bar_length * utilization)
            print(f"  {'█'*bar_length}")
            print(f"  {bar:<30} {utilization:.2%}")


def visualize_pipeline_speedup(events: List[Dict]):
    """可视化 Pipeline Speedup"""
    
    print("\n" + "="*80)
    print("指标 4 & 5: Pipeline Speedup 和 Speedup 效率计算步骤")
    print("="*80)
    
    # 提取 trial 0 的 stage 时间
    trial_0_times = {}
    for event in events:
        if event['event'] == 'complete' and event['trial'] == 0 and event['duration']:
            stage = event['stage']
            trial_0_times[stage] = event['duration']
    
    print("\n📊 Step 1: 获取 trial_0 的每个 stage 执行时间（作为基准）")
    print("-" * 80)
    
    total_baseline = 0
    for stage in sorted(trial_0_times.keys()):
        duration = trial_0_times[stage]
        total_baseline += duration
        print(f"Trial 0, Stage {stage}: {duration:.2f}s")
    
    print(f"单个 trial 顺序时间 = {total_baseline:.2f}s")
    
    # 计算实际完成时间
    all_complete_times = [e['elapsed'] for e in events if e['event'] == 'complete']
    if not all_complete_times:
        print("No complete events found")
        return
    
    actual_time = max(all_complete_times)
    
    # 计算完成的 trial 数（最后一个 stage）
    final_stage_events = [e for e in events if e['event'] == 'complete' and e['stage'] == max(e['stage'] for e in events if e['event'] == 'complete')]
    num_trials = len(set(e['trial'] for e in final_stage_events))
    
    print("\n📊 Step 2: 计算实际执行时间和 trial 数")
    print("-" * 80)
    print(f"从第一个 dispatch 到最后一个 complete: {actual_time:.2f}s")
    print(f"完成的 trial 数: {num_trials}")
    
    print("\n📊 Step 3: 计算顺序执行的时间")
    print("-" * 80)
    sequential_time = total_baseline * num_trials
    print(f"如果不用 pipeline，顺序执行 {num_trials} 个 trial:")
    print(f"顺序时间 = {total_baseline:.2f}s × {num_trials} = {sequential_time:.2f}s")
    
    print("\n📊 Step 4: 计算 Pipeline Speedup")
    print("-" * 80)
    speedup = sequential_time / actual_time if actual_time > 0 else 0
    print(f"Pipeline Speedup = 顺序时间 / 实际时间")
    print(f"                = {sequential_time:.2f}s / {actual_time:.2f}s")
    print(f"                = {speedup:.2f}x")
    
    print("\n📊 Step 5: 计算 Speedup 效率")
    print("-" * 80)
    efficiency = speedup / num_trials if num_trials > 0 else 0
    print(f"Speedup 效率 = Speedup / num_trials")
    print(f"             = {speedup:.2f}x / {num_trials}")
    print(f"             = {efficiency:.2%}")
    
    print("\n💡 解释：")
    if efficiency >= 0.8:
        print(f"  ✅ 很好！接近理想的流水线效率")
    elif efficiency >= 0.5:
        print(f"  ⚠️  还可以，但还有优化空间")
    else:
        print(f"  ❌ 流水线效率很差。建议：")
        print(f"     - 增加 `--architectures-per-step` 提高 trial 并发度")
        print(f"     - 增加 `--num-pipeline-stages` 加深流水线")
        print(f"     - 增加 `--pipeline-stage-train-workers` 提高并发 worker 数")


def main(log_file: str):
    events = parse_dispatch_complete(log_file)
    
    if not events:
        print(f"❌ 无法从 {log_file} 中解析事件")
        return
    
    visualize_gpu_utilization(events)
    visualize_gpu_efficiency(events)
    visualize_stage_utilization(events)
    visualize_pipeline_speedup(events)
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80 + "\n")


if __name__ == '__main__':
    import sys
    import glob
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        trace_files = glob.glob('/home/wanghaoyu/JODIE-simple/outputs*/pipeline_trace*.log')
        if trace_files:
            log_file = sorted(trace_files)[-1]
            print(f"🔍 使用最新 trace 文件: {log_file}\n")
        else:
            print("❌ 未找到 trace 日志文件")
            sys.exit(1)
    
    main(log_file)
