#!/usr/bin/env python3
"""
Pipeline Efficiency Analysis Tool

Computes quantitative metrics from pipeline trace logs:
- GPU utilization rate
- Stage utilization rate
- Trial throughput
- Ideal vs actual speedup
- Timeline visualization
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple
import json


def parse_trace_log(log_file: str) -> List[Dict]:
    """Parse pipeline trace log and extract events."""
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            if '[pipeline-trace]' not in line:
                continue
            
            # Extract fields
            match = re.search(r'elapsed=([\d.]+)s', line)
            elapsed = float(match.group(1)) if match else None
            
            match = re.search(r'event=(\w+)', line)
            event_type = match.group(1) if match else None
            
            match = re.search(r'trial=(\d+)', line)
            trial = int(match.group(1)) if match else None
            
            match = re.search(r'stage=(\d+)/(\d+)', line)
            if match:
                stage = int(match.group(1))
                total_stages = int(match.group(2))
            else:
                stage = None
                total_stages = None
            
            match = re.search(r'phase=(\w+)', line)
            phase = match.group(1) if match else None
            
            match = re.search(r'stage_duration_sec=([\d.]+)', line)
            duration = float(match.group(1)) if match else None
            
            if elapsed is not None and event_type is not None:
                events.append({
                    'elapsed': elapsed,
                    'event': event_type,
                    'trial': trial,
                    'stage': stage,
                    'total_stages': total_stages,
                    'phase': phase,
                    'duration': duration,
                })
    
    return sorted(events, key=lambda x: x['elapsed'])


def analyze_dispatch_complete(events: List[Dict]) -> Dict:
    """Analyze dispatch/complete events to compute GPU utilization."""
    
    # Track active GPU actors over time
    dispatch_complete = defaultdict(list)
    
    for event in events:
        if event['event'] not in ['dispatch', 'complete']:
            continue
        
        trial = event['trial']
        stage = event['stage']
        elapsed = event['elapsed']
        
        key = (trial, stage)
        
        if event['event'] == 'dispatch':
            dispatch_complete[key].append({
                'type': 'dispatch',
                'time': elapsed,
                'duration': None
            })
        elif event['event'] == 'complete':
            # Find matching dispatch
            if dispatch_complete[key] and dispatch_complete[key][-1]['type'] == 'dispatch':
                dispatch_complete[key][-1]['type'] = 'dispatch_complete'
                dispatch_complete[key][-1]['duration'] = event['duration']
                dispatch_complete[key][-1]['complete_time'] = elapsed
    
    return dispatch_complete


def compute_gpu_utilization(events: List[Dict]) -> Dict:
    """Compute GPU utilization metrics."""
    
    dispatch_complete = analyze_dispatch_complete(events)
    
    # Build timeline of GPU actor activity
    timeline = []
    for (trial, stage), activities in dispatch_complete.items():
        for activity in activities:
            if activity['type'] == 'dispatch_complete':
                timeline.append({
                    'trial': trial,
                    'stage': stage,
                    'start': activity['time'],
                    'end': activity['complete_time'],
                    'duration': activity['duration'] or (activity['complete_time'] - activity['time'])
                })
    
    if not timeline:
        return {}
    
    timeline = sorted(timeline, key=lambda x: x['start'])
    
    # Compute metrics
    total_wall_time = timeline[-1]['end']
    start_time = timeline[0]['start']
    
    # Count concurrent GPU actors at each point in time
    concurrent_actors = []
    sample_times = []
    for t in [start_time + i * 0.1 for i in range(int((total_wall_time - start_time) / 0.1) + 1)]:
        count = sum(1 for task in timeline if task['start'] <= t < task['end'])
        concurrent_actors.append(count)
        sample_times.append(t)
    
    avg_concurrent = sum(concurrent_actors) / len(concurrent_actors) if concurrent_actors else 0
    max_concurrent = max(concurrent_actors) if concurrent_actors else 0
    
    # Ideal GPU actors (if all 3 GPUs always busy)
    ideal_gpu_util = 3
    actual_gpu_util = avg_concurrent
    gpu_util_ratio = actual_gpu_util / ideal_gpu_util
    
    # Total GPU-seconds (sum of all task durations)
    total_gpu_seconds = sum(t['duration'] for t in timeline)
    ideal_gpu_seconds = total_wall_time * ideal_gpu_util
    
    # GPU efficiency
    gpu_efficiency = total_gpu_seconds / ideal_gpu_seconds if ideal_gpu_seconds > 0 else 0
    
    return {
        'total_wall_time': total_wall_time,
        'start_time': start_time,
        'num_tasks': len(timeline),
        'avg_concurrent_actors': avg_concurrent,
        'max_concurrent_actors': max_concurrent,
        'ideal_concurrent_actors': ideal_gpu_util,
        'gpu_utilization_ratio': gpu_util_ratio,
        'total_gpu_seconds': total_gpu_seconds,
        'ideal_gpu_seconds': ideal_gpu_seconds,
        'gpu_efficiency': gpu_efficiency,
        'timeline': timeline
    }


def compute_stage_utilization(events: List[Dict], num_stages: int = 2) -> Dict:
    """Compute per-stage utilization metrics."""
    
    stage_utils = {}
    
    for stage in range(1, num_stages + 1):
        stage_events = [e for e in events if e['stage'] == stage and e['event'] in ['dispatch', 'complete']]
        
        if not stage_events:
            continue
        
        dispatch_times = [e['elapsed'] for e in stage_events if e['event'] == 'dispatch']
        complete_times = [e['elapsed'] for e in stage_events if e['event'] == 'complete']
        
        if dispatch_times and complete_times:
            first_dispatch = min(dispatch_times)
            last_complete = max(complete_times)
            stage_duration = last_complete - first_dispatch
            
            # Total work time (sum of all stage_duration_sec)
            total_work_time = sum(e['duration'] for e in stage_events if e['event'] == 'complete' and e['duration'])
            
            utilization = total_work_time / stage_duration if stage_duration > 0 else 0
            
            stage_utils[stage] = {
                'first_dispatch': first_dispatch,
                'last_complete': last_complete,
                'stage_duration': stage_duration,
                'total_work_time': total_work_time,
                'utilization': utilization,
                'num_trials': len(dispatch_times)
            }
    
    return stage_utils


def compute_trial_throughput(events: List[Dict]) -> Dict:
    """Compute trial throughput metrics."""
    
    complete_events = [e for e in events if e['event'] == 'complete' and e['phase'] == 'train']
    
    if not complete_events:
        return {}
    
    # Group by stage
    by_stage = defaultdict(list)
    for e in complete_events:
        by_stage[e['stage']].append(e['elapsed'])
    
    # For the final stage, compute trials per second
    final_stage_times = sorted(by_stage[max(by_stage.keys())])
    
    if len(final_stage_times) > 1:
        total_time = final_stage_times[-1] - final_stage_times[0]
        num_trials = len(final_stage_times)
        throughput = num_trials / total_time if total_time > 0 else 0
    else:
        throughput = 0
    
    return {
        'final_stage_times': final_stage_times,
        'num_completed_trials': len(final_stage_times),
        'trial_throughput_per_sec': throughput,
        'trials_per_minute': throughput * 60
    }


def compute_ideal_speedup(events: List[Dict]) -> Dict:
    """Compute ideal vs actual speedup."""
    
    dispatch_complete = analyze_dispatch_complete(events)
    
    # Single-stage sequential time (ideal 1 trial through all stages)
    single_trial_times = {}
    
    for (trial, stage), activities in dispatch_complete.items():
        if trial == 0:  # Use first trial as baseline
            for activity in activities:
                if activity['type'] == 'dispatch_complete':
                    duration = activity['duration'] or (activity['complete_time'] - activity['time'])
                    single_trial_times[stage] = duration
    
    # Total sequential time for 1 trial through all stages
    baseline_time = sum(single_trial_times.values()) if single_trial_times else 0
    
    # Actual time to complete N trials in pipelined fashion
    complete_events = [e for e in events if e['event'] == 'complete' and e['phase'] == 'train']
    if complete_events:
        max_elapsed = max(e['elapsed'] for e in complete_events)
        num_trials = len(set(e['trial'] for e in complete_events if e['stage'] == max(e['stage'] for e in complete_events)))
        
        # If all stages completed
        final_stage = max(e['stage'] for e in complete_events)
        final_trials = [e['trial'] for e in complete_events if e['stage'] == final_stage]
        
        if final_trials:
            actual_time = max_elapsed
            ideal_speedup = baseline_time * len(final_trials) / actual_time if actual_time > 0 else 0
        else:
            ideal_speedup = 1.0
    else:
        ideal_speedup = 1.0
    
    return {
        'baseline_single_trial_time': baseline_time,
        'actual_total_time': max_elapsed if complete_events else 0,
        'num_completed_trials': len(final_trials) if complete_events else 0,
        'ideal_speedup': ideal_speedup
    }


def print_report(log_file: str):
    """Print comprehensive efficiency report."""
    
    print(f"\n{'='*80}")
    print(f"Pipeline Efficiency Analysis Report")
    print(f"Log File: {log_file}")
    print(f"{'='*80}\n")
    
    events = parse_trace_log(log_file)
    
    if not events:
        print("No events found in trace log")
        return
    
    # GPU Utilization
    print("1. GPU Utilization Metrics")
    print("-" * 80)
    gpu_metrics = compute_gpu_utilization(events)
    if gpu_metrics:
        print(f"   Total wall time: {gpu_metrics['total_wall_time']:.2f}s")
        print(f"   Number of GPU tasks: {gpu_metrics['num_tasks']}")
        print(f"   Average concurrent GPU actors: {gpu_metrics['avg_concurrent_actors']:.2f}")
        print(f"   Max concurrent GPU actors: {gpu_metrics['max_concurrent_actors']}")
        print(f"   Ideal concurrent actors (3 GPUs): {gpu_metrics['ideal_concurrent_actors']}")
        print(f"   GPU Utilization Ratio: {gpu_metrics['gpu_utilization_ratio']:.2%}")
        print(f"     (actual {gpu_metrics['avg_concurrent_actors']:.2f} / ideal {gpu_metrics['ideal_concurrent_actors']})")
        print(f"   GPU Efficiency: {gpu_metrics['gpu_efficiency']:.2%}")
        print(f"     (total GPU-secs {gpu_metrics['total_gpu_seconds']:.2f} / ideal {gpu_metrics['ideal_gpu_seconds']:.2f})")
    
    # Stage Utilization
    print("\n2. Per-Stage Utilization Metrics")
    print("-" * 80)
    stage_utils = compute_stage_utilization(events)
    for stage, utils in sorted(stage_utils.items()):
        print(f"   Stage {stage}:")
        print(f"      Duration: {utils['stage_duration']:.2f}s")
        print(f"      Total work time: {utils['total_work_time']:.2f}s")
        print(f"      Utilization: {utils['utilization']:.2%}")
        print(f"      Trials processed: {utils['num_trials']}")
    
    # Trial Throughput
    print("\n3. Trial Throughput Metrics")
    print("-" * 80)
    throughput = compute_trial_throughput(events)
    if throughput:
        print(f"   Total completed trials: {throughput['num_completed_trials']}")
        print(f"   Trial throughput: {throughput['trial_throughput_per_sec']:.3f} trials/sec")
        print(f"   Trial throughput: {throughput['trials_per_minute']:.1f} trials/min")
    
    # Ideal Speedup
    print("\n4. Ideal vs Actual Speedup")
    print("-" * 80)
    speedup = compute_ideal_speedup(events)
    print(f"   Single trial sequential time: {speedup['baseline_single_trial_time']:.2f}s")
    print(f"   Total time to complete {speedup['num_completed_trials']} trials: {speedup['actual_total_time']:.2f}s")
    if speedup['baseline_single_trial_time'] > 0 and speedup['num_completed_trials'] > 1:
        sequential_time = speedup['baseline_single_trial_time'] * speedup['num_completed_trials']
        actual_time = speedup['actual_total_time']
        print(f"   Sequential (no pipeline) time: {sequential_time:.2f}s")
        print(f"   Actual time: {actual_time:.2f}s")
        print(f"   Pipeline Speedup: {sequential_time / actual_time:.2f}x")
        print(f"   Speedup Efficiency: {(sequential_time / actual_time) / speedup['num_completed_trials']:.2%}")
        print(f"     (speedup / num_trials; ideal 100% means perfect scaling)")
    
    # Timeline Summary
    print("\n5. Task Timeline Summary")
    print("-" * 80)
    if gpu_metrics and 'timeline' in gpu_metrics:
        print(f"   {'Trial':<8} {'Stage':<8} {'Start':<10} {'End':<10} {'Duration':<10}")
        print(f"   {'-'*50}")
        for task in gpu_metrics['timeline'][:20]:  # Show first 20 tasks
            print(f"   {task['trial']:<8} {task['stage']:<8} {task['start']:>8.2f}s {task['end']:>8.2f}s {task['duration']:>8.2f}s")
        if len(gpu_metrics['timeline']) > 20:
            print(f"   ... ({len(gpu_metrics['timeline']) - 20} more tasks)")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Try to find the latest trace log
        import glob
        from pathlib import Path
        
        trace_files = glob.glob('/home/wanghaoyu/JODIE-simple/outputs*/pipeline_trace*.log')
        if trace_files:
            log_file = sorted(trace_files)[-1]  # Latest file
            print(f"Using latest trace file: {log_file}")
        else:
            print("No trace log file specified. Usage: python analyze_pipeline_efficiency.py <log_file>")
            sys.exit(1)
    
    print_report(log_file)
