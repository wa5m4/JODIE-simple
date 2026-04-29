#!/usr/bin/env python3
"""
Real-time Pipeline Efficiency Monitor

持续监控 trace 日志，每隔一段时间计算效率指标，写入效率日志
"""

import re
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple


class PipelineEfficiencyMonitor:
    """实时效率监控器"""
    
    def __init__(self, trace_file: str, efficiency_log_file: str = None, 
                 sampling_interval: int = 10, num_gpus: int = None):
        """
        Args:
            trace_file: trace 日志文件路径
            efficiency_log_file: 效率日志文件路径（如果 None，自动生成）
            sampling_interval: 采样间隔（秒）
            num_gpus: 用于归一化的 GPU 数量，None 时自动从 CUDA_VISIBLE_DEVICES 读取
        """
        self.trace_file = trace_file
        self.sampling_interval = sampling_interval
        self.num_gpus = num_gpus if num_gpus is not None else self._detect_visible_gpu_count()
        if self.num_gpus <= 0:
            self.num_gpus = 1
        
        # 自动生成效率日志文件名
        if efficiency_log_file is None:
            trace_dir = Path(trace_file).parent
            trace_name = Path(trace_file).stem
            efficiency_log_file = trace_dir / f"efficiency_log_{trace_name}.csv"
        
        self.efficiency_log_file = efficiency_log_file
        self.last_read_pos = 0
        self.events_cache = []
        
        # 初始化效率日志文件（写入 header）
        self._init_log_file()

    def _detect_visible_gpu_count(self) -> int:
        """从 CUDA_VISIBLE_DEVICES 推断可见 GPU 数量。"""

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            return 1

        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        return len(devices) if devices else 1
    
    def _init_log_file(self):
        """初始化效率日志文件，写入 CSV header"""
        with open(self.efficiency_log_file, 'w') as f:
            f.write("timestamp,elapsed_time_s,wall_time,avg_concurrent_gpus,gpu_util_ratio,"
                   "gpu_efficiency,avg_stage_util,max_stage_util,trial_throughput,"
                   "pipeline_speedup,speedup_efficiency,num_completed_tasks\n")
        
        print(f"📊 效率日志已创建: {self.efficiency_log_file}")
    
    def _read_new_events(self) -> List[Dict]:
        """增量读取新的 trace 事件"""
        new_events = []
        
        try:
            with open(self.trace_file, 'r') as f:
                f.seek(self.last_read_pos)
                
                for line in f:
                    if '[pipeline-trace]' not in line or 'event=' not in line:
                        continue
                    
                    # 解析事件
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
                    
                    match = re.search(r'stage_duration_sec=([\d.]+)', line)
                    duration = float(match.group(1)) if match else None
                    
                    if elapsed is not None and event_type is not None:
                        new_events.append({
                            'elapsed': elapsed,
                            'event': event_type,
                            'trial': trial,
                            'stage': stage,
                            'total_stages': total_stages,
                            'duration': duration,
                        })
                
                # 更新读取位置
                self.last_read_pos = f.tell()
        
        except FileNotFoundError:
            pass
        
        return new_events
    
    def _compute_metrics(self, events: List[Dict]) -> Dict:
        """从事件列表计算效率指标"""
        
        if not events:
            return None

        dispatch_events = [e for e in events if e['event'] == 'dispatch']
        complete_events = [e for e in events if e['event'] == 'complete']
        if not dispatch_events or not complete_events:
            return None
        
        # 找出时间范围
        min_elapsed = min(e['elapsed'] for e in dispatch_events)
        max_elapsed = max(e['elapsed'] for e in complete_events)
        current_elapsed = max_elapsed
        
        # 1. GPU 利用率
        gpu_util, concurrent_counts = self._compute_gpu_utilization(events)
        
        # 2. GPU 效率
        gpu_efficiency = self._compute_gpu_efficiency(events, max_elapsed - min_elapsed)
        
        # 3. Stage 利用率
        stage_utils = self._compute_stage_utilization(events)
        
        # 4. Trial 吞吐量
        throughput = self._compute_trial_throughput(events)
        
        # 5. Pipeline Speedup
        speedup, speedup_eff = self._compute_pipeline_speedup(events)
        
        # 6. 完成的任务数
        num_tasks = len([e for e in events if e['event'] == 'complete'])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': current_elapsed,
            'wall_time': max_elapsed - min_elapsed,
            'avg_concurrent_gpus': sum(concurrent_counts) / len(concurrent_counts) if concurrent_counts else 0,
            'gpu_util_ratio': gpu_util,
            'gpu_efficiency': gpu_efficiency,
            'avg_stage_util': sum(stage_utils) / len(stage_utils) if stage_utils else 0,
            'max_stage_util': max(stage_utils) if stage_utils else 0,
            'trial_throughput': throughput,
            'pipeline_speedup': speedup,
            'speedup_efficiency': speedup_eff,
            'num_completed_tasks': num_tasks,
        }
    
    def _compute_gpu_utilization(self, events: List[Dict]) -> Tuple[float, List]:
        """计算 GPU 利用率"""
        
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
        
        if not tasks:
            return 0, []
        
        # 采样计算并发度
        start_time = min(t['start'] for t in tasks)
        end_time = max(t['end'] for t in tasks)
        
        concurrent_counts = []
        sample_interval = 0.5  # 每 0.5s 采样一次
        t = start_time
        while t <= end_time:
            count = sum(1 for task in tasks if task['start'] <= t < task['end'])
            concurrent_counts.append(count)
            t += sample_interval
        
        avg_concurrent = sum(concurrent_counts) / len(concurrent_counts) if concurrent_counts else 0
        gpu_util = avg_concurrent / self.num_gpus
        
        return gpu_util, concurrent_counts
    
    def _compute_gpu_efficiency(self, events: List[Dict], wall_time: float) -> float:
        """计算 GPU 效率"""
        
        if wall_time <= 0:
            return 0
        
        # 总 GPU 工作秒数
        total_gpu_seconds = sum(e['duration'] for e in events 
                               if e['event'] == 'complete' and e['duration'])
        
        # 理想 GPU 秒数
        ideal_gpu_seconds = wall_time * self.num_gpus
        
        efficiency = total_gpu_seconds / ideal_gpu_seconds if ideal_gpu_seconds > 0 else 0
        return efficiency
    
    def _compute_stage_utilization(self, events: List[Dict]) -> List[float]:
        """计算每个 stage 的利用率"""
        
        # 按 stage 分组
        by_stage = defaultdict(list)
        for event in events:
            if event['event'] in ['dispatch', 'complete'] and event['stage'] is not None:
                by_stage[event['stage']].append(event)
        
        stage_utils = []
        for stage in sorted(by_stage.keys()):
            events_in_stage = by_stage[stage]
            
            dispatch_times = [e['elapsed'] for e in events_in_stage if e['event'] == 'dispatch']
            complete_times = [e['elapsed'] for e in events_in_stage if e['event'] == 'complete']
            durations = [e['duration'] for e in events_in_stage 
                        if e['event'] == 'complete' and e['duration']]
            
            if dispatch_times and complete_times:
                first_dispatch = min(dispatch_times)
                last_complete = max(complete_times)
                stage_duration = last_complete - first_dispatch
                total_work = sum(durations) if durations else 0
                utilization = total_work / stage_duration if stage_duration > 0 else 0
                stage_utils.append(utilization)
        
        return stage_utils
    
    def _compute_trial_throughput(self, events: List[Dict]) -> float:
        """计算 trial 吞吐量（trials/sec）"""
        
        complete_events = [e for e in events if e['event'] == 'complete' and e['stage'] is not None]
        
        if not complete_events:
            return 0
        
        # 找最后一个 stage
        final_stage = max(e['stage'] for e in complete_events)
        final_events = [e['elapsed'] for e in complete_events if e['stage'] == final_stage]
        
        if len(final_events) < 2:
            return 0
        
        final_events = sorted(final_events)
        time_range = final_events[-1] - final_events[0]
        
        if time_range <= 0:
            return 0
        
        throughput = len(final_events) / time_range
        return throughput
    
    def _compute_pipeline_speedup(self, events: List[Dict]) -> Tuple[float, float]:
        """计算 pipeline speedup 和 efficiency"""
        
        # 提取 trial_0 的完成时间
        trial_0_events = [e for e in events if e['event'] == 'complete' and e['trial'] == 0]
        
        if not trial_0_events:
            return 1.0, 0.0
        
        baseline_time = sum(e['duration'] for e in trial_0_events if e['duration'])
        
        # 所有完成的 trial 数
        complete_events = [e for e in events if e['event'] == 'complete']
        if not complete_events:
            return 1.0, 0.0
        
        final_stage = max(e['stage'] for e in complete_events)
        final_trials = set(e['trial'] for e in complete_events if e['stage'] == final_stage)
        num_trials = len(final_trials)
        
        # 实际时间
        max_elapsed = max(e['elapsed'] for e in complete_events if e['stage'] == final_stage)
        
        sequential_time = baseline_time * num_trials
        actual_time = max_elapsed - min(e['elapsed'] for e in events if e['event'] == 'dispatch')
        
        if actual_time <= 0:
            return 1.0, 0.0
        
        speedup = sequential_time / actual_time
        speedup_eff = speedup / num_trials if num_trials > 0 else 0
        
        return speedup, speedup_eff
    
    def _write_metrics(self, metrics: Dict):
        """写入指标到效率日志"""
        
        if metrics is None:
            return
        
        with open(self.efficiency_log_file, 'a') as f:
            f.write(f"{metrics['timestamp']},"
                   f"{metrics['elapsed_time']:.2f},"
                   f"{metrics['wall_time']:.2f},"
                   f"{metrics['avg_concurrent_gpus']:.2f},"
                   f"{metrics['gpu_util_ratio']:.4f},"
                   f"{metrics['gpu_efficiency']:.4f},"
                   f"{metrics['avg_stage_util']:.4f},"
                   f"{metrics['max_stage_util']:.4f},"
                   f"{metrics['trial_throughput']:.4f},"
                   f"{metrics['pipeline_speedup']:.4f},"
                   f"{metrics['speedup_efficiency']:.4f},"
                   f"{metrics['num_completed_tasks']}\n")
            f.flush()
    
    def start_monitoring(self, duration: int = None):
        """启动监控
        
        Args:
            duration: 监控持续时间（秒），None 表示无限期
        """
        
        print(f"🔍 开始监控 {self.trace_file}")
        print(f"📊 效率日志: {self.efficiency_log_file}")
        print(f"⏱️  采样间隔: {self.sampling_interval}s")
        print(f"🖥️  归一化 GPU 数: {self.num_gpus}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                # 检查是否超时
                if duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed > duration:
                        print(f"\n✅ 监控时间已到 ({elapsed:.0f}s)，停止监控")
                        break
                
                # 读取新事件
                new_events = self._read_new_events()
                if new_events:
                    self.events_cache.extend(new_events)
                
                # 计算和写入指标
                if self.events_cache:
                    metrics = self._compute_metrics(self.events_cache)
                    if metrics:
                        self._write_metrics(metrics)
                        iteration += 1
                        
                        # 终端输出最新指标
                        print(f"[{metrics['timestamp']}] "
                             f"GPU利用率: {metrics['gpu_util_ratio']:.1%} | "
                             f"GPU效率: {metrics['gpu_efficiency']:.1%} | "
                             f"Speedup: {metrics['pipeline_speedup']:.2f}x | "
                             f"完成任务: {metrics['num_completed_tasks']}")
                
                # 等待下一个采样周期
                time.sleep(self.sampling_interval)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  监控已停止 (按 Ctrl+C)")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python monitor_pipeline_efficiency.py <trace_log_file> [interval_sec] [duration_sec] [num_gpus]")
        print("例子: python monitor_pipeline_efficiency.py outputs/pipeline_trace_*.log 10 3600 3")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    num_gpus = int(sys.argv[4]) if len(sys.argv) > 4 else None
    monitor = PipelineEfficiencyMonitor(trace_file, sampling_interval=interval, num_gpus=num_gpus)
    monitor.start_monitoring(duration)


if __name__ == '__main__':
    main()
