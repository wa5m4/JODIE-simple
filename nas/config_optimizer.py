"""
自动化配置优化器：根据 GPU 数量、事件数、分区数自动分配 worker、stage、partition。
专门用于 Pipeline-Smart 的智能化配置。

核心特性：
  - 启发式算法：快速配置，用于初始化
  - 动态规划成本优化：基于事件成本模型，最优化 partition 分配到 stage
  - 成本模型：考虑事件数、用户多样性、物品多样性、时间跨度
"""

from typing import Dict, List, Tuple, Optional
import math


class CostModel:
    """事件成本估计模型（同 ray_pipeline.py 中的逻辑）"""
    
    def __init__(self, user_weight: float = 0.25, item_weight: float = 0.25, span_weight: float = 0.0):
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.span_weight = span_weight
    
    def estimate_partition_costs(self, partition_info_list: List[Dict]) -> List[float]:
        """
        估计每个 partition 的成本
        
        partition_info 应包含：
        {
            'num_events': int,
            'unique_users': int,
            'new_users': int,
            'unique_items': int,
            'new_items': int,
            'time_span': float,
        }
        """
        costs = []
        for info in partition_info_list:
            cost = float(info.get('num_events', 1))
            cost += self.user_weight * float(info.get('unique_users', 0) + info.get('new_users', 0))
            cost += self.item_weight * float(info.get('unique_items', 0) + info.get('new_items', 0))
            cost += self.span_weight * float(info.get('time_span', 0))
            costs.append(max(cost, 1.0))
        return costs
    
    def optimize_partition_grouping(self, partition_costs: List[float], num_stages: int) -> List[Tuple[int, int]]:
        """
        用动态规划最优化地将 partition 分组到 num_stages 个 stage
        
        返回：[(start_idx, end_idx), ...] 列表，表示每个 stage 包含的 partition 范围
        """
        if num_stages <= 0 or not partition_costs:
            return []
        
        num_stages = min(num_stages, len(partition_costs))
        n = len(partition_costs)
        total_cost = sum(partition_costs)
        target_cost = total_cost / num_stages
        
        # 构造前缀和
        prefix_costs = [0.0]
        for cost in partition_costs:
            prefix_costs.append(prefix_costs[-1] + cost)
        
        def segment_cost(start_idx: int, end_idx: int) -> float:
            """计算 partition[start_idx:end_idx] 的总成本"""
            return prefix_costs[end_idx] - prefix_costs[start_idx]
        
        # DP 数组：dp[i][j] = 将前 i 个 partition 分为 j 个 stage 的最小平衡成本
        # (最小化 stage 间的方差)
        inf = float('inf')
        dp = [[inf] * (num_stages + 1) for _ in range(n + 1)]
        backtrack = [[-1] * (num_stages + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        
        for end_idx in range(1, n + 1):
            max_stages_here = min(num_stages, end_idx)
            for stage_count in range(1, max_stages_here + 1):
                best_score = inf
                best_start = -1
                min_start = stage_count - 1  # 至少需要 stage_count 个 partition
                for start_idx in range(min_start, end_idx):
                    prev_score = dp[start_idx][stage_count - 1]
                    if prev_score == inf:
                        continue
                    current_cost = segment_cost(start_idx, end_idx)
                    # 最小化方差：(cost - target_cost)^2
                    score = prev_score + (current_cost - target_cost) ** 2
                    if score < best_score:
                        best_score = score
                        best_start = start_idx
                dp[end_idx][stage_count] = best_score
                backtrack[end_idx][stage_count] = best_start
        
        if dp[n][num_stages] == inf:
            # 回退到均匀分割
            return self._uniform_grouping(n, num_stages)
        
        # 回溯得到分组
        grouping = []
        end_idx = n
        stage_count = num_stages
        while stage_count > 0:
            start_idx = backtrack[end_idx][stage_count]
            if start_idx < 0:
                return self._uniform_grouping(n, num_stages)
            grouping.append((start_idx, end_idx))
            end_idx = start_idx
            stage_count -= 1
        
        grouping.reverse()
        return grouping
    
    def _uniform_grouping(self, n: int, num_stages: int) -> List[Tuple[int, int]]:
        """均匀分割 n 个 partition 到 num_stages 个 stage"""
        grouping = []
        base = n // num_stages
        remainder = n % num_stages
        start = 0
        for i in range(num_stages):
            chunk_size = base + (1 if i < remainder else 0)
            end = start + chunk_size
            grouping.append((start, end))
            start = end
        return grouping


class ConfigOptimizer:
    """基于资源和数据特征的智能配置优化器"""

    @staticmethod
    def _distribute_workers(gpu_count: int, num_stages: int) -> List[int]:
        """把可用 GPU 分摊到各 stage，确保总 worker 数不超过 GPU 数。"""
        gpu_count = max(int(gpu_count), 1)
        num_stages = max(1, min(int(num_stages), gpu_count))

        base = gpu_count // num_stages
        remainder = gpu_count % num_stages
        return [max(1, base + (1 if idx < remainder else 0)) for idx in range(num_stages)]

    @staticmethod
    def _choose_stage_count(gpu_count: int, num_events: int, max_stages: int) -> int:
        """选择 stage 数：保留至少 1 张卡用于 stage 内并行，避免退化成 1 worker/stage。"""
        gpu_count = max(int(gpu_count), 1)
        max_stages = max(int(max_stages), 1)

        if gpu_count <= 2:
            return min(gpu_count, max_stages)

        events_per_gpu = num_events / max(1, gpu_count)
        if events_per_gpu < 5000:
            preferred = 2
        elif events_per_gpu < 20000:
            preferred = 3
        else:
            preferred = 4

        # 关键：stage 数通常应小于 GPU 数，给每个 stage 留出多个 worker 的空间。
        return max(2, min(preferred, max_stages, gpu_count - 1))

    @staticmethod
    def _allocate_stage_workers(num_stages: int, gpu_count: int, stage_weights: Optional[List[float]] = None) -> List[int]:
        """根据 stage 权重把 GPU 分到各 stage，尽量让重 stage 拿到更多 worker。"""
        num_stages = max(int(num_stages), 1)
        gpu_count = max(int(gpu_count), 1)
        if num_stages == 1:
            return [gpu_count]

        # 先保证每个 stage 至少 1 个 worker。
        workers = [1 for _ in range(num_stages)]
        remaining = gpu_count - num_stages
        if remaining <= 0:
            return workers

        if not stage_weights or len(stage_weights) != num_stages:
            # 没有权重时，默认给前面的 stage 更多 worker，避免平均切分退化。
            stage_weights = [float(num_stages - idx) for idx in range(num_stages)]

        weight_sum = sum(max(float(w), 0.0) for w in stage_weights)
        if weight_sum <= 0:
            stage_weights = [1.0 for _ in range(num_stages)]
            weight_sum = float(num_stages)

        ideal = [remaining * max(float(w), 0.0) / weight_sum for w in stage_weights]
        extra = [int(x) for x in ideal]
        allocated = sum(extra)
        workers = [workers[idx] + extra[idx] for idx in range(num_stages)]

        leftover = remaining - allocated
        if leftover > 0:
            fractional_order = sorted(
                range(num_stages),
                key=lambda idx: (ideal[idx] - extra[idx], stage_weights[idx]),
                reverse=True,
            )
            for idx in fractional_order[:leftover]:
                workers[idx] += 1

        return workers

    @staticmethod
    def parse_gpu_list(gpu_list_str: str) -> List[int]:
        """解析 GPU 列表字符串 (e.g., '0,1,2') 为整数列表。"""
        if not gpu_list_str or not gpu_list_str.strip():
            return []
        return [int(x.strip()) for x in gpu_list_str.split(',')]

    @staticmethod
    def auto_allocate_config(
        gpu_count: int,
        num_events: int,
        num_partitions: int,
        architectures_per_step: int = 2,
        min_workers_per_stage: int = 1,
        max_stages: int = 8,
        coarse_trials: int = 6,
    ) -> Dict:
        """自动化配置算法：根据资源和数据特征计算最优 pipeline 配置。"""
        info_lines = []

        num_stages = ConfigOptimizer._choose_stage_count(gpu_count, num_events, max_stages)
        info_lines.append(f"GPUs: {gpu_count}, Stages: {num_stages}")

        train_worker_counts = ConfigOptimizer._allocate_stage_workers(num_stages, gpu_count)
        eval_worker_counts = ConfigOptimizer._allocate_stage_workers(num_stages, gpu_count)
        info_lines.append(f"Train workers per stage: {train_worker_counts}")
        info_lines.append(f"Eval workers per stage: {eval_worker_counts}")

        if num_partitions > 0:
            target_events_per_partition = num_events // max(1, num_partitions)
            partition_size = max(1000, target_events_per_partition // 2)
        else:
            if num_events < 10000:
                partition_size = max(500, num_events // 4)
            elif num_events < 100000:
                partition_size = max(2000, num_events // 8)
            else:
                partition_size = max(5000, num_events // 16)

        info_lines.append(
            f"Events: {num_events}, Partitions: {num_partitions}, Partition size: {partition_size}"
        )
        info_lines.append(f"Trials: {coarse_trials}, Architectures/step: {architectures_per_step}")

        return {
            'num_pipeline_stages': num_stages,
            'pipeline_stage_train_workers': ','.join(str(count) for count in train_worker_counts),
            'pipeline_stage_eval_workers': ','.join(str(count) for count in eval_worker_counts),
            'partition_size': partition_size,
            'architectures_per_step': architectures_per_step,
            'info': '\n'.join(info_lines),
        }

    @staticmethod
    def auto_allocate_config_with_cost_model(
        gpu_count: int,
        num_events: int,
        partition_costs: Optional[List[float]] = None,
        architectures_per_step: int = 2,
        coarse_trials: int = 6,
        user_weight: float = 0.25,
        item_weight: float = 0.25,
        span_weight: float = 0.0,
    ) -> Dict:
        """使用成本模型的高级自动化配置：考虑 partition 的实际工作量分布。"""
        info_lines = []

        cost_model = CostModel(
            user_weight=user_weight,
            item_weight=item_weight,
            span_weight=span_weight,
        )
        info_lines.append(
            f"[Cost Model] User weight={user_weight}, Item weight={item_weight}, Span weight={span_weight}"
        )

        num_stages = ConfigOptimizer._choose_stage_count(gpu_count, num_events, 8)

        info_lines.append(f"GPUs: {gpu_count}, Stages (DP-optimized): {num_stages}")

        if partition_costs and len(partition_costs) > 0:
            grouping_preview = cost_model.optimize_partition_grouping(partition_costs, num_stages)
            stage_weights = [float(sum(partition_costs[start_idx:end_idx])) for start_idx, end_idx in grouping_preview]
        else:
            stage_weights = None

        train_workers = ConfigOptimizer._allocate_stage_workers(num_stages, gpu_count, stage_weights)
        eval_workers = ConfigOptimizer._allocate_stage_workers(num_stages, gpu_count, stage_weights)
        info_lines.append(f"Train workers per stage: {train_workers}")
        info_lines.append(f"Eval workers per stage: {eval_workers}")

        if partition_costs and len(partition_costs) > 0:
            grouping = cost_model.optimize_partition_grouping(partition_costs, num_stages)
            if grouping:
                imbalance = []
                for start_idx, end_idx in grouping:
                    stage_cost = sum(partition_costs[start_idx:end_idx])
                    imbalance.append(f"{end_idx - start_idx} partitions (cost={stage_cost:.0f})")
                info_lines.append(f"[DP Grouping] {', '.join(imbalance)}")

        partition_size = max(500, num_events // max(1, min(8, gpu_count * 2))) if num_events > 0 else 500
        info_lines.append(f"Events: {num_events}, Partition size: {partition_size}")
        info_lines.append(f"Trials: {coarse_trials}, Architectures/step: {architectures_per_step}")

        return {
            'num_pipeline_stages': num_stages,
            'pipeline_stage_train_workers': ','.join(str(count) for count in train_workers),
            'pipeline_stage_eval_workers': ','.join(str(count) for count in eval_workers),
            'partition_size': partition_size,
            'architectures_per_step': architectures_per_step,
            'info': '\n'.join(info_lines),
            'cost_model': cost_model,
        }

    @staticmethod
    def _optimal_worker_allocation(stage_costs: List[float], gpu_count: int) -> List[int]:
        """
        最优 worker 分配：w_i* = m * T_i / sum(T_j)

        数学推导：pipeline 吞吐 = min_i(w_i/T_i)，最大化该值等价于令所有 stage 吞吐相等。
        令 w_i/T_i = λ* 对所有 i，则 w_i = λ*·T_i，代入 Σw_i=m 得 λ*=m/ΣT_j，
        故 w_i* = m·T_i/ΣT_j。此解在 T_i 已知时为全局最优（Lagrange 乘数法可证）。
        """
        m = max(int(gpu_count), 1)
        S = len(stage_costs)
        if S == 0:
            return []
        if S == 1:
            return [m]
        T_sum = sum(stage_costs)
        if T_sum <= 0:
            # 均匀分配
            base, rem = divmod(m, S)
            return [base + (1 if i < rem else 0) for i in range(S)]
        # 按比例分配，每 stage 至少 1 个 worker
        raw = [m * t / T_sum for t in stage_costs]
        workers = [max(1, int(x)) for x in raw]
        leftover = m - sum(workers)
        if leftover > 0:
            order = sorted(range(S), key=lambda i: raw[i] - int(raw[i]), reverse=True)
            for i in order[:leftover]:
                workers[i] += 1
        elif leftover < 0:
            # 超出（因 max(1,...) 导致），从最小 stage 减
            order = sorted(range(S), key=lambda i: workers[i], reverse=True)
            for i in order[:-leftover]:
                if workers[i] > 1:
                    workers[i] -= 1
        return workers

    @staticmethod
    def auto_allocate_config_advanced(
        gpu_count: int,
        num_events: int,
        num_partitions: int,
        architectures_per_step: int = 2,
        coarse_trials: int = 6,
        epochs: int = 1,
        partition_costs: Optional[List[float]] = None,
        num_users: int = 0,
        num_items: int = 0,
        max_embedding_dim: int = 128,
        max_neighbors: int = 10,
        gpu_memory_mb: int = 0,
    ) -> Dict:
        """
        智慧分配策略：
        1. S：基于 events_per_gpu（overhead 占比），数据量大时多 stage 形成流水线
        2. worker：均等分配
        3. partition_size：每 stage 至少 5 个 partition
        4. max_batch_size：基于显存估算批处理的最大 batch 大小
        """
        m = max(int(gpu_count), 1)
        info_lines = []

        # Step 1: stage 数（基于 events_per_gpu）
        events_per_gpu = num_events / m if m > 0 else num_events
        if m <= 1 or events_per_gpu < 10000:
            S = 1
        elif events_per_gpu < 50000:
            S = min(2, m)
        elif events_per_gpu < 200000:
            S = min(3, m)
        else:
            S = min(max(2, int(math.log2(m)) + 1), m)
        info_lines.append(f"GPUs: {m}, Stages: {S} (events/GPU={events_per_gpu:.0f})")

        # Step 2: 均等分配 worker
        workers = ConfigOptimizer._optimal_worker_allocation([1.0] * S, m)
        info_lines.append(f"Workers: {workers}")

        # Step 3: partition_size
        target_partitions = S * 5 * max(1, (epochs + 1) // 2)
        partition_size = max(200, num_events // max(1, target_partitions)) if num_events > 0 else 500
        info_lines.append(f"Partition size: {partition_size} (target={target_partitions}, epochs={epochs})")

        # Step 4: max_batch_size（显存估算）
        if gpu_memory_mb <= 0:
            try:
                import torch
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 16000
            except Exception:
                gpu_memory_mb = 16000
        usable_mb = gpu_memory_mb * 0.8
        model_params_est = max_embedding_dim ** 2 * 4 + (num_users + num_items) * max_embedding_dim
        static_mem_mb = (model_params_est * 4 * 4 + (num_users + num_items) * max_neighbors * max_embedding_dim * 4) / 1024**2
        event_mem_bytes = max_embedding_dim * 6 * 4
        remaining_mb = max(0.0, usable_mb - static_mem_mb)
        max_batch_size = max(1, int(remaining_mb * 1024**2 / event_mem_bytes)) if event_mem_bytes > 0 else 1024
        info_lines.append(f"Max batch_size: {max_batch_size} (static_mem≈{static_mem_mb:.0f}MB)")

        auto_arch_per_step = m * 3
        info_lines.append(f"Architectures per step: {auto_arch_per_step}")

        return {
            'num_pipeline_stages': S,
            'pipeline_stage_train_workers': ','.join(str(w) for w in workers),
            'pipeline_stage_eval_workers': ','.join(str(w) for w in workers),
            'partition_size': partition_size,
            'architectures_per_step': auto_arch_per_step,
            'max_batch_size': max_batch_size,
            'info': '\n'.join(info_lines),
        }
