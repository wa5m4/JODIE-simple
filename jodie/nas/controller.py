"""
GraphNAS 控制器：支持随机搜索与强化学习搜索（REINFORCE）。
"""

import random
from typing import Dict, List, Tuple

import torch

from .search_space import sanitize_config


class GraphNASController:
    """NAS 控制器基类，提供通用工具方法。"""

    def topk(self, results: List[Dict], k: int = 3) -> List[Dict]:
        return sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[:k]


class RandomGraphNASController(GraphNASController):
    """随机采样控制器。"""

    def __init__(self, search_space: Dict[str, List], seed: int = 42):
        self.search_space = search_space
        self.random = random.Random(seed)
        self.reward_baseline = 0.0  # 异步 pipeline 路径需要此属性

    def sample_arch(self) -> Dict:
        arch = {k: self.random.choice(v) for k, v in self.search_space.items()}
        return sanitize_config(arch)

    def sample_arch_batch(self, batch_size: int) -> List[Dict]:
        return [self.sample_arch() for _ in range(batch_size)]


class RLGraphNASController(GraphNASController):
    """REINFORCE 控制器：按策略分布采样架构并用奖励更新。"""

    def __init__(self, search_space: Dict[str, List], seed: int = 42, lr: float = 1e-2):
        self.search_space = search_space
        self.keys = list(search_space.keys())
        self.choice_lens = {k: len(v) for k, v in search_space.items()}

        torch.manual_seed(seed)
        # ★ 保真度修复 Fix A(2026-09-15):控制器采样改用专用 Generator,
        # 与进程内其他任何 torch 全局 RNG 消耗(数据加载、模型构建、executor
        # 初始化、params 统计等)完全脱钩。第 k 次采样的 RNG 状态只由控制器
        # 自身的历史采样决定 → 各执行路径(serial / 同步流水线 / 异步流水线)
        # 在相同策略状态 θ 下的第 k 次采样逐位一致。
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self.logits = {
            k: torch.nn.Parameter(torch.zeros(self.choice_lens[k], dtype=torch.float32))
            for k in self.keys
        }
        self.optimizer = torch.optim.Adam(self.logits.values(), lr=lr)
        self.reward_baseline = 0.0

    def sample_arch(self) -> Dict:
        arch, _ = self.sample_arch_with_logprob()
        return arch

    def sample_arch_with_logprob(self) -> Tuple[Dict, torch.Tensor]:
        arch = {}
        logprob = torch.tensor(0.0)

        for k in self.keys:
            dist = torch.distributions.Categorical(logits=self.logits[k])
            # 与 Categorical.sample() 相同的 multinomial 采样,但走专用 Generator
            probs = torch.softmax(self.logits[k], dim=-1)
            idx = torch.multinomial(probs, num_samples=1, generator=self._rng).squeeze(0)
            arch[k] = self.search_space[k][int(idx.item())]
            logprob = logprob + dist.log_prob(idx)

        arch = sanitize_config(arch)
        return arch, logprob

    def compute_logprob(self, arch_config: Dict) -> torch.Tensor:
        """用当前 logits 重新计算给定架构的 logprob（off-policy 更新用）。"""
        device = self.logits[self.keys[0]].device
        logprob = torch.tensor(0.0, requires_grad=True, device=device)
        for k in self.keys:
            v = arch_config.get(k)
            if v not in self.search_space[k]:
                continue
            idx = self.search_space[k].index(v)
            dist = torch.distributions.Categorical(logits=self.logits[k])
            logprob = logprob + dist.log_prob(torch.tensor(idx))
        return logprob

    def sample_arch_batch(self, batch_size: int) -> List[Dict]:
        return [self.sample_arch() for _ in range(batch_size)]

    def sample_arch_batch_with_logprob(self, batch_size: int) -> List[Tuple[Dict, torch.Tensor]]:
        return [self.sample_arch_with_logprob() for _ in range(batch_size)]

    def reinforce_step(self, logprob: torch.Tensor, reward: float):
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
        advantage = reward - self.reward_baseline
        loss = -(logprob * advantage)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reinforce_step_batch(self, samples: List[Tuple[torch.Tensor, float]]):
        if not samples:
            return

        self.optimizer.zero_grad()
        total_loss = None

        for logprob, reward in samples:
            self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
            advantage = reward - self.reward_baseline
            sample_loss = -(logprob * advantage)
            total_loss = sample_loss if total_loss is None else total_loss + sample_loss

        total_loss.backward()
        self.optimizer.step()
