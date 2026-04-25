"""
GraphNAS 控制器：支持随机搜索与强化学习搜索（REINFORCE）。
"""

import random
from typing import Dict, List, Tuple

import torch

from nas.search_space import sanitize_config


class RandomGraphNASController:
    """随机采样控制器。"""

    def __init__(self, search_space: Dict[str, List], seed: int = 42):
        self.search_space = search_space
        self.random = random.Random(seed)

    def sample_arch(self) -> Dict:
        arch = {k: self.random.choice(v) for k, v in self.search_space.items()}
        return sanitize_config(arch)

    def sample_arch_batch(self, batch_size: int) -> List[Dict]:
        return [self.sample_arch() for _ in range(batch_size)]

    def topk(self, results: List[Dict], k: int = 3) -> List[Dict]:
        return sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[:k]


class RLGraphNASController:
    """REINFORCE 控制器：按策略分布采样架构并用奖励更新。"""

    def __init__(self, search_space: Dict[str, List], seed: int = 42, lr: float = 1e-2):
        self.search_space = search_space
        self.keys = list(search_space.keys())
        self.choice_lens = {k: len(v) for k, v in search_space.items()}

        torch.manual_seed(seed)
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
            idx = dist.sample()
            arch[k] = self.search_space[k][int(idx.item())]
            logprob = logprob + dist.log_prob(idx)

        arch = sanitize_config(arch)
        return arch, logprob

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

    def topk(self, results: List[Dict], k: int = 3) -> List[Dict]:
        return sorted(results, key=lambda x: (x["score"], -x["params"], -x["time_sec"]), reverse=True)[:k]
