"""
纯 PyTorch 事件级图操作器：每次交互触发邻域聚合与注意力计算。
"""

'''
这部分代码主要就是实现了聚合操作，即在每个事件发生时候进行的聚合操作
然后这里提供三种聚合选项，分别是mean，sum，attn，
其中前两者是基于时间衰减权重，后者则是基于时间衰减和注意力
（可选dot点积注意力或者nip注意力）综合得到权重，
也就暗含了这部分进行架构搜索时候的搜索空间就是这三类聚合方式

search_space = {
    "event_agg": ["mean", "sum", "attn"],
    "attn_type": ["dot", "mlp"],      # 仅 attn 时有效
    "time_decay": ["none", "exp", "inverse"],
}

'''
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventGraphOperator(nn.Module):
    """事件级图聚合模块（mean/sum/attn）。"""

    def __init__(
        self,
        embedding_dim: int,
        event_agg: str = "mean",
        agg_activation: str = "none",
        hidden_dim: int = None,
        attn_type: str = "dot",
        time_decay: str = "none",
        msg_linear: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim # 嵌入维度
        self.event_agg = event_agg # 事件聚合方式
        self.agg_activation = agg_activation # 聚合后激活函数
        self.attn_type = attn_type # 注意力类型
        self.time_decay = time_decay # 时间衰减方式
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim

        self.msg_linear = nn.Linear(embedding_dim, embedding_dim) if msg_linear else nn.Identity()
        #对邻居节点的嵌入向量进行线性变换，维度不变
        self.attn_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, self.hidden_dim),
            #将中心节点嵌入、邻居节点嵌入和它们的差的绝对值拼接起来，输入维度是 embedding_dim * 3，输出维度是 embedding_dim
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            #将输出维度从 embedding_dim 变为 1，用于计算注意力权重
        )
        #也就是这部分定义了一个函数
        #可以用于把一个中心节点和他的一个邻居的特征输入 （单个embedding_dim的向量，表示一个中心节点/邻居节点的嵌入向量）
        # 得到这个邻居对于中心结点的重要性（注意力权重，一维，就一个数）



    def _decay_weight(self, delta_t: torch.Tensor) -> torch.Tensor:
        #根据时间间隔计算衰减权重，越旧的交互权重越小
        #这个函数输入是一个时间间隔delta_t，输出是一个权重向量，维度和输入相同
        #权重向量的每个元素表示对应时间间隔的衰减权重，越旧的交互权重越小
        #代表着时间衰减权重，与注意力权重是并列的，也可结合/单独使用
        if self.time_decay == "exp": # 指数衰减（快）
            return torch.exp(-delta_t.clamp(min=0.0))
        if self.time_decay == "inverse":# 反比例衰减（慢）
            return 1.0 / (1.0 + delta_t.clamp(min=0.0))
        return torch.ones_like(delta_t)

    def _attention_score(self, center_emb: torch.Tensor, neigh_emb: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        #计算注意力分数，结合中心节点和邻居节点的嵌入向量，以及时间衰减权重
        #这个函数输入是中心节点的嵌入向量center_emb（维
        # 邻居节点的嵌入向量neigh_emb（维
        # 时间间隔delta_t（维度和邻居节点数量相同），输出是一个注意力分数向量，维度和邻居节点数量相同
        #支持传入单个中心节点，多个邻居节点的情况，计算每个邻居节点相对于中心节点的注意力分数
        if self.attn_type == "dot":
            #点积注意力
            #这个注意力分数是中心节点和邻居节点的嵌入向量的点积，维度是邻居节点数量，输出是一个注意力分数向量，维度和邻居节点数量相同
            score = (center_emb * neigh_emb).sum(dim=-1)
        else:
            feat = torch.cat([center_emb, neigh_emb, (center_emb - neigh_emb).abs()], dim=-1)
            #将中心节点嵌入、邻居节点嵌入和它们的差的绝对值拼接起来，维度是邻居节点数量，输出是一个特征向量，维度和邻居节点数量相同
            score = self.attn_mlp(feat).squeeze(-1)
            #将特征向量传入MLP，得到一个注意力分数向量，维度和邻居节点数量相同
        return score + torch.log(self._decay_weight(delta_t) + 1e-8)
        #将注意力分数向量和时间衰减权重向量相加，得到最终的注意力分数向量，维度和邻居节点数量相同

    def _apply_agg_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.agg_activation == "relu":
            return F.relu(x)
        if self.agg_activation == "tanh":
            return torch.tanh(x)
        if self.agg_activation == "gelu":
            return F.gelu(x)
        return x

    def event_aggregate(
        #把之前所有组件整合在一起，完成邻居信息的聚合
        self,
        center_idx: int, # 中心节点ID
        center_emb: torch.Tensor, # 中心节点嵌入向量
        memory: torch.Tensor, # 所有节点的嵌入表
        neighbors: List[int], # 邻居节点ID列表
        edge_last_time: Dict[Tuple[int, int], float], # 边的最后时间字典
        current_time: float, # 当前时间
    ) -> torch.Tensor:
        if self.event_agg == "none":
            return torch.zeros_like(center_emb)

        if len(neighbors) == 0: # 邻居节点为空，返回全零向量
            return torch.zeros_like(center_emb)

        neigh_ids = torch.tensor(neighbors, dtype=torch.long, device=memory.device)
        # 将邻居ID列表转换为PyTorch张量
        neigh_emb = memory[neigh_ids]
        # 从嵌入表中获取邻居节点的嵌入向量
        center_expand = center_emb.unsqueeze(0).expand_as(neigh_emb)
        # 将中心节点的嵌入向量扩展为与邻居节点数量相同的形状，以便后续计算（同维度）

        delta_t = []
        for nb in neighbors:
            t_last = edge_last_time.get((center_idx, nb), current_time)
            # 从边的最后时间字典中获取当前中心节点和邻居节点的最后交互时间，默认是当前时间
            delta_t.append(max(current_time - t_last, 0.0))
            # 计算当前中心节点和邻居节点的交互时间间隔，取最大值（防止负数）
        delta_t = torch.tensor(delta_t, dtype=center_emb.dtype, device=center_emb.device)
        # 将时间间隔列表转换为PyTorch张量，维度是邻居节点数量

        msg = self.msg_linear(neigh_emb)
        # 对邻居节点的嵌入向量进行线性变换,维度是邻居节点数量 x embedding_dim

    #以下三种聚合方式，分别对应不同的事件聚合策略（mean/sum/attn），根据配置选择使用哪一种

        if self.event_agg == "sum":
            # 对邻居节点的嵌入向量进行加权求和，维度是embedding_dim
            #使用时间衰减权重对邻居节点的嵌入向量进行加权求和，得到最终的聚合结果，维度是embedding_dim
            w = self._decay_weight(delta_t).unsqueeze(-1)
            out = (msg * w).sum(dim=0)
            return self._apply_agg_activation(out)

        if self.event_agg == "attn":
            # 对邻居节点的嵌入向量进行注意力聚合，维度是embedding_dim
            # 计算注意力分数，维度是邻居节点数量
            #同时使用时间衰减权重对注意力分数进行加权，得到最终的注意力分数，维度是邻居节点数量
            score = self._attention_score(center_expand, neigh_emb, delta_t)
            alpha = F.softmax(score, dim=0).unsqueeze(-1)
            out = (msg * alpha).sum(dim=0)
            return self._apply_agg_activation(out)

        # 对邻居节点的嵌入向量进行均值聚合，维度是embedding_dim
        #使用时间衰减权重对邻居节点的嵌入向量进行加权求和，得到最终的聚合结果，维度是embedding_dim
        w = self._decay_weight(delta_t).unsqueeze(-1)
        norm = w.sum(dim=0).clamp(min=1e-8)
        out = (msg * w).sum(dim=0) / norm
        return self._apply_agg_activation(out)
