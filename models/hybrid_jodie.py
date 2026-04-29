"""
时序事件驱动 GNN-JODIE：每个事件执行图操作并更新记忆。
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.gnn_encoder import EventGraphOperator


class TemporalEventGNNJODIE(nn.Module):
    """事件级时序图神经网络 JODIE。"""

    def __init__(
        #主模型的初始化
        self,
        num_users: int,              # 用户数量
        num_items: int,              # 商品数量
        embedding_dim: int,           # 嵌入维度
        feature_dim: int,             # 特征维度
        event_agg: str = "mean",       # 事件聚合方式
        agg_activation: str = "none",  # 聚合后激活函数
        attn_type: str = "dot",       # 注意力方式
        time_decay: str = "none",     # 时间衰减方式
        max_neighbors: int = 20,       # 最大邻域数量
        hidden_dim: int = 128,         # 隐藏层维度
        memory_cell: str = "gru",       # 内存单元类型
        time_proj: str = "linear",     # 时间投影方式
        memory_gate: str = "on",       # 内存门类型
        enable_event_agg: bool = True,   # 是否启用事件聚合
        enable_graph_update: bool = True, # 是否更新动态图结构
        message_mode: str = "agg",      # 消息来源（agg/peer）
        msg_linear: bool = True,         # 聚合消息线性层开关
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items  # 总节点数
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.max_neighbors = max_neighbors
        self.hidden_dim = hidden_dim
        self.memory_cell = memory_cell
        self.time_proj_mode = time_proj
        self.memory_gate = memory_gate
        self.enable_event_agg = bool(enable_event_agg)
        self.enable_graph_update = bool(enable_graph_update)
        self.message_mode = str(message_mode).lower()
        if self.message_mode not in {"agg", "peer"}:
            raise ValueError(f"Unsupported message_mode: {message_mode}")

        if self.message_mode == "peer":
            self.enable_event_agg = False

        op_agg = event_agg if self.enable_event_agg else "none"

        self.register_buffer("memory", torch.zeros(self.num_nodes, embedding_dim))
        self.register_buffer("last_time", torch.zeros(self.num_nodes))
        # LSTM cell state buffers (only used when memory_cell == "lstm")
        if memory_cell == "lstm":
            self.register_buffer("user_cell_state", torch.zeros(num_users, embedding_dim))
            self.register_buffer("item_cell_state", torch.zeros(num_items, embedding_dim))

        self.event_operator = EventGraphOperator(
            embedding_dim=embedding_dim,
            event_agg=op_agg,
            agg_activation=agg_activation,
            hidden_dim=hidden_dim,
            attn_type=attn_type,
            time_decay=time_decay,
            msg_linear=msg_linear,
        )
        # 事件级图聚合模块（mean/sum/attn），在每个事件发生时候进行的聚合操作
        #实现部分在gnn_encoder.py中，提供三种聚合选项，分别是mean，sum，attn


        msg_in_dim = embedding_dim * 2 + feature_dim
        #RNN 输入维度：用户特征 + 物品特征 + 交互特征
        if memory_cell == "rnn":
            self.user_cell = nn.RNNCell(msg_in_dim, embedding_dim)
            self.item_cell = nn.RNNCell(msg_in_dim, embedding_dim)
        elif memory_cell == "gru":
            self.user_cell = nn.GRUCell(msg_in_dim, embedding_dim)
            self.item_cell = nn.GRUCell(msg_in_dim, embedding_dim)
        elif memory_cell == "lstm":
            self.user_cell = nn.LSTMCell(msg_in_dim, embedding_dim)
            self.item_cell = nn.LSTMCell(msg_in_dim, embedding_dim)
        else:
            self.user_cell = None
            self.item_cell = None
            self.add_linear_user = nn.Linear(msg_in_dim, embedding_dim)
            self.add_linear_item = nn.Linear(msg_in_dim, embedding_dim)

        if time_proj == "linear":
            self.time_proj = nn.Linear(1, embedding_dim, bias=False)
        elif time_proj == "mlp":
            self.time_proj = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.time_proj = None

        self.gate_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        self.predict_layer = nn.Linear(embedding_dim, embedding_dim)

        self.reset_state()

    def reset_state(self):
        self.memory.zero_()
        self.last_time.zero_()
        if self.memory_cell == "lstm":
            self.user_cell_state.zero_()
            self.item_cell_state.zero_()

    def export_runtime_state(self) -> Dict[str, torch.Tensor]:
        state = {
            "memory": self.memory.detach().clone(),
            "last_time": self.last_time.detach().clone(),
        }
        if self.memory_cell == "lstm":
            state["user_cell_state"] = self.user_cell_state.detach().clone()
            state["item_cell_state"] = self.item_cell_state.detach().clone()
        return state

    def import_runtime_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.memory.copy_(state["memory"].to(self.memory.device))
        self.last_time.copy_(state["last_time"].to(self.last_time.device))
        if self.memory_cell == "lstm":
            self.user_cell_state.copy_(state["user_cell_state"].to(self.user_cell_state.device))
            self.item_cell_state.copy_(state["item_cell_state"].to(self.item_cell_state.device))

    def _project_time(self, emb: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # 时间投影层，将时间差映射到嵌入维度
        if self.time_proj is None or self.time_proj_mode == "off":
            return emb
        factor = self.time_proj(delta_t)
        return emb * (1 + factor)

    def _node_ids(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将用户ID和商品ID转换为节点ID
        #如user_nodes = [0, 5, 10] ， item_nodes = [1000, 1100, 1200]
        user_nodes = user_ids
        item_nodes = item_ids + self.num_users#物品特征偏移，因为前面是用户节点，所以物品节点的ID需要加上用户数量的偏移
        return user_nodes, item_nodes

    def _neighbors(self, graph_state: Dict, node_id: int) -> List[int]:
        # 获取节点的邻居列表，返回一个包含邻居节点ID的列表
        return graph_state["adj"].get(node_id, [])

    def _trim_neighbors(self, graph_state: Dict, node_id: int):
        # 修剪邻居列表，确保不超过最大邻居数量
        neigh = graph_state["adj"].get(node_id, [])
        if len(neigh) > self.max_neighbors:
            graph_state["adj"][node_id] = neigh[-self.max_neighbors :]

    def _update_graph_state(self, graph_state: Dict, user_node: int, item_node: int, ts: float):
        #输入当前事件的用户节点ID、物品节点ID和时间戳，更新图状态，包括邻接表、边的最后时间和权重等信息
        # 更新图状态，包括邻接表、边的最后时间和权重等信息
        graph_state["adj"].setdefault(user_node, []).append(item_node)
        graph_state["adj"].setdefault(item_node, []).append(user_node)
        self._trim_neighbors(graph_state, user_node)
        self._trim_neighbors(graph_state, item_node)
        graph_state["edge_last_time"][(user_node, item_node)] = ts
        graph_state["edge_last_time"][(item_node, user_node)] = ts
        graph_state["edge_weight"][(user_node, item_node)] = graph_state["edge_weight"].get((user_node, item_node), 0.0) + 1.0
        graph_state["edge_weight"][(item_node, user_node)] = graph_state["edge_weight"].get((item_node, user_node), 0.0) + 1.0

    def _memory_update(self, cell_type: str, update_input: torch.Tensor, old_state: torch.Tensor) -> torch.Tensor:
        if self.memory_cell == "add":
            if cell_type == "user":
                return old_state + torch.tanh(self.add_linear_user(update_input))
            return old_state + torch.tanh(self.add_linear_item(update_input))

        if cell_type == "user":
            return self.user_cell(update_input, old_state)
        return self.item_cell(update_input, old_state)

    def _apply_gate(self, old_state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
        if self.memory_gate == "off":
            return new_state
        g = torch.sigmoid(self.gate_layer(torch.cat([old_state, new_state], dim=-1)))
        return g * new_state + (1 - g) * old_state

    def _predict_item_embedding(self, user_state: torch.Tensor) -> torch.Tensor:
        return self.predict_layer(user_state)

    def forward(
        self,
        user_ids: torch.Tensor,     # 用户ID张量，形状为 (batch_size,)
        item_ids: torch.Tensor,     # 商品ID张量，形状为 (batch_size,)
        timestamps: torch.Tensor,   # 交互时间戳张量，形状为 (batch_size,)
        features: torch.Tensor,     # 交互特征张量，形状为 (batch_size, feature_dim)
        query_time: float,          # 查询时间
        graph_ctx=None,             # 图状态上下文，包含动态图的状态信息
    ):
        if graph_ctx is None:
            raise ValueError("TemporalEventGNNJODIE requires graph_ctx dynamic state")
  
        graph_state = graph_ctx
        #图状态更新和维护都依赖于外部传入的 graph_ctx 参数，模型本身不保存图状态，而是通过 graph_ctx 来访问和更新图状态
        user_nodes, item_nodes = self._node_ids(user_ids, item_ids)

        uid = int(user_nodes[0].item())
        iid = int(item_nodes[0].item())
        ts = float(timestamps[0].item())
        #从批量张量中提取第一个样本的标量值

        old_user = self.memory[user_nodes].clone()
        old_item = self.memory[item_nodes].clone()
        # clone() 防止后续 memory 原地更新破坏 autograd 计算图

        last_u = self.last_time[user_nodes].clone()
        last_i = self.last_time[item_nodes].clone()

        du = (timestamps - last_u).unsqueeze(-1)
        di = (timestamps - last_i).unsqueeze(-1)
        # 计算时间差

        proj_user = self._project_time(old_user, du)
        proj_item = self._project_time(old_item, di)
        # 时间投影层，将时间差映射到嵌入维度

        # 查询阶段可使用 query_time 外推用户状态，训练时 query_time=timestamp
        d_query = (torch.tensor([query_time], dtype=timestamps.dtype, device=timestamps.device) - last_u).unsqueeze(-1)
        query_user = self._project_time(old_user, d_query)

        if self.message_mode == "peer":
            user_msg = proj_item
            item_msg = proj_user
        else:
            user_neighbors = self._neighbors(graph_state, uid)
            item_neighbors = self._neighbors(graph_state, iid)
            # 获取用户和物品的邻居列表

            user_msg = self.event_operator.event_aggregate(
                # 计算用户邻居的消息聚合
                center_idx=uid,
                center_emb=proj_user.squeeze(0),
                memory=self.memory,
                neighbors=user_neighbors,
                edge_last_time=graph_state["edge_last_time"],
                current_time=ts,
            ).unsqueeze(0)

            item_msg = self.event_operator.event_aggregate(
                # 聚合物品邻居的消息
                center_idx=iid,
                center_emb=proj_item.squeeze(0),
                memory=self.memory,
                neighbors=item_neighbors,
                edge_last_time=graph_state["edge_last_time"],
                current_time=ts,
            ).unsqueeze(0)

        user_input = torch.cat([proj_user, item_msg, features], dim=-1)
        item_input = torch.cat([proj_item, user_msg, features], dim=-1)
        # 计算用户和物品的输入向量

        if self.memory_cell == "lstm":
            user_c = self.user_cell_state[user_ids].clone()
            item_c = self.item_cell_state[item_ids].clone()
            new_user, new_user_c = self.user_cell(user_input, (old_user, user_c))
            new_item, new_item_c = self.item_cell(item_input, (old_item, item_c))
            self.user_cell_state[user_ids] = new_user_c.detach()
            self.item_cell_state[item_ids] = new_item_c.detach()

        else:
            new_user = self._memory_update("user", user_input, old_user)
            new_item = self._memory_update("item", item_input, old_item)

        new_user = self._apply_gate(old_user, new_user)
        new_item = self._apply_gate(old_item, new_item)
        # 应用门控机制，更新用户和物品的状态

        pred_item_emb = self._predict_item_embedding(query_user)

        self.memory[user_nodes] = new_user.detach()
        self.memory[item_nodes] = new_item.detach()
        self.last_time[user_nodes] = timestamps
        self.last_time[item_nodes] = timestamps

        if self.enable_graph_update:
            self._update_graph_state(graph_state, uid, iid, ts)

        return pred_item_emb, new_user, new_item


# 兼容旧工厂命名
HybridJODIE = TemporalEventGNNJODIE
