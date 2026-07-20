"""
JODIE-simple: 时序图神经网络架构搜索 (NAS) 框架。

为 JODIE 风格的时序动态图模型搜索最优架构，
支持串行、数据并行、Ray 流水线并行三种执行模式。
"""

# 数据层
from jodie.data.synthetic import Interaction, generate_synthetic_data, init_dynamic_graph_state

# 模型层
from jodie.models.factory import build_model
from jodie.models.hybrid_jodie import TemporalEventGNNJODIE
from jodie.models.jodie_rnn import JODIERNN

# NAS 层 (需要 ray，未安装时跳过)
try:
    from jodie.nas.search_space import get_search_space
    from jodie.nas.controller import RandomGraphNASController, RLGraphNASController
    from jodie.nas.trainer import GraphNASTrainer
except ImportError:
    get_search_space = None  # type: ignore
    RandomGraphNASController = None  # type: ignore
    RLGraphNASController = None  # type: ignore
    GraphNASTrainer = None  # type: ignore
