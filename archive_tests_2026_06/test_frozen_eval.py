#!/usr/bin/env python3
"""
验证冻结评估模式是否正确工作
测试embeddings在frozen=True时不会被更新
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.jodie_rnn import JODIERNN
from data.synthetic import Interaction
from models.training import evaluate_ranking_metrics

# 创建小型测试数据
NUM_USERS, NUM_ITEMS = 10, 5
test_interactions = [
    Interaction(
        timestamp=float(i),
        user_id=i % NUM_USERS,
        item_id=i % NUM_ITEMS,
        features=torch.randn(4),
    )
    for i in range(50)
]

# 创建模型
model = JODIERNN(
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    embedding_dim=16,
    feature_dim=4,
    cell_type="rnn"
)
model.eval()

print("="*60)
print("测试冻结评估模式")
print("="*60)

# 测试1: 非冻结模式（默认）
print("\n[测试1] 非冻结模式 (frozen=False)")
initial_user_emb = model.user_embeddings.data.clone()
initial_item_emb = model.item_embeddings.data.clone()

metrics_unfrozen = evaluate_ranking_metrics(
    model, test_interactions, k=3, frozen=False
)

final_user_emb = model.user_embeddings.data
final_item_emb = model.item_embeddings.data

user_changed = not torch.allclose(initial_user_emb, final_user_emb)
item_changed = not torch.allclose(initial_item_emb, final_item_emb)

print(f"  User embeddings changed: {user_changed}")
print(f"  Item embeddings changed: {item_changed}")
print(f"  MRR: {metrics_unfrozen['mrr']:.4f}")
print(f"  Recall@3: {metrics_unfrozen['recall_at_k']:.4f}")

# 测试2: 冻结模式
print("\n[测试2] 冻结模式 (frozen=True)")
initial_user_emb = model.user_embeddings.data.clone()
initial_item_emb = model.item_embeddings.data.clone()

metrics_frozen = evaluate_ranking_metrics(
    model, test_interactions, k=3, frozen=True
)

final_user_emb = model.user_embeddings.data
final_item_emb = model.item_embeddings.data

user_changed = not torch.allclose(initial_user_emb, final_user_emb)
item_changed = not torch.allclose(initial_item_emb, final_item_emb)

print(f"  User embeddings changed: {user_changed}")
print(f"  Item embeddings changed: {item_changed}")
print(f"  MRR: {metrics_frozen['mrr']:.4f}")
print(f"  Recall@3: {metrics_frozen['recall_at_k']:.4f}")

# 验证结果
print("\n" + "="*60)
print("验证结果")
print("="*60)

if user_changed or item_changed:
    print("❌ 失败: 冻结模式下embeddings仍然被更新")
    sys.exit(1)
else:
    print("✅ 成功: 冻结模式正确阻止了embedding更新")

# 性能对比
print(f"\n性能对比:")
print(f"  非冻结MRR: {metrics_unfrozen['mrr']:.4f}")
print(f"  冻结MRR:   {metrics_frozen['mrr']:.4f}")
print(f"  差异:      {metrics_unfrozen['mrr'] - metrics_frozen['mrr']:.4f}")

print("\n✅ 所有测试通过！")
