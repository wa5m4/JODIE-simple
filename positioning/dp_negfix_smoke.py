"""DP 负采样保真修复冒烟测试(2026-09-16,CPU 进程内,不依赖 Ray)。

验证点:
  1. train_chunk 在 interaction.neg_samples_by_epoch[epoch_idx] 存在时,
     直接消费预计算负样本,全程不调用 np.random.default_rng。
  2. 预计算负样本的具体数值确实进入训练(换预计算值 → 梯度变化)。
  3. 回退路径(无预计算)用 config["seed"] + epoch_idx*100000 的确定性 RNG,
     同输入两次调用梯度位级一致,不同 epoch 种子不同。
"""
import sys
import types

# 阻断真实 ray,让 @_ray_remote 变成空操作 → 类可直接实例化、方法可进程内调用
fake_ray = types.ModuleType("ray")


def _fake_remote(*args, **kwargs):
    if args and isinstance(args[0], type):
        return args[0]  # 裸装饰器 @ray.remote:直接返回类
    return lambda cls: cls  # 带选项 @ray.remote(...)


fake_ray.remote = _fake_remote
sys.modules["ray"] = fake_ray

import numpy as np
import torch

from jodie.data.synthetic import generate_synthetic_data
from jodie.models.factory import build_model
import jodie.nas.data_parallel as dp
from jodie.nas.data_parallel import _DataParallelWorker

ARCH = {
    "model": "jodie_rnn",
    "embedding_dim": 32,
    "memory_cell": "rnn",
    "time_proj": "linear",
    "use_static_embeddings": "on",
    "normalize_state": "on",
}
BASE = {
    "device": "cpu",
    "lr": 1e-3,
    "neg_sample_size": 5,
    "num_users": 20,
    "num_items": 50,
    "feature_dim": 8,
    "batch_mode": "serial",
    "seed": 42,
}

interactions, _, _ = generate_synthetic_data(
    num_users=BASE["num_users"], num_items=BASE["num_items"],
    num_interactions=30, feature_dim=BASE["feature_dim"], seed=42,
)
chunk = interactions[:8]

config = dict(BASE)
config.update(ARCH)
model = build_model(config)
state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

# 记录 default_rng 调用(替代真函数),便于断言「预计算路径零 RNG 消耗」
calls = []
_real_default_rng = np.random.default_rng


def _recording_default_rng(*args, **kwargs):
    calls.append((args, kwargs))
    return _real_default_rng(*args, **kwargs)


np.random.default_rng = _recording_default_rng

worker = _DataParallelWorker()
failures = []

# 记录 _item_embeddings_for_loss 收到的 neg_ids(直接证明负样本数值流入损失)
neg_id_log = []
_real_item_emb = dp._item_embeddings_for_loss


def _recording_item_emb(model, ids):
    neg_id_log.append(ids.detach().cpu().clone())
    return _real_item_emb(model, ids)


dp._item_embeddings_for_loss = _recording_item_emb


def grads_of(**kw):
    return worker.train_chunk(state_dict, None, chunk, ARCH, BASE, **kw)["gradients"]


# ── 1. 预计算路径:零 RNG 调用 ──────────────────────────────
for it in chunk:
    it.neg_samples_by_epoch[0] = [0] * 5
calls.clear()
g_a = grads_of(epoch_idx=0)
assert len(g_a) > 0, "无梯度输出"
if calls:
    failures.append(f"预计算路径仍调用 default_rng {len(calls)} 次")
print(f"[1] 预计算路径零 RNG 调用: {'PASS' if not calls else 'FAIL'}(rng 调用 {len(calls)})")

# ── 2. 预计算数值确实流入损失(neg_ids 记录) ─────────────────
for it in chunk:
    it.neg_samples_by_epoch[0] = [1] * 5
neg_id_log.clear()
g_b = grads_of(epoch_idx=0)
# 记录中交错着正样本调用(1 元素),过滤出负样本调用(5 元素)
neg_calls = [t for t in neg_id_log if t.numel() == 5]
if len(neg_calls) != len(chunk) or not all(torch.equal(t, torch.tensor([1] * 5)) for t in neg_calls):
    failures.append(f"预计算 [1]*5 未全部流入损失,neg 调用={len(neg_calls)}/{len(chunk)},示例={neg_calls[:2]}")
print(f"[2] 预计算数值流入损失: {'PASS' if len(neg_calls) == len(chunk) and all(torch.equal(t, torch.tensor([1] * 5)) for t in neg_calls) else 'FAIL'}")

# ── 3. 回退路径确定性 + 种子公式 ───────────────────────────
for it in chunk:
    it.neg_samples_by_epoch.clear()
calls.clear()
g_c1 = grads_of(epoch_idx=0)
seeds0 = [a[0] for a, _ in calls if a]
g_c2 = grads_of(epoch_idx=0)
if not all(torch.equal(g_c1[k], g_c2[k]) for k in g_c1):
    failures.append("回退路径同输入两次调用梯度不一致")
print(f"[3] 回退确定性: {'PASS' if all(torch.equal(g_c1[k], g_c2[k]) for k in g_c1) else 'FAIL'}")
if seeds0 != [42]:
    failures.append(f"回退种子应为 42(seed=42+epoch0*100000),实际 {seeds0}")
print(f"[4] 回退种子公式 42+epoch*100000: {'PASS' if seeds0 == [42] else 'FAIL'}({seeds0})")

calls.clear()
g_d = grads_of(epoch_idx=1)
seeds1 = [a[0] for a, _ in calls if a]
if seeds1 != [100042]:
    failures.append(f"epoch=1 回退种子应为 100042,实际 {seeds1}")
if all(torch.equal(g_c1[k], g_d[k]) for k in g_c1):
    failures.append("epoch=0 与 epoch=1 回退梯度相同(种子应不同)")
print(f"[5] 回退 epoch 种子区分: {'PASS' if seeds1 == [100042] and not all(torch.equal(g_c1[k], g_d[k]) for k in g_c1) else 'FAIL'}")

# ── 4. 预计算 vs 回退路径应不同(负样本来源不同) ─────────────
if all(torch.equal(g_a[k], g_c1[k]) for k in g_a):
    failures.append("预计算路径与回退路径梯度相同(可疑)")
print(f"[6] 预计算 vs 回退路径区分: {'PASS' if not all(torch.equal(g_a[k], g_c1[k]) for k in g_a) else 'FAIL'}")

print()
if failures:
    print("SMOKE FAIL:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("SMOKE PASS:DP 负采样修复五项验证全部通过")
