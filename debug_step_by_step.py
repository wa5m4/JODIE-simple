"""
MOOC 数据，单架构，Serial vs Pipeline 逐步对比。
每步打印权重指纹，精确定位首次偏差。
"""
import torch, ray, copy
from jodie.data.public_dataset import load_public_dataset
from jodie.data.temporal_partition import build_partition_plan
from jodie.models.factory import build_model
from jodie.training.loops import train_partition_bpr, BPRLoss, reset_model_state
from jodie.nas.ray_pipeline import _safe_ray_init, PartitionShardWorker, PipelineModelPayload
from jodie.nas.search_space import sanitize_config

MAX_EVENTS, FEATURE_DIM = 5000, 4
SEED, EP, PS, N_STAGES = 42, 2, 500, 3

ARCH = {
    'model': 'jodie_rnn', 'embedding_dim': 64, 'memory_cell': 'rnn',
    'time_proj': 'off', 'use_static_embeddings': 'off',
    'normalize_state': 'off',
    'event_agg': 'none', 'agg_activation': 'none', 'attn_type': 'dot',
    'time_decay': 'none', 'hidden_dim': 0, 'memory_gate': 'off',
    'enable_event_agg': 'off', 'enable_graph_update': 'off',
    'message_mode': 'peer', 'msg_linear': 'off',
}

print("=" * 60)
print("加载 MOOC 数据...")
ints_raw, num_users, num_items = load_public_dataset(
    dataset_name='public_csv', dataset_dir='data/public',
    feature_dim=FEATURE_DIM, max_events=MAX_EVENTS,
    local_data_path='data/public/mooc.csv')
ints = sorted(ints_raw, key=lambda x: x.timestamp)
tn = int(len(ints) * 0.7)
vn = int(len(ints) * (0.7 + 0.1))
plan = build_partition_plan(ints[:tn], ints[tn:vn], ints[vn:],
                            partition_size=PS, strategy='count')
tps = sorted(plan.get_split_partitions('train'),
             key=lambda p: (float(p.start_ts), p.partition_id))
print(f"  {len(ints)} events, {num_users} users, {num_items} items")
print(f"  {len(tps)} train partitions")

# 3-stage 分组
groups = []; b = len(tps) // N_STAGES; r = len(tps) % N_STAGES; s = 0
for i in range(N_STAGES):
    sz = b + (1 if i < r else 0)
    groups.append(tps[s:s + sz]); s += sz
print(f"  Stage groups: {[len(g) for g in groups]}")

def make_config():
    c = {
        'dataset': 'public_csv', 'num_users': num_users, 'num_items': num_items,
        'max_events': MAX_EVENTS, 'feature_dim': FEATURE_DIM, 'lr': 1e-3,
        'neg_sample_size': 5, 'k': 10, 'selection_metric': 'mrr', 'device': 'cpu',
        'seed': SEED, 'partition_size': PS, 'partition_strategy': 'count',
        'batch_mode': 'serial', 'train_batch_size': 32, 'max_neighbors': 0,
    }
    c.update(ARCH)
    return sanitize_config(c)

CONFIG = make_config()

def wsum(model, label=""):
    """返回 (参数sum, 缓冲sum, user_emb_sum, item_emb_sum)"""
    ps = sum(p.data.float().sum().item() for p in model.parameters())
    bs = sum(b.float().sum().item() for b in model.buffers())
    ue = model.user_embeddings.sum().item()
    ie = model.item_embeddings.sum().item()
    if label:
        print(f"  [{label}] params={ps:.4f} buffers={bs:.4f} user_emb={ue:.4f} item_emb={ie:.4f}")
    return ps, bs, ue, ie

def compare(label, s_ps, s_bs, s_ue, s_ie, t_ps, t_bs, t_ue, t_ie):
    pd = abs(s_ps - t_ps)
    bd = abs(s_bs - t_bs)
    ud = abs(s_ue - t_ue)
    ok = pd < 1e-4 and bd < 1e-4 and ud < 1e-4
    s = "✅" if ok else f"❌ Δp={pd:.4f} Δb={bd:.4f} Δu={ud:.4f}"
    print(f"  [{label}] {s}")
    return ok

# ================================================================
print(f"\n{'='*60}")
print("Step 0: Serial 确定性验证 (同代码跑两次)")
print("=" * 60)

def serial_full(epochs=EP):
    torch.manual_seed(SEED)
    m = build_model(CONFIG)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    c = BPRLoss()
    for ep in range(epochs):
        if ep > 0: reset_model_state(m)
        for p in tps:
            train_partition_bpr(m, p, opt, c, neg_sample_size=5, graph_ctx=None,
                                seed=SEED + ep * 100000 + p.partition_id, progress_every=1000)
    return m

m_a = serial_full()
a = wsum(m_a, "Serial Run1")
m_b = serial_full()
b = wsum(m_b, "Serial Run2")
compare("determinism", *a, *b)

# ================================================================
print(f"\n{'='*60}")
print("Step 1: 同进程 state_dict round-trip (单epoch，不切stage)")
print("=" * 60)

# 获取初始状态
torch.manual_seed(SEED)
m_init = build_model(CONFIG)
sd_init = {k: v.clone() for k, v in m_init.state_dict().items()}
rt_init_raw = m_init.export_runtime_state()
rt_init = {k: v.clone() for k, v in rt_init_raw.items()}

# Serial: 1 epoch, 全部 partition
torch.manual_seed(SEED)
m_s0 = build_model(CONFIG)
opt = torch.optim.Adam(m_s0.parameters(), lr=1e-3)
c = BPRLoss()
for p in tps:
    train_partition_bpr(m_s0, p, opt, c, neg_sample_size=5, graph_ctx=None,
                        seed=SEED + 0 * 100000 + p.partition_id, progress_every=1000)
s0 = wsum(m_s0, "Serial 1ep")

# 同进程模拟: build → load → train (等价 Pipeline 的 _build_model 行为)
torch.manual_seed(SEED + 999)  # 不同RNG验证load覆盖
m_t0 = build_model(CONFIG)
m_t0.load_state_dict(sd_init)
m_t0.import_runtime_state(rt_init)
opt_t = torch.optim.Adam(m_t0.parameters(), lr=1e-3)
for p in tps:
    train_partition_bpr(m_t0, p, opt_t, c, neg_sample_size=5, graph_ctx=None,
                        seed=SEED + 0 * 100000 + p.partition_id, progress_every=1000)
t0 = wsum(m_t0, "同进程load+train 1ep")
compare("同进程1ep", *s0, *t0)

# ================================================================
print(f"\n{'='*60}")
print("Step 2: 同进程 2-epoch (不切stage，epoch间 reset_state + fresh optimizer)")
print("=" * 60)

m_s2 = serial_full(2)
s2 = wsum(m_s2, "Serial 2ep")

# 同进程: load → train ep0 → reset → train ep1
m_t2 = build_model(CONFIG)
m_t2.load_state_dict(sd_init)
m_t2.import_runtime_state(rt_init)
opt_t2 = torch.optim.Adam(m_t2.parameters(), lr=1e-3)
for ep in range(2):
    if ep > 0:
        # epoch boundary: save state_dict, rebuild, reset_state, fresh optimizer
        sd_save = {k: v.clone() for k, v in m_t2.state_dict().items()}
        m_t2 = build_model(CONFIG)
        m_t2.load_state_dict(sd_save)
        m_t2.reset_state()
        opt_t2 = torch.optim.Adam(m_t2.parameters(), lr=1e-3)
    for p in tps:
        train_partition_bpr(m_t2, p, opt_t2, c, neg_sample_size=5, graph_ctx=None,
                            seed=SEED + ep * 100000 + p.partition_id, progress_every=1000)
t2 = wsum(m_t2, "同进程 2ep(opt重置)")
compare("同进程2ep", *s2, *t2)

# ================================================================
print(f"\n{'='*60}")
print("Step 3: 同进程 3-stage 2-epoch (模拟完整 Pipeline 但同进程)")
print("=" * 60)

m_p3 = build_model(CONFIG)
m_p3.load_state_dict(sd_init)
m_p3.import_runtime_state(rt_init)
opt_p3 = torch.optim.Adam(m_p3.parameters(), lr=1e-3)

for ep in range(2):
    for si in range(N_STAGES):
        for p in groups[si]:
            train_partition_bpr(m_p3, p, opt_p3, c, neg_sample_size=5, graph_ctx=None,
                                seed=SEED + ep * 100000 + p.partition_id, progress_every=1000)
        if si < N_STAGES - 1:
            # stage boundary
            sd_s = {k: v.clone() for k, v in m_p3.state_dict().items()}
            osd_s = opt_p3.state_dict()
            rt_s = m_p3.export_runtime_state()
            m_p3 = build_model(CONFIG)
            m_p3.load_state_dict(sd_s)
            m_p3.import_runtime_state(rt_s)
            opt_p3 = torch.optim.Adam(m_p3.parameters(), lr=1e-3)
            opt_p3.load_state_dict(osd_s)
    if ep < 1:
        # epoch boundary: reset, fresh optimizer
        sd_e = {k: v.clone() for k, v in m_p3.state_dict().items()}
        m_p3 = build_model(CONFIG)
        m_p3.load_state_dict(sd_e)
        m_p3.reset_state()
        opt_p3 = torch.optim.Adam(m_p3.parameters(), lr=1e-3)

p3 = wsum(m_p3, "同进程3st2ep")
compare("同进程3st2ep", *s2, *p3)

# ================================================================
print(f"\n{'='*60}")
print("Step 4: Ray 3-stage 2-epoch (真实 Pipeline)")
print("=" * 60)

if ray.is_initialized(): ray.shutdown()
_safe_ray_init(ignore_reinit_error=True)
WorkerCls = ray.remote(num_cpus=1.0)(PartitionShardWorker)

torch.manual_seed(SEED)
m0 = build_model(CONFIG)
rt0 = m0.export_runtime_state()
payload = PipelineModelPayload(
    trial_id=0, arch_config=CONFIG,
    model_state_dict={k: v.cpu() for k, v in m0.state_dict().items()},
    runtime_state={k: v.cpu() for k, v in rt0.items()},
    graph_state=None, optimizer_state=None, seed=SEED)

for ep in range(2):
    if ep > 0:
        payload = PipelineModelPayload(
            trial_id=0, arch_config=CONFIG,
            model_state_dict=payload.model_state_dict,
            runtime_state=None, graph_state=None,
            optimizer_state=payload.optimizer_state, seed=SEED)
    for si in range(N_STAGES):
        w = WorkerCls.remote(groups[si], CONFIG)
        ref = w.run_train_stage_batch.remote(
            payload, [p.partition_id for p in groups[si]], use_bpr=True, num_epochs=1)
        payload = ray.get(ref)
        ray.kill(w)

m_ray_final = build_model(CONFIG)
m_ray_final.load_state_dict({k: v for k, v in payload.model_state_dict.items()})
if payload.runtime_state:
    m_ray_final.import_runtime_state({k: v for k, v in payload.runtime_state.items()})

r4 = wsum(m_ray_final, "Ray 3st2ep")
compare("Ray 3st2ep", *s2, *r4)

# ================================================================
print(f"\n{'='*60}")
print("Step 5: 同进程逐 stage 对比 (在第一次 stage 边界对比同进程 vs Ray)")
print("=" * 60)

# 同进程 epoch0 stage0
m_s0s0 = build_model(CONFIG)
m_s0s0.load_state_dict(sd_init)
m_s0s0.import_runtime_state(rt_init)
opt_s0s0 = torch.optim.Adam(m_s0s0.parameters(), lr=1e-3)
for p in groups[0]:
    train_partition_bpr(m_s0s0, p, opt_s0s0, c, neg_sample_size=5, graph_ctx=None,
                        seed=SEED + 0 * 100000 + p.partition_id, progress_every=1000)
s0s0 = wsum(m_s0s0, "同进程 ep0stage0")

# Ray epoch0 stage0
torch.manual_seed(SEED)
m0b = build_model(CONFIG)
rt0b = m0b.export_runtime_state()
payload_b = PipelineModelPayload(
    trial_id=0, arch_config=CONFIG,
    model_state_dict={k: v.cpu() for k, v in m0b.state_dict().items()},
    runtime_state={k: v.cpu() for k, v in rt0b.items()},
    graph_state=None, optimizer_state=None, seed=SEED)

w_ray0 = WorkerCls.remote(groups[0], CONFIG)
ref0 = w_ray0.run_train_stage_batch.remote(
    payload_b, [p.partition_id for p in groups[0]], use_bpr=True, num_epochs=1)
payload_s1 = ray.get(ref0)

# 从 Ray payload 重建模型
m_ray_s0 = build_model(CONFIG)
m_ray_s0.load_state_dict({k: v for k, v in payload_s1.model_state_dict.items()})
if payload_s1.runtime_state:
    m_ray_s0.import_runtime_state({k: v for k, v in payload_s1.runtime_state.items()})

r0s0 = wsum(m_ray_s0, "Ray ep0stage0")
compare("第一次stage后 同进程vsRay", *s0s0, *r0s0)

ray.kill(w_ray0)
ray.shutdown()

print(f"\n{'='*60}")
print("结论")
print("=" * 60)
print("如果 Step1-3 全PASS但 Step4 FAIL → 问题在 Ray 跨进程序列化")
print("如果 Step3 就 FAIL → 问题在同进程 3-stage 重建")
print("如果 Step5 FAIL → 问题在第一次 Ray stage 传递就出现")
