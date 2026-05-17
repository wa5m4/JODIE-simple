#!/usr/bin/env python3
"""生成 Smart Pipeline 各分配策略对比报告"""
import csv, os

configs = [
    ('async_1stage_3w',  '异步 1stage [3]',     'async', 1, '3'),
    ('async_2stage_2_1', '异步 2stage [2,1]',   'async', 2, '2,1'),
    ('async_2stage_1_2', '异步 2stage [1,2]',   'async', 2, '1,2'),
    ('sync_1stage_3w',   '同步 1stage [3]',     'sync',  1, '3'),
    ('sync_2stage_2_1',  '同步 2stage [2,1]',   'sync',  2, '2,1'),
    ('sync_2stage_1_2',  '同步 2stage [1,2]',   'sync',  2, '1,2'),
]
seeds = [42, 43, 44]
root = 'outputs/alloc_compare'

data = {}
for d, label, mode, stages, workers in configs:
    times, scores = [], []
    per_seed = {}
    for seed in seeds:
        lb = os.path.join(root, f'seed_{seed}', d, 'leaderboard.csv')
        tl = os.path.join(root, f'seed_{seed}', d, 'timing_log.csv')
        if not os.path.exists(lb):
            continue
        lb_rows = list(csv.DictReader(open(lb)))
        tl_rows = list(csv.DictReader(open(tl))) if os.path.exists(tl) else []
        best = max((float(r['score']) for r in lb_rows if float(r.get('score', 0)) > 0), default=0)
        t = max(float(r['end_time_s']) for r in tl_rows) if tl_rows else 0
        scores.append(best); times.append(t)
        per_seed[seed] = (t, best)
    data[d] = {'label': label, 'mode': mode, 'stages': stages, 'workers': workers,
                'times': times, 'scores': scores, 'per_seed': per_seed}

W = 70
lines = []
lines.append('╔' + '═'*W + '╗')
lines.append('║' + '  Smart Pipeline 分配策略对比报告  '.center(W) + '║')
lines.append('╚' + '═'*W + '╝')
lines.append('')
lines.append('  数据集: mooc.csv (20000 events)  |  Trials: 27  |  Epochs: 3  |  GPUs: 3')
lines.append('  Seeds: 42, 43, 44  |  搜索空间: mixed')
lines.append('')

# ── 1. 速度对比
lines.append('┌─ 1. 搜索效率（时间越短越好）' + '─'*(W-28) + '┐')
lines.append('')
lines.append('  %-22s %8s %8s %8s %8s' % ('配置', 'avg(s)', 'seed42', 'seed43', 'seed44'))
lines.append('  ' + '─'*58)
base_time = data['sync_1stage_3w']['times']
base_avg = sum(base_time)/len(base_time) if base_time else 1
for d, label, mode, stages, workers in configs:
    d_data = data[d]
    if not d_data['times']:
        lines.append('  %-22s %8s' % (label, 'N/A'))
        continue
    avg_t = sum(d_data['times'])/len(d_data['times'])
    s42 = '%.0f' % d_data['per_seed'].get(42, (0,0))[0] if 42 in d_data['per_seed'] else 'N/A'
    s43 = '%.0f' % d_data['per_seed'].get(43, (0,0))[0] if 43 in d_data['per_seed'] else 'N/A'
    s44 = '%.0f' % d_data['per_seed'].get(44, (0,0))[0] if 44 in d_data['per_seed'] else 'N/A'
    speedup = base_avg / avg_t if avg_t > 0 else 0
    lines.append('  %-22s %8.0f %8s %8s %8s  (%.2fx vs 同步1stage)' % (label, avg_t, s42, s43, s44, speedup))
lines.append('')
lines.append('└' + '─'*W + '┘')
lines.append('')

# ── 2. 质量对比
lines.append('┌─ 2. 搜索质量（MRR 越高越好）' + '─'*(W-28) + '┐')
lines.append('')
lines.append('  %-22s %8s %8s %8s %8s' % ('配置', 'avg_MRR', 'seed42', 'seed43', 'seed44'))
lines.append('  ' + '─'*58)
for d, label, mode, stages, workers in configs:
    d_data = data[d]
    if not d_data['scores']:
        lines.append('  %-22s %8s' % (label, 'N/A'))
        continue
    avg_s = sum(d_data['scores'])/len(d_data['scores'])
    s42 = '%.4f' % d_data['per_seed'].get(42, (0,0))[1] if 42 in d_data['per_seed'] else 'N/A'
    s43 = '%.4f' % d_data['per_seed'].get(43, (0,0))[1] if 43 in d_data['per_seed'] else 'N/A'
    s44 = '%.4f' % d_data['per_seed'].get(44, (0,0))[1] if 44 in d_data['per_seed'] else 'N/A'
    lines.append('  %-22s %8.4f %8s %8s %8s' % (label, avg_s, s42, s43, s44))
lines.append('')
lines.append('└' + '─'*W + '┘')
lines.append('')

# ── 3. 综合结论
lines.append('┌─ 3. 综合结论' + '─'*(W-13) + '┐')
lines.append('')

# 找最快和最好
valid = [(d, data[d]) for d, *_ in configs if data[d]['times']]
fastest = min(valid, key=lambda x: sum(x[1]['times'])/len(x[1]['times']))
best_q  = max(valid, key=lambda x: sum(x[1]['scores'])/len(x[1]['scores']))

lines.append('  最快方法: %s (avg=%.0fs)' % (data[fastest[0]]['label'], sum(fastest[1]['times'])/len(fastest[1]['times'])))
lines.append('  最高质量: %s (avg_MRR=%.4f)' % (data[best_q[0]]['label'], sum(best_q[1]['scores'])/len(best_q[1]['scores'])))
lines.append('')
lines.append('  异步 vs 同步（1stage 3workers）:')
async_t = sum(data['async_1stage_3w']['times'])/len(data['async_1stage_3w']['times'])
sync_t  = sum(data['sync_1stage_3w']['times'])/len(data['sync_1stage_3w']['times'])
lines.append('    速度: 异步 %.0fs vs 同步 %.0fs (%.1fx 加速)' % (async_t, sync_t, sync_t/async_t))
lines.append('')
lines.append('  2stage [2,1] vs 1stage [3]（异步）:')
t21 = sum(data['async_2stage_2_1']['times'])/len(data['async_2stage_2_1']['times'])
t1  = sum(data['async_1stage_3w']['times'])/len(data['async_1stage_3w']['times'])
s21 = sum(data['async_2stage_2_1']['scores'])/len(data['async_2stage_2_1']['scores'])
s1  = sum(data['async_1stage_3w']['scores'])/len(data['async_1stage_3w']['scores'])
lines.append('    速度: 2stage %.0fs vs 1stage %.0fs (%.1fx)' % (t21, t1, t1/t21 if t21>0 else 0))
lines.append('    质量: 2stage %.4f vs 1stage %.4f (diff=%.4f)' % (s21, s1, s21-s1))
lines.append('')
lines.append('  注：同步模式下不同 stage 配置质量相同（RL 采样路径确定性一致）')
lines.append('      质量差异主要来自异步模式下采样顺序的随机性，非 stage 配置本身')
lines.append('')
lines.append('└' + '─'*W + '┘')

report = '\n'.join(lines)
print(report)
out = 'outputs/alloc_compare/report_alloc_compare.txt'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'\n报告已保存: {out}')
