#!/usr/bin/env python3
import csv, os

configs = [
    ('async_1stage_3w',  '异步 1stage [3]'),
    ('async_2stage_2_1', '异步 2stage [2,1]'),
    ('async_2stage_1_2', '异步 2stage [1,2]'),
    ('sync_1stage_3w',   '同步 1stage [3]'),
    ('sync_2stage_2_1',  '同步 2stage [2,1]'),
    ('sync_2stage_1_2',  '同步 2stage [1,2]'),
]
seeds = [42, 43, 44]
root = 'outputs/alloc_compare'

print('%-22s %8s %9s %8s' % ('配置', 'avg时间', 'avg_MRR', 'seeds'))
print('-' * 52)
for d, label in configs:
    times, scores = [], []
    for seed in seeds:
        lb = os.path.join(root, f'seed_{seed}', d, 'leaderboard.csv')
        tl = os.path.join(root, f'seed_{seed}', d, 'timing_log.csv')
        if not os.path.exists(lb):
            continue
        lb_rows = list(csv.DictReader(open(lb)))
        tl_rows = list(csv.DictReader(open(tl))) if os.path.exists(tl) else []
        best = max((float(r['score']) for r in lb_rows if float(r.get('score', 0)) > 0), default=0)
        t = max(float(r['end_time_s']) for r in tl_rows) if tl_rows else 0
        scores.append(best)
        times.append(t)
    if scores:
        print('%-22s %8.0f %9.4f %8d' % (
            label,
            sum(times) / len(times),
            sum(scores) / len(scores),
            len(scores),
        ))
    else:
        print('%-22s %8s' % (label, 'N/A'))
