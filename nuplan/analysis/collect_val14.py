"""Collect per-config Val14 CLS across the 16 shards of the val14_zoo array.
Unions each config's per-token scores from its shard aggregators and reports n + mean CLS.
Prints partial results as shards finish."""
import sys, glob, os
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric

BASE = "/scratch/patodia.pa/av-policy-lab/sim_results/eval/val14_zoo_r1"
configs = {}
shards_seen = {}
for d in sorted(glob.glob(BASE + "/*_shard*")):
    name = os.path.basename(d).rsplit("_shard", 1)[0]
    a = latest_aggregator(d)
    if not a:
        continue
    dd = read_cell_metric(a, "score")[0]
    configs.setdefault(name, {}).update({t: float(v) for t, v in dd.items()})
    shards_seen[name] = shards_seen.get(name, 0) + 1

print("config            shards  n_tokens  CLS")
for name in sorted(configs):
    v = list(configs[name].values())
    print(f"{name:16s}  {shards_seen[name]:2d}/16    {len(v):5d}   {np.mean(v):.4f}")
