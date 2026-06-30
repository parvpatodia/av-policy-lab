"""Collect per-config Val14 CLS across the 16 shards of the val14_zoo array.
Robust: skips unreadable/partial shard aggregators (prints a warning) so one bad
parquet cannot block the whole result. Reports n + mean CLS per config."""
import sys, glob, os
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric

BASE = "/scratch/patodia.pa/av-policy-lab/sim_results/eval/val14_zoo_r1"
configs = {}
shards_ok = {}
shards_bad = {}
for d in sorted(glob.glob(BASE + "/*_shard*")):
    name = os.path.basename(d).rsplit("_shard", 1)[0]
    a = latest_aggregator(d)
    if not a:
        shards_bad[name] = shards_bad.get(name, 0) + 1
        continue
    try:
        dd = read_cell_metric(a, "score")[0]
    except Exception as e:
        print(f"  WARN skip {os.path.basename(d)}: {e}")
        shards_bad[name] = shards_bad.get(name, 0) + 1
        continue
    configs.setdefault(name, {}).update({t: float(v) for t, v in dd.items()})
    shards_ok[name] = shards_ok.get(name, 0) + 1

print("config            shards_ok  n_tokens  CLS")
for name in sorted(configs):
    v = list(configs[name].values())
    print(f"{name:16s}  {shards_ok.get(name,0):2d}/16     {len(v):5d}   {np.mean(v):.4f}  (bad_shards={shards_bad.get(name,0)})")
