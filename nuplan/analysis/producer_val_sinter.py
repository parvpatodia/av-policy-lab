"""Compute per-token interaction-criticality (s_inter) from f0 shards via
f4_score.score_sample (map-API-free). Aggregates per token as PEAK over the scene's
samples. Usage:
  producer_val_sinter.py [shard_glob] [out_json]
  defaults: f0_val shards -> val_sinter.json."""
import sys, glob, json
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
import torch
from features.f4_score import score_sample

SHARDS = sys.argv[1] if len(sys.argv) > 1 else "/scratch/patodia.pa/av-policy-lab/features/f0_val/task_*/scene_shard_*.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/scratch/patodia.pa/av-policy-lab/eval_tokens/val_sinter.json"

per_tok = {}
n_shards = 0
for sp in sorted(glob.glob(SHARDS)):
    n_shards += 1
    data = torch.load(sp, map_location="cpu", weights_only=False)
    cfg = data.get("config", {})
    for s in data.get("samples", []):
        tok = s.get("scenario_token")
        if not tok:
            continue
        r = score_sample(s, cfg)
        if r.get("excluded", 0.0) == 1.0:
            continue
        si = r.get("s_inter")
        if si is None or (isinstance(si, float) and np.isnan(si)):
            continue
        per_tok.setdefault(tok, []).append(float(si))

agg = {t: float(np.max(v)) for t, v in per_tok.items() if v}
json.dump(agg, open(OUT, "w"))
print("shards", n_shards, "tokens_with_sinter", len(agg), "->", OUT)
if agg:
    vals = np.array(list(agg.values()))
    print("s_inter mean %.3f p50 %.3f p90 %.3f max %.3f" %
          (vals.mean(), np.percentile(vals, 50), np.percentile(vals, 90), vals.max()))
