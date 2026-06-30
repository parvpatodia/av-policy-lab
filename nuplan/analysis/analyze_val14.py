"""Phase-1 zoo analysis (ADR-050/054/056). Per-config CLS + multimodality (WTA-det) and
selector contrasts regressed on interaction-criticality (s_inter) with scenario_type FIXED
EFFECTS + wild-cluster bootstrap + TOST equivalence. Usage:
  analyze_val14.py [split] [sinter_file]
  split default 'val14'; sinter_file default 'val_sinter.json' for val14 else '<split>_sinter.json'.
Robust: prints whatever data has landed; skips a contrast if data missing."""
import sys, glob, os, json
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric, wild_cluster_test, tost_equivalence

SC = "/scratch/patodia.pa/av-policy-lab"
split = sys.argv[1] if len(sys.argv) > 1 else "val14"
sinter_file = sys.argv[2] if len(sys.argv) > 2 else ("val_sinter.json" if split == "val14" else f"{split}_sinter.json")
BASE = f"{SC}/sim_results/eval/{split}_zoo_r1"
CONFIGS = ["pdm", "det_route", "wta_route", "sel_route"]
COL = "score"

def collect(config):
    cls, typ = {}, {}
    for d in sorted(glob.glob(f"{BASE}/{config}_shard*")):
        a = latest_aggregator(d)
        if not a:
            continue
        try:
            v, t = read_cell_metric(a, COL)
        except Exception as e:
            print(f"  WARN skip {os.path.basename(d)}: {e}"); continue
        cls.update(v); typ.update(t)
    return cls, typ

data, types = {}, {}
print(f"=== {split} per-config CLS (under IDM) ===")
for c in CONFIGS:
    cls, typ = collect(c)
    data[c] = cls; types.update(typ)
    print(f"{c:10s} n={len(cls):4d} CLS={np.mean(list(cls.values())):.4f}" if cls else f"{c:10s} n=0")

sinter = {}
sp = f"{SC}/eval_tokens/{sinter_file}"
if os.path.exists(sp):
    sinter = json.load(open(sp)); print(f"s_inter ({sinter_file}) tokens: {len(sinter)}")
else:
    print(f"{sinter_file} not present -- stratification skipped")

def contrast(a, b, name):
    if not (data.get(a) and data.get(b) and sinter):
        print(f"[{name}] skipped (missing data)"); return
    toks = sorted(set(data[a]) & set(data[b]) & set(sinter))
    if len(toks) < 30:
        print(f"[{name}] only {len(toks)} paired tokens -- skip"); return
    delta = np.array([data[a][t] - data[b][t] for t in toks])
    x = np.array([float(sinter[t]) for t in toks])
    g = np.array([types.get(t, "?") for t in toks])
    print(f"[{name}] n={len(toks)} mean_delta={delta.mean():+.4f} (CLS_{a}-CLS_{b})")
    res = wild_cluster_test(x, delta, g, B=9999)
    print(f"   slope(delta~s_inter): beta1={res['beta1']:+.4f} se={res['cluster_se']:.4f} "
          f"t={res['t']:+.2f} p2={res['p_wcr_twosided']:.3f} clusters={res['n_clusters']}")
    to = tost_equivalence(res['beta1'], res['cluster_se'], len(toks), margin=0.05)
    print(f"   TOST(|slope|<0.05): p={to['p_tost']:.3f} ci90={to['ci90']} equivalent={to['equivalent']}")

print("=== contrasts regressed on interaction-criticality (s_inter) ===")
contrast("wta_route", "det_route", "MULTIMODALITY (WTA-det)")
contrast("sel_route", "det_route", "SELECTOR (sel-det)")
contrast("sel_route", "wta_route", "SELECTOR vs WTA (sel-wta)")
