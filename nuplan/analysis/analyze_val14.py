"""Phase-1 Val14 analysis (ADR-050/054). Per-config CLS on Val14 under IDM, plus the
multimodality (WTA-det) and selector (sel-det) contrasts regressed on interaction-criticality
(s_inter) with scenario_type FIXED EFFECTS + wild-cluster bootstrap (clustered by scenario_type)
and TOST equivalence. This is the H1 interaction-criticality retest on the standard split.
Runs partial: prints whatever data has landed so far."""
import sys, glob, os, json
import numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric, wild_cluster_test, tost_equivalence

SC = "/scratch/patodia.pa/av-policy-lab"
BASE = f"{SC}/sim_results/eval/val14_zoo_r1"
CONFIGS = ["pdm", "det_route", "wta_route", "sel_route"]
COL = "score"

def collect(config):
    cls, typ = {}, {}
    for d in sorted(glob.glob(f"{BASE}/{config}_shard*")):
        a = latest_aggregator(d)
        if not a:
            continue
        v, t = read_cell_metric(a, COL)
        cls.update(v); typ.update(t)
    return cls, typ

data, types = {}, {}
print("=== per-config Val14 CLS (under IDM) ===")
for c in CONFIGS:
    cls, typ = collect(c)
    data[c] = cls; types.update(typ)
    if cls:
        vv = np.array(list(cls.values()))
        print(f"{c:10s} n={len(cls):4d} CLS={vv.mean():.4f}")
    else:
        print(f"{c:10s} n=0 (no aggregators yet)")

sinter = {}
sp = f"{SC}/eval_tokens/val_sinter.json"
if os.path.exists(sp):
    sinter = json.load(open(sp))
    print(f"s_inter tokens: {len(sinter)}")
else:
    print("val_sinter.json not present yet -- stratification skipped")

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
