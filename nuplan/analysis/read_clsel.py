import sys, numpy as np
sys.path.insert(0, "/home/patodia.pa/av-policy-lab")
sys.path.insert(0, "/home/patodia.pa/av-policy-lab/nuplan/analysis")
from analyze_moderation_v2 import latest_aggregator, read_cell_metric
base = "/scratch/patodia.pa/av-policy-lab/sim_results/clsel_eval"
agg = latest_aggregator(base)
print("agg:", agg)
d = read_cell_metric(agg, "score")[0] if agg else {}
vals = [float(v) for v in d.values()]
print("n:", len(vals))
print("deployed_selector_CLS_mean:", round(float(np.mean(vals)), 4) if vals else None)
print("refs: prev_deployed 0.75 | det 0.868 | oracle_best_of_modes 0.883")
