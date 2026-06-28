"""H1 re-test against a PRESENT, scene-adaptive multimodal treatment (the RL policy).
Delta_i = CLS(RL_multimodal)_i - CLS(deterministic)_i ; regress Delta ~ F4 with the
pre-registered inference (scenario_type fixed effects + wild-cluster bootstrap). The
SLOPE tests whether the multimodal-vs-unimodal closed-loop gap GROWS with interaction-
criticality (H1, predicted > 0); robust to the constant maturity-gap intercept (ADR-042).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "nuplan"); sys.path.insert(0, "nuplan/analysis")
from analysis.analyze_moderation_v2 import read_cell_metric, latest_aggregator, wild_cluster_test, tost_equivalence

SC = "/scratch/patodia.pa/av-policy-lab"
rl, rtypes = read_cell_metric(latest_aggregator(f"{SC}/sim_results/rl_h1_eval"), "score")
det, dtypes = read_cell_metric(latest_aggregator(f"{SC}/merged_r0_full/det_route"), "score")
f4all = json.load(open(f"{SC}/features/f4/f4_scores_v11.json"))
f4 = {t: r["f4"] for t, r in f4all.items() if r.get("f4") is not None}
styp = {t: f4all[t].get("scenario_type", "?") for t in f4all}

toks = sorted(set(rl) & set(det) & set(f4))
print(f"matched tokens: {len(toks)} (RL {len(rl)}, det {len(det)})")
d = np.array([rl[t] - det[t] for t in toks])
x = np.array([f4[t] for t in toks])
g = np.array([styp.get(t, "?") for t in toks])
print(f"mean CLS: RL {np.mean([rl[t] for t in toks]):.3f}  det {np.mean([det[t] for t in toks]):.3f}  "
      f"mean Delta {d.mean():+.3f}")
wc = wild_cluster_test(x, d, g, B=4999)
wc["tost"] = tost_equivalence(wc["beta1"], wc["cluster_se"], len(toks) - 2, 0.05)
print(json.dumps(wc, indent=2))
print("\nH1 predicts beta1 > 0 (multimodal advantage grows with interaction-criticality).")
print("one-sided p for H1:", round(wc["p_wcr_onesided_gt"], 3))
Path(f"{SC}/h1_retest_result.json").write_text(json.dumps({
    "n": len(toks), "mean_cls_rl": float(np.mean([rl[t] for t in toks])),
    "mean_cls_det": float(np.mean([det[t] for t in toks])), "mean_delta": float(d.mean()),
    **wc}, indent=2))
print("wrote h1_retest_result.json")
