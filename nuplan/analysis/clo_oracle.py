"""GATE-CL-2 analysis: closed-loop best-of-modes oracle. Reads the 6 per-mode closed-
loop evals (each executed a fixed WTA mode), takes best-of-K CLS per token, and compares
to the deterministic baseline. Decides: is the SELECTOR the bottleneck (best-of-modes >>
top-scored 0.75, -> baseline) or are the MODES the ceiling (~0.75)?
"""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "nuplan"); sys.path.insert(0, "nuplan/analysis")
from analyze_moderation_v2 import read_cell_metric, latest_aggregator, wild_cluster_test
SC = "/scratch/patodia.pa/av-policy-lab"

per_mode = {}
for m in range(6):
    agg = latest_aggregator(f"{SC}/sim_results/clo_mode{m}")
    per_mode[m] = read_cell_metric(agg, "score")[0] if agg else {}
    print(f"mode {m}: {len(per_mode[m])} tokens")
det, dtypes = read_cell_metric(latest_aggregator(f"{SC}/merged_r0_full/det_route"), "score")
f4all = json.load(open(f"{SC}/features/f4/f4_scores_v11.json"))
f4 = {t: r["f4"] for t, r in f4all.items() if r.get("f4") is not None}
styp = {t: f4all[t].get("scenario_type", "?") for t in f4all}

toks = set(det) & set(f4)
for m in range(6):
    toks &= set(per_mode[m])
toks = sorted(toks)
if not toks:
    print("NO COMMON TOKENS YET"); sys.exit()
best = np.array([max(per_mode[m][t] for m in range(6)) for t in toks])
topm = np.array([per_mode[int(np.argmax([per_mode[mm].get(t, -1) for mm in range(6)]))][t] for t in toks])  # = best
detc = np.array([det[t] for t in toks])
permode_mean = {m: round(float(np.mean([per_mode[m][t] for t in toks])), 3) for m in range(6)}
x = np.array([f4[t] for t in toks]); g = np.array([styp.get(t, "?") for t in toks])
adv = best - detc
wc = wild_cluster_test(x, adv, g, B=4999)
res = {"n": len(toks), "per_mode_mean_CLS": permode_mean,
       "best_of_modes_mean_CLS": round(float(best.mean()), 3),
       "det_baseline_mean_CLS": round(float(detc.mean()), 3),
       "deployed_top_scored_ref": 0.75,
       "best_of_modes_minus_det": round(float(adv.mean()), 3),
       "frac_a_mode_beats_det": round(float((best > detc + 1e-6).mean()), 3),
       "moderation_bestofmodes_adv_vs_F4": wc,
       "verdict_hint": "best_of_modes >> 0.75 and -> det 0.86 => SELECTOR is the bottleneck (modes good); "
                       "~0.75 => MODES are the ceiling"}
print(json.dumps(res, indent=2))
Path(f"{SC}/clo_oracle.json").write_text(json.dumps(res, indent=2)); print("wrote clo_oracle.json")
