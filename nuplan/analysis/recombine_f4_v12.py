"""Recombine F4 v1.2: take s_inter (+ g_stop) recomputed by the patched score_f4
pass_shards, keep the unchanged map components (s_branch, s_lane) from v11, and
re-combine. Avoids the heavy lane-graph map pass since only s_inter changed.
Also prints the artifact-removal impact (how much high-band mass the fix removes)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "nuplan"); sys.path.insert(0, "nuplan/features")
from features.f4_score import combine, F4_VERSION

F4 = Path("/scratch/patodia.pa/av-policy-lab/features/f4")
v11 = json.load(open(F4 / "f4_scores_v11.json"))
shard = json.load(open(F4 / "shard_scores_v12.json"))   # patched s_inter + g_stop

# WHY s_lane=0.0: v1.1 (ADR-016) DROPPED S_lane from the scalar; combine() still
# accepts it (audit m1 footgun) and the canonical scorer passes 0.0. Passing the
# stored raw s_lane would resurrect the deprecated v1.0 formula. Reproduce v11 f4
# from stored components with s_lane=0 to PROVE we match before changing s_inter.
SLANE = 0.0
max_err = max(abs(combine(r["s_branch"], SLANE, r["s_inter"], r["g_stop"]) - r["f4"])
              for r in v11.values() if not r.get("excluded"))
assert max_err < 1e-6, f"v11 f4 not reproduced with s_lane=0 (max_err={max_err}); wrong combine usage"
print(f"v11 f4 reproduced from components (s_lane=0), max_err={max_err:.2e}")

out, changed = {}, 0
for tok, r in v11.items():
    if r.get("excluded"):
        out[tok] = r; continue
    sh = shard.get(tok)
    if sh is None or sh.get("excluded"):
        out[tok] = r; continue
    si_new, gs = sh["s_inter"], sh["g_stop"]
    nr = dict(r)
    nr["s_inter"], nr["g_stop"] = si_new, gs
    nr["f4"] = combine(r["s_branch"], SLANE, si_new, gs)   # s_lane=0 per v1.1
    nr["f4_version"] = F4_VERSION
    out[tok] = nr
    if abs(si_new - r["s_inter"]) > 1e-9:
        changed += 1
json.dump(out, open(F4 / "f4_scores_v12.json", "w"))
print(f"recombined {len(out)} tokens; s_inter changed in {changed}")

incl = [t for t, r in v11.items() if not r.get("excluded") and t in shard and not shard[t].get("excluded")]
si11 = np.array([v11[t]["s_inter"] for t in incl]); si12 = np.array([out[t]["s_inter"] for t in incl])
f411 = np.array([v11[t]["f4"] for t in incl]); f412 = np.array([out[t]["f4"] for t in incl])
print(f"s_inter: mean {si11.mean():.3f}->{si12.mean():.3f} | frac>0.5 {np.mean(si11>0.5):.3f}->{np.mean(si12>0.5):.3f} "
      f"| frac>0.9 {np.mean(si11>0.9):.3f}->{np.mean(si12>0.9):.3f}")
print(f"F4:      mean {f411.mean():.3f}->{f412.mean():.3f} | frac>0.5 {np.mean(f411>0.5):.3f}->{np.mean(f412>0.5):.3f}")
# how many high-v11 s_inter dropped substantially (the artifact)
hi = si11 > 0.5
print(f"of {hi.sum()} high(v11) s_inter scenes: median s_inter {np.median(si11[hi]):.3f}->{np.median(si12[hi]):.3f}; "
      f"frac dropping below 0.5 now = {np.mean(si12[hi] < 0.5):.3f}")
