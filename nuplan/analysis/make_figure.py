"""WS-F headline figure: the 4-panel diagnostic chain, from committed artifacts.
(A) the diffusion policy collapses over training; (B) but the head CAN be bimodal
on bimodal supervision (positive control); (C) the supervised WTA fix fans, it does
not split; (D) every nuPlan CLS outcome is saturated. CPU, Agg backend."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = Path("/scratch/patodia.pa/av-policy-lab")

# ---- A: collapse curve (epoch dispersion, % of 35m path) ----
ep_order = ["010", "050", "090", "130", "150"]
eps, disp_pct = [], []
for e in ep_order:
    p = SC / f"mm_diff_route_seed0_e{e}.json"
    if p.exists():
        r = json.load(open(p)); eps.append(int(e)); disp_pct.append(100 * r["dispersion_over_displacement_median"])

# ---- B: synthetic-bimodal control ----
sb = json.load(open(SC / "synth_bimodal_result.json"))
bvals = [sb["frac_samples_near_modeA_median"], sb["frac_samples_near_midpoint_median"], sb["frac_samples_near_modeB_median"]]

# ---- C: WTA fan vs lane width ----
w05 = json.load(open(SC / "wta_derisk_probe6k.json"))
w01 = json.load(open(SC / "wta_derisk_eps01_probe6k.json"))
wta_disp = [0.131, w05["endpoint_dispersion_m"]["median"], w01["endpoint_dispersion_m"]["median"]]
wta_lbl = ["diffusion\n(collapsed)", "WTA eps=.05", "WTA eps=.01"]
wta_modes = [0.05, 100 * w05["modes"]["frac_ge2"], 100 * w01["modes"]["frac_ge2"]]

# ---- D: ceiling fractions (pooled over 4 cells, r0) ----
import pandas as pd
cols = ["score", "ego_progress_along_expert_route", "time_to_collision_within_bound",
        "no_ego_at_fault_collisions", "drivable_area_compliance", "ego_is_comfortable"]
short = ["CLS", "progress", "TTC", "no-collision", "drivable", "comfort"]
ceil = []
for c in cols:
    vals = []
    for cell in ["det_route", "diff_route", "det_precise", "diff_precise"]:
        df = pd.read_parquet(SC / f"merged_r0_full/{cell}/eval/aggregator_metric/merged.parquet")
        df = df[df.scenario != "final_score"]; vals.append(df[c])
    s = pd.concat(vals).dropna(); ceil.append(float((s >= 0.99).mean()))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Why the diffusion-vs-deterministic interaction-criticality benefit is unrealized & untestable on nuPlan CLS",
             fontsize=12, fontweight="bold")

a = ax[0, 0]
a.plot(eps, disp_pct, "o-", color="#c0392b", lw=2)
a.axhline(0.37, ls=":", color="gray"); a.set_xlabel("training epoch"); a.set_ylabel("sample dispersion (% of 35 m path)")
a.set_title("(A) Diffusion policy collapses over training\n(K=32 samples -> point estimator)")
a.set_ylim(0, max(disp_pct) * 1.2)

b = ax[0, 1]
b.bar(["left arc", "midpoint\n(collapse)", "right arc"], bvals, color=["#2980b9", "#bbb", "#2980b9"])
b.set_ylabel("fraction of samples"); b.set_ylim(0, 0.6)
b.set_title("(B) Positive control: the SAME head IS bimodal\non bimodal supervision (50/50, 0 at midpoint)")

c = ax[1, 0]
bars = c.bar(wta_lbl, wta_disp, color=["#c0392b", "#27ae60", "#27ae60"])
c.axhline(3.5, ls="--", color="k"); c.text(0.05, 3.6, "lane width 3.5 m", fontsize=9)
c.set_ylabel("endpoint dispersion (m)")
c.set_title("(C) Supervised WTA fix fans, it does not split")
for i, (bar, m) in enumerate(zip(bars, wta_modes)):
    c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{m:.1f}%\n>=2 modes", ha="center", fontsize=8)

d = ax[1, 1]
d.barh(short, [100*x for x in ceil], color="#8e44ad")
d.axvline(100, ls=":", color="gray"); d.set_xlabel("% of scenarios at ceiling (>=0.99)"); d.set_xlim(0, 105)
d.set_title("(D) Every nuPlan CLS outcome is saturated\n(no headroom to express a moderated gap)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = SC / "diagnostic_figure.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
print(f"A epochs={eps} disp%={[round(x,2) for x in disp_pct]}")
print(f"B bimodal split={[round(x,2) for x in bvals]}")
print(f"C wta_disp={[round(x,2) for x in wta_disp]} modes%={[round(x,1) for x in wta_modes]}")
print(f"D ceiling%={[round(100*x,0) for x in ceil]}")
