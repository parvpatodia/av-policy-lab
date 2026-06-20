"""Construct-validity diagnostic for F4's S_inter branch: does it over-fire on
same-direction / merging traffic, or fire on genuine crossing conflicts?

S_inter scores an agent by the PrET band-pass over the FIRST geometric crossing
of the ego nominal path and the agent's constant-turn-rate rollout. Two failure
modes would inflate F4 without real interaction ambiguity:

  1. same-direction traffic: a car going the same way whose path happens to
     intersect the ego corridor (lane merge, adjacent lane) - debatable whether
     this is a genuine yield-or-go fork or just following.
  2. curvature artifact: the agent heading-rate is a 2-point finite difference
     of a quantized history; a noisy hr bends the 5 s rollout across the ego
     path, manufacturing a crossing that does not exist for the straight agent.

For every high-F4 scenario this reruns the exact s_inter agent loop, but for
each TRIGGERING agent records: its band-pass contribution I_j, the PrET gap,
its heading relative to the ego (same/cross/oncoming), the crossing angle, the
heading-rate magnitude, and crucially whether the crossing SURVIVES re-rolling
the agent straight (hr=0). It then attributes each scenario's score to its top
contributor and reports the share of high-F4 mass that is same-direction or
curvature-only. That share is the over-fire rate.

Run: ./.venv/bin/python s_inter_diagnostic.py            # uses local f0_v2
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# f4_score sits beside this file locally, or in ../features in the HPC repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
try:
    import f4_score as F
except ImportError:
    from nuplan.features import f4_score as F  # packaged layout

# data root is configurable so the tool runs on the Mac (local f0_v2) or HPC.
# AV_F0_GLOB: glob to the scene shards; AV_F4_SCORES: the f4 scores json.
ROOT = Path(os.environ.get("AV_ASSETS", "/Users/parvpatodia/av_assets"))
SHARDS = sorted(glob.glob(os.environ.get(
    "AV_F0_GLOB", str(ROOT / "f0_v2" / "task_*" / "scene_shard_*.pt"))))
F4 = json.load(open(os.environ.get(
    "AV_F4_SCORES", str(ROOT / "f4_rating" / "f4_scores_v11.json"))))
SHARD_CFG = {"pos_scale_m": 120.0, "vel_scale_mps": 15.0}
I_MIN = 0.05                 # an agent must contribute at least this to count
MAX_SCENES = 250
HIGH = 2 / 3


def _ang(a: float, b: float) -> float:
    """Smallest absolute angle between two headings, degrees in [0,180]."""
    d = math.atan2(math.sin(a - b), math.cos(a - b))
    return abs(math.degrees(d))


def _first_crossing_dirs(pa, pb):
    """First crossing of two polylines -> (i,j seg dirs) or None. Mirrors
    f4_score._path_crossing_times but returns the local segment directions so
    we can measure the crossing angle."""
    for i in range(len(pa) - 1):
        for j in range(len(pb) - 1):
            uv = F._seg_intersect(pa[i], pa[i + 1], pb[j], pb[j + 1])
            if uv is not None:
                da = pa[i + 1] - pa[i]
                db = pb[j + 1] - pb[j]
                return (math.atan2(da[1], da[0]), math.atan2(db[1], db[0]))
    return None


def analyze_scene(sample: dict) -> dict | None:
    dn = F.DenormConfig.from_shard_config(SHARD_CFG)
    d = F.denorm_sample(sample, dn)
    nominal = F.ego_nominal_path(d)
    if nominal is None:
        return None
    ego_path, ego_t = nominal
    full_path = np.concatenate([np.zeros((1, 2)), ego_path], axis=0)
    e_tf = np.concatenate([[0.0], ego_t])
    ego_dir0 = math.atan2(full_path[1, 1] - full_path[0, 1],
                          full_path[1, 0] - full_path[0, 0])

    agents = []
    for j in range(d["agents"].shape[0]):
        st = F._agent_state(d["agents"][j], d["agent_mask"][j], dn)
        if st is None or np.linalg.norm(st["xy"]) < 1e-6:
            continue
        a_path, a_t = F._agent_rollout(st, F.AGENT_ROLLOUT_S, F.ROLLOUT_DT_S)
        a_full = np.concatenate([st["xy"][None], a_path], axis=0)
        a_tf = np.concatenate([[0.0], a_t])
        I_j, gap, cross_ang = 0.0, float("nan"), float("nan")
        cross = F._path_crossing_times(full_path, e_tf, a_full, a_tf)
        if cross is not None:
            gap = abs(cross[0] - cross[1])
            I_j = math.exp(-(((gap - F.PRET_CENTER_S) / F.PRET_WIDTH_S) ** 2))
            dirs = _first_crossing_dirs(full_path, a_full)
            if dirs is not None:
                cross_ang = _ang(dirs[0], dirs[1])
        # straight (hr=0) re-roll: does the crossing survive?
        st0 = dict(st); st0["hr"] = 0.0
        s_path, s_t = F._agent_rollout(st0, F.AGENT_ROLLOUT_S, F.ROLLOUT_DT_S)
        s_full = np.concatenate([st["xy"][None], s_path], axis=0)
        straight_cross = F._path_crossing_times(full_path, e_tf, s_full,
                                                np.concatenate([[0.0], s_t]))
        ped = bool(st["is_ped"] and st["v"] < F.PED_SPEED_MS
                   and F._ped_near_crossing_crosswalk(st["xy"], d, full_path))
        I_eff = max(I_j, F.PED_OVERRIDE) if ped else I_j
        if I_eff < I_MIN:
            continue
        rel = _ang(st["h"], ego_dir0)
        # validity check on the heading CHANNEL: for a moving agent the stored
        # heading must match its velocity direction, else s_inter itself (which
        # rolls out along st["h"]) is using a bad channel and this whole
        # diagnostic is moot. Recorded per fast agent, summarized in report.
        vx, vy = (d["agents"][j][np.flatnonzero(d["agent_mask"][j])[-1], 4],
                  d["agents"][j][np.flatnonzero(d["agent_mask"][j])[-1], 5])
        h_vel_err = (_ang(st["h"], math.atan2(vy, vx))
                     if (vx * vx + vy * vy) > (0.5 / dn.vel_scale_mps) ** 2
                     else float("nan"))
        kind = ("ped" if ped else
                "same" if rel < 45 else
                "oncoming" if rel > 135 else "cross")
        agents.append({
            "I": I_eff, "gap": gap, "rel_deg": rel, "cross_deg": cross_ang,
            "hr_dps": abs(math.degrees(st["hr"])), "v": st["v"], "kind": kind,
            "h_vel_err": h_vel_err,
            "curvature_only": bool(I_j >= I_MIN and not ped
                                   and straight_cross is None),
        })
    if not agents:
        return None
    agents.sort(key=lambda a: a["I"], reverse=True)
    top = agents[:F.TOP_K_AGENTS]
    # noisy-OR of the top-k reproduces s_inter
    si = 1.0 - math.prod(1.0 - a["I"] for a in top)
    return {"s_inter": si, "top": top, "lead": agents[0], "n_trig": len(agents)}


def main():
    high = {t for t, r in F4.items()
            if r.get("f4") is not None and not r.get("excluded")
            and r["f4"] > HIGH}
    print(f"high-band tokens (F4>{HIGH:.2f}): {len(high)}; scanning shards "
          f"for up to {MAX_SCENES}...")
    seen, recs = set(), []
    for n, sp in enumerate(SHARDS):
        if len(recs) >= MAX_SCENES:
            break
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for s in data["samples"]:
            tok = s.get("scenario_token")
            if tok in high and tok not in seen:
                seen.add(tok)
                r = analyze_scene(s)
                if r is not None:
                    r["token"] = tok
                    r["type"] = F4[tok].get("scenario_type", "?")
                    recs.append(r)
        if n % 40 == 0:
            print(f"  shard {n}/{len(SHARDS)}  collected {len(recs)}")
    report(recs)


def report(recs: list):
    n = len(recs)
    if n == 0:
        print("no high-F4 scenes found locally"); return
    lead_kind = Counter(r["lead"]["kind"] for r in recs)
    curv_lead = sum(r["lead"]["curvature_only"] for r in recs)
    # score-mass attribution: each scene's s_inter, split by its lead kind
    mass = sum(r["s_inter"] for r in recs)
    mass_by_kind = Counter()
    for r in recs:
        mass_by_kind[r["lead"]["kind"]] += r["s_inter"]
    curv_mass = sum(r["s_inter"] for r in recs if r["lead"]["curvature_only"])

    print(f"\n=== S_inter over-fire diagnostic ({n} high-F4 scenes) ===")
    print("lead-contributor kind (count and share of scenes):")
    for k in ("cross", "oncoming", "same", "ped"):
        c = lead_kind.get(k, 0)
        print(f"  {k:8s} {c:4d}  {c/n:5.1%}   score-mass {mass_by_kind.get(k,0)/mass:5.1%}")
    print(f"\ncurvature-only lead (crossing vanishes when agent re-rolled "
          f"straight): {curv_lead}/{n} = {curv_lead/n:.1%}  "
          f"(score-mass {curv_mass/mass:.1%})")

    leads = [r["lead"] for r in recs]
    print("\nlead-agent distributions (median / 90th pct):")
    for key, lab in (("rel_deg", "rel heading deg"), ("cross_deg", "crossing angle deg"),
                     ("hr_dps", "|heading rate| deg/s"), ("gap", "PrET gap s")):
        vals = np.array([l[key] for l in leads if not math.isnan(l[key])])
        if len(vals):
            print(f"  {lab:22s} median={np.median(vals):6.1f}  p90={np.percentile(vals,90):6.1f}")

    # validity gate on the diagnostic itself: heading channel vs velocity
    herr = np.array([l["h_vel_err"] for l in leads if not math.isnan(l["h_vel_err"])])
    if len(herr):
        print(f"\nheading-channel check (lead agents, |heading - velocity-dir|): "
              f"median={np.median(herr):.1f} deg  p90={np.percentile(herr,90):.1f} deg  "
              f"(small => s_inter rolls along the right direction; tool is valid)")

    # union, not sum: same-direction OR curvature-only (these overlap)
    suspect = sum(1 for r in recs
                  if r["lead"]["kind"] == "same" or r["lead"]["curvature_only"]) / n
    print(f"\nSUSPECT lead share (same-direction OR curvature-only) = {suspect:.1%}")
    if suspect < 0.20:
        print("VERDICT: S_inter is sound. High F4 is driven by genuine "
              "cross/oncoming conflicts; over-fire is not a material problem.")
    elif suspect < 0.40:
        print("VERDICT: minor over-fire. Document it; consider a crossing-angle "
              "gate as a robustness variant, not a blocker.")
    else:
        print("VERDICT: material over-fire. Recommend a pre-freeze S_inter fix "
              "(gate crossings by min angle and/or require straight-roll survival).")

    worst = sorted(recs, key=lambda r: (r["lead"]["kind"] in ("same",)) * r["s_inter"],
                   reverse=True)[:6]
    print("\nmost same-direction-driven high-F4 scenes (inspect against rating):")
    for r in worst:
        l = r["lead"]
        print(f"  {r['token'][:10]} F4={F4[r['token']]['f4']:.2f} s_inter={r['s_inter']:.2f} "
              f"lead={l['kind']} rel={l['rel_deg']:.0f} cross={l['cross_deg']:.0f} "
              f"hr={l['hr_dps']:.0f} type={r['type']}")


if __name__ == "__main__":
    main()
