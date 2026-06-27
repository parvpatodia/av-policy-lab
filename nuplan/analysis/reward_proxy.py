"""RL capstone GATE-RL-1: an OPEN-LOOP proxy reward for an ego candidate trajectory,
computed from scene tensors (no nuPlan sim), reusing f4_score machinery. Reward =
progress along route - collision risk vs agent rollouts - off-route - discomfort.

Validation (this file's __main__): the EXPERT (ego_future) trajectory must score HIGH;
off-route / collision-aimed / jerky perturbations must score LOW. If the reward cannot
rank trajectories sensibly, RL cannot work -> gate fails.

Candidate traj: (H,2) meters, ego frame, H=16 @ 2 Hz (dt 0.5 s), ego starts at (0,0).
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "features"))
from features.f4_score import DenormConfig, denorm_sample, _agent_state, _agent_rollout  # noqa

DT_EGO = 0.5            # ego_future cadence (16 @ 2 Hz)
COLL_SIGMA = 1.0        # m; collision kernel width (only genuine near-overlap penalizes;
                        # safe passing at 2-3 m -> ~0. A 1.5 m gap -> 0.10, 1.0 m -> 0.37)
CORRIDOR_M = 5.0        # m; off-route normalizer
OFF_TOL = 4.0           # m; tolerance hinge. The route polyline is systematically ~3 m
                        # laterally offset from the ego path (diagnosed: a frame/centerline
                        # offset, not real off-road driving), so penalize only deviation
                        # BEYOND OFF_TOL -- the expert (~2.9 m) pays 0, gross departures pay.
W = dict(prog=1.0, coll=2.0, off=2.0, comf=0.3)


def _route_arclen(route):
    d = np.linalg.norm(np.diff(route, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def progress_and_offroute(traj, route):
    """progress = along-route fraction reached by the endpoint; offroute = mean
    point-to-POLYLINE (segment, not vertex) distance. WHY segment: route polylines
    are sparsely sampled, so point-to-vertex grossly overestimates the true lateral
    offset (the expert measured 3.1 m off via vertices, ~0.5 m via segments)."""
    s = _route_arclen(route)
    seg0, seg1 = route[:-1], route[1:]
    dseg = seg1 - seg0                                  # (R-1, 2)
    L2 = (dseg ** 2).sum(1) + 1e-9
    best = np.empty(len(traj)); best_s = np.empty(len(traj))
    for k in range(len(traj)):
        t = np.clip(((traj[k] - seg0) * dseg).sum(1) / L2, 0.0, 1.0)   # (R-1,)
        proj = seg0 + t[:, None] * dseg
        dist = np.linalg.norm(proj - traj[k], axis=1)
        j = int(dist.argmin())
        best[k] = dist[j]; best_s[k] = s[j] + t[j] * (s[j + 1] - s[j])
    offroute = float(best.mean())
    prog = float(best_s[-1] / (s[-1] + 1e-6))
    return prog, offroute


def collision_risk(traj, d, dn):
    """max over agents of a time-aligned proximity kernel; ~1 if a candidate point
    coincides with an agent at the same time, ->0 when far."""
    H = traj.shape[0]
    ego_t = np.arange(1, H + 1) * DT_EGO
    risk = 0.0
    for j in range(d["agents"].shape[0]):
        st = _agent_state(d["agents"][j], d["agent_mask"][j], dn)
        if st is None or np.linalg.norm(st["xy"]) < 1e-6:
            continue
        a_path, a_t = _agent_rollout(st, 5.0, 0.25)           # (T,2),(T,)
        a_full = np.concatenate([st["xy"][None], a_path], 0)
        a_tf = np.concatenate([[0.0], a_t])
        ax = np.interp(ego_t, a_tf, a_full[:, 0], right=a_full[-1, 0])
        ay = np.interp(ego_t, a_tf, a_full[:, 1], right=a_full[-1, 1])
        dist = np.hypot(traj[:, 0] - ax, traj[:, 1] - ay)
        risk = max(risk, float(np.exp(-((dist.min() / COLL_SIGMA) ** 2))))
    return risk


def comfort_pen(traj):
    """normalized jerk-ish: 2nd differences of position (accel proxy)."""
    if len(traj) < 3:
        return 0.0
    acc = np.diff(traj, n=2, axis=0) / (DT_EGO ** 2)
    return float(np.linalg.norm(acc, axis=1).mean())


def reward(traj, d, dn, w=W):
    route = d["route"][d["route_mask"]]
    if len(route) < 2:
        return None, None
    prog, off = progress_and_offroute(traj, route)
    coll = collision_risk(traj, d, dn)
    comf = comfort_pen(traj)
    off_pen = max(0.0, off - OFF_TOL)                   # tolerance hinge (absorb frame offset)
    R = w["prog"] * prog - w["coll"] * coll - w["off"] * (off_pen / CORRIDOR_M) - w["comf"] * (comf / 5.0)
    return float(R), {"prog": prog, "coll": round(coll, 3), "offroute_m": round(off, 2),
                      "off_pen_m": round(off_pen, 2), "comfort": round(comf, 2)}


# ---------------- GATE-RL-1 validation ----------------

def _expert_traj(sample):
    return np.asarray(sample["ego_future"], np.float64)[:, :2]   # (16,2) meters ego frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--n-scenes", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shards = sorted(glob.glob(a.shard_glob))
    rng = np.random.default_rng(0)
    rows = {"expert": [], "offroute": [], "collision": [], "jerky": [], "reversed": []}
    comp_expert = []
    got = 0
    for sp in shards:
        data = torch.load(sp, map_location="cpu", weights_only=False)
        dn = DenormConfig.from_shard_config(data["config"])
        for s in data["samples"]:
            d = denorm_sample(s, dn)
            if not d["route_mask"].any():
                continue
            ex = _expert_traj(s)
            Rex, cex = reward(ex, d, dn)
            if Rex is None:
                continue
            rows["expert"].append(Rex); comp_expert.append(cex)
            # perturbations
            # offroute: shift 8 m laterally to the side that INCREASES offroute (a genuine
            # off-route departure; a fixed-sign shift can move toward the route and is ambiguous)
            cand = [ex + np.array([0.0, 8.0]), ex + np.array([0.0, -8.0])]
            offR = min((reward(c, d, dn)[0] for c in cand))      # worse (lower-R) side
            rows["offroute"].append(offR)
            # collision: aim at the nearest agent's near-future position
            agentxy = None
            for j in range(d["agents"].shape[0]):
                st = _agent_state(d["agents"][j], d["agent_mask"][j], dn)
                if st is not None and np.linalg.norm(st["xy"]) > 1e-6:
                    agentxy = st["xy"]; break
            if agentxy is not None:
                t = np.linspace(0, 1, len(ex))[:, None]
                coll = (1 - t) * np.zeros(2) + t * agentxy            # straight line into the agent
                rows["collision"].append(reward(coll, d, dn)[0])
            jerky = ex + rng.normal(0, 1.5, ex.shape)            # noisy/jerky
            rows["jerky"].append(reward(jerky, d, dn)[0])
            rows["reversed"].append(reward(ex[::-1].copy(), d, dn)[0])  # reversed = wrong direction
            got += 1
            if got >= a.n_scenes:
                break
        if got >= a.n_scenes:
            break
    import json
    summ = {k: {"n": len(v), "mean_R": round(float(np.mean(v)), 3), "median_R": round(float(np.median(v)), 3)}
            for k, v in rows.items() if v}
    # expert vs each perturbation: fraction of scenes where expert wins (paired)
    ex = np.array(rows["expert"])
    wins = {}
    for k in ["offroute", "collision", "jerky", "reversed"]:
        v = np.array(rows[k])
        m = min(len(ex), len(v))
        wins[k] = round(float(np.mean(ex[:m] > v[:m])), 3)
    ce = {kk: round(float(np.mean([c[kk] for c in comp_expert])), 3) for kk in comp_expert[0]}
    res = {"n_scenes": got, "reward_summary": summ,
           "expert_beats_perturbation_frac": wins, "expert_components_mean": ce,
           "GATE_RL_1_PASS": bool(all(w > 0.8 for w in wins.values()))}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2)); print("wrote", a.out)


if __name__ == "__main__":
    main()
