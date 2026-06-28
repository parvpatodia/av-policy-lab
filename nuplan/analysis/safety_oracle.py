"""Best-of-modes SAFETY ORACLE (ADR-045 follow-up): does multimodality carry VALUE
that executed-CLS is structurally blind to? CLS scores the single executed trajectory;
this asks, per scene, whether ANY of the RL policy's K modes is SAFER (lower collision +
off-route, via the validated open-loop proxy) than the deterministic policy's single
trajectory -- and whether that safety advantage GROWS with interaction-criticality (F4).

If the advantage rises with F4 but executed-CLS did not (ADR-044), that PINPOINTS metric-
blindness as the barrier: multimodality provides a safe option at decision points that a
single-trajectory metric cannot credit. Open-loop, existing ckpts, no closed-loop sim.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "features")); sys.path.insert(0, str(REPO / "analysis"))
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import DeterministicHead, WTAHead, HeadConfig
from models.f0_dataset import unscale_future
from features.f4_score import DenormConfig, denorm_sample
import reward_proxy as rp
from analyze_moderation_v2 import wild_cluster_test

EK = ("ego","agents","agent_mask","map_polylines","map_mask","crosswalks","crosswalk_mask",
      "route_polyline","route_mask","traffic_lights")


def unsafety(traj, d, dn):
    """collision risk + off-route hinge (lower = safer); the safety part of the proxy."""
    coll = rp.collision_risk(traj, d, dn)
    route = d["route"][d["route_mask"]]
    if len(route) < 2:
        return rp.W["coll"] * coll
    _, off = rp.progress_and_offroute(traj, route)
    off_pen = max(0.0, off - rp.OFF_TOL)
    return rp.W["coll"] * coll + rp.W["off"] * (off_pen / rp.CORRIDOR_M)


def load_policy(ckpt, head_cls, dev, n_modes=None):
    enc = SceneEncoder(SceneEncoderConfig()).to(dev).eval()
    head = (head_cls(HeadConfig(), n_modes=n_modes) if n_modes else head_cls(HeadConfig())).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    ema = ck.get("ema")
    if ema:                                   # det #18 ckpts carry EMA; use it (deployment weights)
        sh = ema["shadow"]
        for mod, pre in ((enc, "encoder."), (head, "head.")):
            sd = {k[len(pre):]: v for k, v in sh.items() if k.startswith(pre)}
            mod.load_state_dict(sd, strict=False)
    return enc, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-ckpt", default="/scratch/patodia.pa/av-policy-lab/runs/f5_2x2_v4/det_route_seed0/epoch_120.pt")
    ap.add_argument("--rl-ckpt", default="/scratch/patodia.pa/av-policy-lab/runs/wta_derisk/rl_long_s6000.pt")
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--f4", default="/scratch/patodia.pa/av-policy-lab/features/f4/f4_scores_v11.json")
    ap.add_argument("--n-scenes", type=int, default=2000)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denc, dhead = load_policy(a.det_ckpt, DeterministicHead, dev)
    renc, rhead = load_policy(a.rl_ckpt, WTAHead, dev, n_modes=a.n_modes)

    shards = sorted(glob.glob(a.shard_glob)); enc_in = {k: [] for k in EK}; dicts = []; tok = []; dn = None
    got = 0
    for sp in shards:
        data = torch.load(sp, map_location="cpu", weights_only=False)
        if dn is None: dn = DenormConfig.from_shard_config(data["config"])
        for s in data["samples"]:
            d = denorm_sample(s, dn)
            if not d["route_mask"].any(): continue
            for k in EK: enc_in[k].append(s[k])
            dicts.append(d); tok.append(s.get("scenario_token", "")); got += 1
            if got >= a.n_scenes: break
        if got >= a.n_scenes: break
    batch = {k: torch.stack(v).to(dev) for k, v in enc_in.items()}
    with torch.no_grad():
        det = unscale_future(dhead(denc(batch), goal=None)).cpu().numpy()        # (N,H,3)
        modes = unscale_future(rhead(renc(batch), goal=None)[0]).cpu().numpy()   # (N,K,H,3)
    N, K = det.shape[0], modes.shape[1]
    H = det.shape[1]
    ramp = (np.arange(1, H + 1) / H)[:, None]                # linear growth to the endpoint
    rng = np.random.default_rng(0)
    det_un = np.empty(N); best_un = np.empty(N); det_bestK = np.empty(N)
    for i in range(N):
        det_un[i] = unsafety(det[i, :, :2], dicts[i], dn)
        best_un[i] = min(unsafety(modes[i, m, :, :2], dicts[i], dn) for m in range(K))
        # FAIR CONTROL: K matched-dispersion random perturbations of the det trajectory.
        # noise endpoint-std set to the RL modes' per-scene endpoint std -> det gets K equally-
        # spread tries, netting out the best-of-K + spread-magnitude effects. If RL learned modes
        # still beat this, their safety value is REAL (placement), not a selection artifact.
        ep = modes[i, :, -1, :2]                             # (K,2) RL mode endpoints
        sig = float(ep.std(0).mean()) + 1e-6
        pert = det[i, :, :2][None] + rng.normal(0, 1, (K, H, 2)) * ramp[None] * sig   # (K,H,2)
        det_bestK[i] = min(unsafety(pert[m], dicts[i], dn) for m in range(K))
    adv = det_un - best_un                                   # >0: a mode is SAFER than det (1 traj)
    fair_adv = det_bestK - best_un                           # >0: RL modes safer than matched random

    f4all = json.load(open(a.f4)); f4 = {t: r["f4"] for t, r in f4all.items() if r.get("f4") is not None}
    styp = {t: f4all[t].get("scenario_type", "?") for t in f4all}
    keep = [i for i, t in enumerate(tok) if t in f4]
    adv_k = adv[keep]; fair_k = fair_adv[keep]
    x = np.array([f4[tok[i]] for i in keep]); g = np.array([styp.get(tok[i], "?") for i in keep])
    wc = wild_cluster_test(x, adv_k, g, B=4999)
    wc_fair = wild_cluster_test(x, fair_k, g, B=4999)
    hi = x >= 0.5
    res = {"n": int(len(keep)), "det_ckpt": a.det_ckpt, "rl_ckpt": a.rl_ckpt,
           "mean_det_unsafety": float(det_un.mean()), "mean_best_mode_unsafety": float(best_un.mean()),
           "mean_det_bestK_unsafety": float(det_bestK[keep].mean()),
           "UNFAIR_mean_safety_advantage": float(adv_k.mean()),
           "UNFAIR_frac_scenes_a_mode_safer_than_det": float((adv_k > 1e-6).mean()),
           "FAIR_mean_advantage_vs_matched_random": float(fair_k.mean()),
           "FAIR_frac_RL_modes_beat_matched_random": float((fair_k > 1e-6).mean()),
           "FAIR_adv_high_sinter": float(fair_k[hi].mean()) if hi.sum() else None,
           "FAIR_adv_low_sinter": float(fair_k[~hi].mean()) if (~hi).sum() else None,
           "moderation_UNFAIR_adv_vs_F4": wc,
           "moderation_FAIR_adv_vs_F4": wc_fair}
    print(json.dumps(res, indent=2))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=2)); print("wrote", a.out)


if __name__ == "__main__":
    main()
