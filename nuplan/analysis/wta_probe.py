"""Tier-3 Step-1 de-risk probe: does the WINNER-TAKE-ALL head produce genuinely
multimodal trajectories on REAL data? Directly comparable to ADR-029 (the collapsed
diffusion policy: 0.13m sample dispersion, 0.05% scenes >=2 modes).

Loads a WTA ckpt, forwards the encoder + WTAHead on N real f0_v3 scenes (route, no
goal), measures the M hypotheses' endpoint dispersion + mode count (union-find,
eps=lane-width 3.5m), and the spread of the TOP-SCORED-vs-others (closed-loop
selection diversity). Stratifies by interaction-criticality (s_inter).

PASS (Tier 3 viable): WTA dispersion >> 0.13m, meaningful frac>=2 modes, modes
on-road; higher at interaction-critical scenes. FAIL: WTA also collapses.
"""
import argparse, glob, json
import numpy as np, torch
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import WTAHead, HeadConfig
from models.f0_dataset import unscale_future

ENCODER_KEYS = ("ego","agents","agent_mask","map_polylines","map_mask",
                "crosswalks","crosswalk_mask","route_polyline","route_mask","traffic_lights")

def load_scenes(shard_glob, n):
    shards = sorted(glob.glob(shard_glob))
    enc, tok = {k: [] for k in ENCODER_KEYS}, []
    got = 0
    for sp in shards:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for s in d["samples"]:
            for k in ENCODER_KEYS: enc[k].append(s[k])
            tok.append(s.get("scenario_token", ""))
            got += 1
            if got >= n: break
        if got >= n: break
    return {k: torch.stack(v) for k, v in enc.items()}, tok

def n_modes(ep, eps):
    K = len(ep); par = list(range(K))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
    for i in range(K):
        for j in range(i+1, K):
            if D[i, j] < eps: par[f(i)] = f(j)
    return len({f(i) for i in range(K)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--f4", default="/scratch/patodia.pa/av-policy-lab/features/f4/f4_scores_v11.json")
    ap.add_argument("--n-scenes", type=int, default=2000)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--eps-m", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ema", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = SceneEncoder(SceneEncoderConfig()).to(dev).eval()
    head = WTAHead(HeadConfig(), n_modes=a.n_modes).to(dev).eval()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    used_ema = False; ema = ck.get("ema")
    if ema and not a.no_ema:
        sh = ema["shadow"]
        for mod, pre in ((enc, "encoder."), (head, "head.")):
            sd = {k[len(pre):]: v for k, v in sh.items() if k.startswith(pre)}
            mod.load_state_dict(sd, strict=False); used_ema = True

    batch, tok = load_scenes(a.shard_glob, a.n_scenes)
    batch = {k: v.to(dev) for k, v in batch.items()}
    with torch.no_grad():
        trajs, scores = head(enc(batch), goal=None)     # (N,M,H,3),(N,M)
        trajs = unscale_future(trajs)
    end = trajs[..., -1, :2].float().cpu().numpy()       # (N,M,2)
    top = scores.argmax(1).cpu().numpy()                 # selected mode per scene
    N, M = end.shape[0], end.shape[1]
    disp = np.empty(N); modes = np.empty(N, int); topgap = np.empty(N)
    for i in range(N):
        ep = end[i]; D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
        disp[i] = D[np.triu_indices(M, 1)].mean()
        modes[i] = n_modes(ep, a.eps_m)
        cen = ep.mean(0); topgap[i] = np.linalg.norm(ep[top[i]] - cen)  # selected vs centroid
    f4 = json.load(open(a.f4))
    s_inter = np.array([f4.get(t, {}).get("s_inter", np.nan) for t in tok])
    def strat(mask):
        if mask.sum() < 20: return None
        return {"n": int(mask.sum()), "disp_median_m": float(np.median(disp[mask])),
                "frac_ge2_modes": float((modes[mask] >= 2).mean()),
                "mean_modes": float(modes[mask].mean())}
    valid = ~np.isnan(s_inter)
    res = {"ckpt": a.ckpt, "used_ema": used_ema, "n_scenes": N, "M": M, "eps_m": a.eps_m,
           "endpoint_dispersion_m": {"median": float(np.median(disp)), "mean": float(disp.mean()),
                "p90": float(np.percentile(disp, 90)), "max": float(disp.max())},
           "modes": {"frac_ge2": float((modes >= 2).mean()), "frac_ge3": float((modes >= 3).mean()),
                "mean": float(modes.mean()), "max": int(modes.max())},
           "selected_mode_offset_from_centroid_median_m": float(np.median(topgap)),
           "compare_collapsed_diffusion_m": 0.131,
           "high_sinter": strat(valid & (s_inter >= 0.5)),
           "low_sinter": strat(valid & (s_inter < 0.5))}
    print(json.dumps(res, indent=2))
    if a.out: json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
