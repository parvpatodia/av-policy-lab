"""Experiment 1 (rigor-upgrade, gates any Tier-3 retrain): is there MULTIMODAL
SUPERVISION in the real data? ADR-029 = policy collapsed; ADR-030 = the arch CAN be
multimodal, so the collapse is a data property. This asks whether the data even
contains multimodality to capture, and whether it concentrates at interaction-
critical scenes (the data-level version of the original H1 premise).

Method (CPU, no training):
- Load N scenes from f0_v3 (ego_future + scenario_token + scenario_type + encoder inputs).
- Context embedding WITHOUT future leakage: a RANDOMLY-INITIALIZED SceneEncoder (untrained),
  mean-pooled. WHY untrained: a trained encoder is optimized to predict the future, so its
  neighbors share futures tautologically; a random fixed nonlinear projection captures input
  geometry only (no future leakage).
- For each scene: k nearest neighbors (cosine) in context space. Dispersion of the k neighbors'
  logged-future ENDPOINTS (ego frame, m) + mode count (union-find eps=3.5m) = "available" spread.
- Compare available vs captured (policy per-scene sample dispersion ~0.13m, ADR-029) vs marginal
  (random-pair future dispersion). Stratify available by interaction-criticality (s_inter; joined
  by scenario_token) and by scenario_type.

available >> captured + meaningful frac>=2 modes -> real multimodality discarded -> Tier 3 viable.
available ~= captured -> data locally unimodal -> Tier 3 futile -> honest three-finding end-state.
NOTE: kNN-neighbor dispersion is an UPPER bound on the true conditional spread (residual context
differences inflate it), so it is a conservative test for ABSENCE of multimodality.
"""
import argparse, glob, json
import numpy as np, torch
from models.scene_encoder import SceneEncoder, SceneEncoderConfig

ENCODER_KEYS = ("ego","agents","agent_mask","map_polylines","map_mask",
                "crosswalks","crosswalk_mask","route_polyline","route_mask","traffic_lights")

def load_scenes(shard_glob, n):
    shards = sorted(glob.glob(shard_glob))
    enc, fut, tok, styp = {k: [] for k in ENCODER_KEYS}, [], [], []
    got = 0
    for sp in shards:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for s in d["samples"]:
            if "ego_future" not in s:  # encoder-only shard, skip
                break
            for k in ENCODER_KEYS: enc[k].append(s[k])
            fut.append(s["ego_future"].float())
            tok.append(s.get("scenario_token", ""))
            styp.append(s.get("scenario_type", "?"))
            got += 1
            if got >= n: break
        if got >= n: break
    batch = {k: torch.stack(v) for k, v in enc.items()}
    return batch, torch.stack(fut), tok, styp

def embed(batch, dev, bs=256):
    torch.manual_seed(123)  # fixed random (untrained) encoder
    enc = SceneEncoder(SceneEncoderConfig()).to(dev).eval()
    outs = []
    with torch.no_grad():
        N = batch["ego"].shape[0]
        for i in range(0, N, bs):
            sub = {k: v[i:i+bs].to(dev) for k, v in batch.items()}
            mem = enc(sub)                      # (b, L, d)
            outs.append(mem.mean(dim=1).cpu())  # (b, d) mean-pool, context vector
    return torch.cat(outs, 0).numpy()

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
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--f4", default="/scratch/patodia.pa/av-policy-lab/features/f4/f4_scores_v11.json")
    ap.add_argument("--n-scenes", type=int, default=4000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--eps-m", type=float, default=3.5)
    ap.add_argument("--captured-m", type=float, default=0.131)  # ADR-029 policy sample dispersion
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, fut, tok, styp = load_scenes(a.shard_glob, a.n_scenes)
    N = fut.shape[0]
    end = fut[:, -1, :2].numpy()                      # (N,2) logged future endpoints (m, ego frame)
    emb = embed(batch, dev)                           # (N,d) untrained-encoder context vectors
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    f4 = json.load(open(a.f4))
    s_inter = np.array([f4.get(t, {}).get("s_inter", np.nan) for t in tok])

    # cosine kNN (exclude self) -> neighbor future dispersion + mode count
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1e9)
    knn = np.argpartition(-sim, a.k, axis=1)[:, :a.k]   # (N,k)
    disp = np.empty(N); modes = np.empty(N, dtype=int)
    for i in range(N):
        ep = end[knn[i]]                                # (k,2) neighbor endpoints
        D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
        disp[i] = D[np.triu_indices(a.k, 1)].mean()
        modes[i] = n_modes(ep, a.eps_m)

    # marginal: random-pair endpoint dispersion (upper bound)
    rng = np.random.default_rng(0)
    pa, pb = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
    marginal = float(np.linalg.norm(end[pa] - end[pb], axis=-1).mean())

    def stratum(mask):
        if mask.sum() < 20: return None
        return {"n": int(mask.sum()),
                "available_disp_median_m": float(np.median(disp[mask])),
                "available_disp_mean_m": float(disp[mask].mean()),
                "frac_ge2_modes": float((modes[mask] >= 2).mean()),
                "mean_modes": float(modes[mask].mean())}

    valid = ~np.isnan(s_inter)
    hi = valid & (s_inter >= 0.5)      # interaction-critical
    lo = valid & (s_inter < 0.5)
    res = {
        "device": str(dev), "n_scenes": N, "k": a.k, "eps_m": a.eps_m,
        "captured_policy_disp_m": a.captured_m,                 # ADR-029 (what the policy emits)
        "marginal_randompair_disp_m": marginal,                # upper bound
        "available_overall": stratum(np.ones(N, bool)),        # kNN neighbor future spread
        "available_high_sinter": stratum(hi),
        "available_low_sinter": stratum(lo),
        "n_with_sinter": int(valid.sum()),
    }
    # top scenario types by available dispersion
    types = {}
    for t in set(styp):
        m = np.array([x == t for x in styp])
        st = stratum(m)
        if st: types[t] = st
    res["by_scenario_type_top"] = dict(sorted(types.items(),
        key=lambda kv: -kv[1]["available_disp_median_m"])[:8])
    print(json.dumps(res, indent=2))
    if a.out: json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
