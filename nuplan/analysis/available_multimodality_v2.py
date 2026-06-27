"""WS-D: a DEFENSIBLE available-multimodality estimate, replacing the degenerate
random-untrained-encoder kNN of available_multimodality.py (audit C1: that embedding
maps all scenes to ~0.97 cosine, so its neighbors are ~random -> the 3.14 m / 46%
figure is inflated by a meaningless neighborhood).

Here context similarity uses INTERPRETABLE, future-independent scene descriptors --
ego speed (v0), agent count (n_par), stop context (g_stop), branch presence (b_r) --
and neighbors are restricted to the SAME scenario_type. No learned/random embedding,
no degeneracy, fully reviewable. We measure the logged-future endpoint dispersion of
the matched neighbors = an estimate of the conditional future spread ("available
multimodality"). Still an UPPER bound (scalar matches are coarse), reported honestly,
but far tighter than a degenerate embedding.

Triangulation reported: captured (policy 0.13 m) vs interpretable-match vs the old
random-encoder figure vs marginal (random pairs). Verdict per ADR-031/032: is the
available multimodality real per-scene ambiguity, or mostly residual context variation?
"""
import argparse, glob, json
import numpy as np, torch

EK = ("ego", "agents", "agent_mask", "map_polylines", "map_mask",
      "crosswalks", "crosswalk_mask", "route_polyline", "route_mask", "traffic_lights")


def load_futures(shard_glob, n):
    shards = sorted(glob.glob(shard_glob))
    fut, tok = [], []
    got = 0
    for sp in shards:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for s in d["samples"]:
            fut.append(s["ego_future"].float()[-1, :2].numpy())   # endpoint (m, ego frame)
            tok.append(s.get("scenario_token", ""))
            got += 1
            if got >= n:
                break
        if got >= n:
            break
    return np.array(fut), tok


def n_modes(ep, eps):
    K = len(ep); par = list(range(K))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
    for i in range(K):
        for j in range(i + 1, K):
            if D[i, j] < eps: par[f(i)] = f(j)
    return len({f(i) for i in range(K)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--f4", default="/scratch/patodia.pa/av-policy-lab/features/f4/f4_scores_v11.json")
    ap.add_argument("--n-scenes", type=int, default=6000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--eps-m", type=float, default=3.5)
    ap.add_argument("--captured-m", type=float, default=0.131)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    end, tok = load_futures(a.shard_glob, a.n_scenes)
    f4 = json.load(open(a.f4))
    # interpretable, future-independent context features + scenario_type
    feats, styp, keep = [], [], []
    for i, t in enumerate(tok):
        r = f4.get(t)
        if r is None or r.get("excluded"):
            continue
        feats.append([r.get("v0", 0.0), r.get("n_par", 0), r.get("g_stop", 0.0), r.get("b_r", 0)])
        styp.append(r.get("scenario_type", "?"))
        keep.append(i)
    keep = np.array(keep); end = end[keep]
    F = np.array(feats, float); styp = np.array(styp)
    N = len(F)
    # standardize features
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)

    disp, modes = [], []
    matched_pair = []   # k=1 tightest same-type match: future-endpoint distance
    for i in range(N):
        same = np.where((styp == styp[i]))[0]
        same = same[same != i]
        if len(same) < a.k:
            continue
        d2 = np.linalg.norm(F[same] - F[i], axis=1)
        nn = same[np.argsort(d2)[:a.k]]
        ep = end[nn]
        D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
        disp.append(D[np.triu_indices(a.k, 1)].mean())
        modes.append(n_modes(ep, a.eps_m))
        matched_pair.append(np.linalg.norm(end[nn[0]] - end[i]))   # tightest neighbor
    disp = np.array(disp); modes = np.array(modes); mp = np.array(matched_pair)

    rng = np.random.default_rng(0)
    pa, pb = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
    marginal = float(np.linalg.norm(end[pa] - end[pb], axis=-1).mean())

    res = {
        "n_scenes": int(N), "k": a.k, "eps_m": a.eps_m,
        "captured_policy_disp_m": a.captured_m,
        "marginal_randompair_disp_m": round(marginal, 3),
        "interpretable_match_available": {
            "n_evaluated": int(len(disp)),
            "disp_median_m": round(float(np.median(disp)), 3),
            "disp_mean_m": round(float(np.mean(disp)), 3),
            "frac_ge2_modes": round(float(np.mean(modes >= 2)), 3),
            "mean_modes": round(float(np.mean(modes)), 3),
        },
        "tightest_match_pair_future_dist_m": {
            "median": round(float(np.median(mp)), 3), "mean": round(float(np.mean(mp)), 3),
            "p90": round(float(np.percentile(mp, 90)), 3),
        },
        "random_encoder_old_disp_m": 3.14,   # available_multimodality.py (degenerate), for contrast
        "note": "interpretable same-type match; UPPER bound (coarse scalar match) but no embedding degeneracy",
    }
    print(json.dumps(res, indent=2))
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(res, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
