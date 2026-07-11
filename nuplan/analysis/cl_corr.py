"""GATE-CL-1 (closed-loop-reward RL, option 1): does the OPEN-LOOP proxy reward predict
the actual CLOSED-LOOP CLS? Decides the path: high corr -> a CLS-faithful proxy can train
a closed-loop-good policy; low corr -> the open-loop!=closed-loop gap (ADR-045) is
fundamental and sim-in-the-loop reward is required.

Per rl_h1 token (where we have real closed-loop CLS): forward the RL policy, take the
TOP-SCORED mode (the deployed selection that produced the CLS), compute the open-loop
proxy reward on it, and correlate with the token's closed-loop CLS.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "features")); sys.path.insert(0, str(REPO / "analysis"))
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import WTAHead, HeadConfig
from models.f0_dataset import unscale_future
from features.f4_score import DenormConfig, denorm_sample
import reward_proxy as rp
from analyze_moderation_v2 import read_cell_metric, latest_aggregator

EK = ("ego","agents","agent_mask","map_polylines","map_mask","crosswalks","crosswalk_mask",
      "route_polyline","route_mask","traffic_lights")


def open_loop_reward(traj, d, dn):
    r, _ = rp.reward(traj, d, dn)          # progress - collision - offroute(hinge) - comfort
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-ckpt", default="/scratch/patodia.pa/av-policy-lab/runs/wta_derisk/rl_long_s6000.pt")
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--min-tokens", type=int, default=300, help="early-stop once this many matched (corr needs few)")
    ap.add_argument("--cls-glob", default="rl_h1*",
                    help="glob (under sim_results/) of the closed-loop eval dirs that supply the "
                         "per-token deployed CLS. Default rl_h1* (mini). Boston de-risk: "
                         "eval/boston_zoo_r1/wta_route_shard* (default-WTA = deployed top-scored mode).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # closed-loop CLS per token (union of all matching eval shards)
    SC = "/scratch/patodia.pa/av-policy-lab"
    cls = {}
    for d_ in sorted(glob.glob(f"{SC}/sim_results/{a.cls_glob}")):
        agg = latest_aggregator(d_)
        if agg: cls.update(read_cell_metric(agg, "score")[0])
    print("closed-loop CLS tokens:", len(cls), flush=True)

    enc = SceneEncoder(SceneEncoderConfig()).to(dev).eval()
    head = WTAHead(HeadConfig(), n_modes=a.n_modes).to(dev).eval()
    ck = torch.load(a.rl_ckpt, map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])

    # scan f0_v3 for the scenes whose token has a CLS; forward policy; open-loop reward on top mode
    want = set(cls); rew = {}; dn = None; seen = 0
    for sp in sorted(glob.glob(a.shard_glob)):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        if dn is None: dn = DenormConfig.from_shard_config(data["config"])
        batchrows = []; dicts = []; toks = []
        for s in data["samples"]:
            t = s.get("scenario_token", "")
            if t not in want or t in rew: continue
            d = denorm_sample(s, dn)
            if not d["route_mask"].any(): continue
            batchrows.append(s); dicts.append(d); toks.append(t)
        if not batchrows: continue
        batch = {k: torch.stack([r[k] for r in batchrows]).to(dev) for k in EK}
        with torch.no_grad():
            trajs, scores = head(enc(batch), goal=None)
            trajs = unscale_future(trajs)            # (B,M,H,3) m
            top = scores.argmax(1).cpu().numpy()
        trajs = trajs.cpu().numpy()
        for i, t in enumerate(toks):
            r = open_loop_reward(trajs[i, top[i], :, :2], dicts[i], dn)
            if r is not None: rew[t] = r
        seen += len(toks)
        print(f"  matched {len(rew)}/{len(want)} (scanned shard, +{len(toks)})", flush=True)
        if len(rew) >= a.min_tokens: break       # corr needs only a few hundred

    toks = sorted(set(rew) & set(cls))
    R = np.array([rew[t] for t in toks]); C = np.array([cls[t] for t in toks])
    def pear(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float((x*y).sum() / (np.sqrt((x**2).sum()*(y**2).sum()) + 1e-12))
    def spear(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        return pear(rx, ry)
    res = {"n": len(toks), "pearson_proxy_vs_CLS": round(pear(R, C), 3),
           "spearman_proxy_vs_CLS": round(spear(R, C), 3),
           "proxy_mean": round(float(R.mean()), 3), "cls_mean": round(float(C.mean()), 3),
           "interpretation": "high (|r|>~0.5) -> open-loop proxy predicts closed-loop CLS, a better "
                             "proxy is viable; low -> open-loop!=closed-loop gap fundamental, "
                             "sim-in-the-loop reward required"}
    print(json.dumps(res, indent=2))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=2)); print("wrote", a.out)


if __name__ == "__main__":
    main()
