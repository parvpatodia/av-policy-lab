"""RL capstone GATE-RL-2: reward-guided (AWR/GRPO) training of the multi-hypothesis
head, to test whether a per-scene reward makes diversity SCENE-ADAPTIVE (unlike the
fixed repulsion of ADR-037, which over-diversified uniformly).

Per scene, per mode: perturb the mode E times (exploration), reward each perturbation
with the validated open-loop proxy (reward_proxy), form a GROUP-relative advantage
(GRPO: baseline = per-scene mean over all modes x perturbations), and AWR-regress each
mode toward its advantage-weighted perturbations. A light GT anchor (best mode -> expert)
keeps modes realistic; score CE ranks modes by reward. Diversity then emerges only where
multiple high-reward options exist (junctions) -> scene-adaptive by construction.

Bounded de-risk: small B/E/steps. Saves a ckpt to probe with wta_probe (scene-adaptive
check = dispersion higher at decision types than at stationary).
"""
import argparse, glob, time, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "features"))
sys.path.insert(0, str(REPO / "analysis"))   # for reward_proxy
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import WTAHead, HeadConfig
from models.f0_dataset import scale_future, unscale_future
from training.train_policy import ENCODER_KEYS
from features.f4_score import DenormConfig, denorm_sample
import reward_proxy as rp


def load(shard_glob, n):
    shards = sorted(glob.glob(shard_glob))
    enc = {k: [] for k in ENCODER_KEYS}; fut = []; dicts = []; dn = None
    got = 0
    for sp in shards:
        data = torch.load(sp, map_location="cpu", weights_only=False)
        if dn is None: dn = DenormConfig.from_shard_config(data["config"])
        for s in data["samples"]:
            d = denorm_sample(s, dn)
            if not d["route_mask"].any():
                continue
            for k in ENCODER_KEYS: enc[k].append(s[k])
            fut.append(s["ego_future"].float()); dicts.append(d)
            got += 1
            if got >= n: break
        if got >= n: break
    return {k: torch.stack(v) for k, v in enc.items()}, torch.stack(fut), dicts, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--n-scenes", type=int, default=4000)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--explore", type=int, default=4)       # perturbations per mode
    ap.add_argument("--sigma", type=float, default=0.15)    # exploration noise (scaled units)
    ap.add_argument("--temp", type=float, default=0.5)      # AWR temperature
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor-w", type=float, default=0.5)  # GT realism anchor weight
    ap.add_argument("--init-ckpt", default=None)            # warm-start (e.g. plain WTA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, "loading...", flush=True)
    batch_all, fut_all, dicts, dn = load(a.shard_glob, a.n_scenes)
    fut_s = scale_future(fut_all); N = fut_s.shape[0]
    print("loaded", N, flush=True)

    enc = SceneEncoder(SceneEncoderConfig()).to(dev).train()
    head = WTAHead(HeadConfig(), n_modes=a.n_modes).to(dev).train()
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"]); print("warm-start", a.init_ckpt, flush=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=1e-4)

    M, H, E = a.n_modes, HeadConfig().horizon, a.explore
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, N, (a.batch,))
        sub = {k: v[idx].to(dev) for k, v in batch_all.items()}
        memory = enc(sub)
        trajs, scores = head(memory, goal=None)            # (B,M,H,3),(B,M) scaled
        xy = trajs[..., :2]                                 # (B,M,H,2)
        # exploration: E perturbations per mode
        noise = torch.randn(a.batch, M, E, H, 2, device=dev) * a.sigma
        cand = xy.unsqueeze(2) + noise                      # (B,M,E,H,2) scaled
        cand_m = (cand * 10.0).detach().cpu().numpy()       # meters (FUTURE_SCALE=10)
        # reward each perturbation (CPU, validated proxy)
        R = np.empty((a.batch, M, E), np.float32)
        for b in range(a.batch):
            d = dicts[idx[b].item()]
            for m in range(M):
                for e in range(E):
                    r, _ = rp.reward(cand_m[b, m, e], d, dn)
                    R[b, m, e] = r if r is not None else -5.0
        Rt = torch.tensor(R, device=dev)                    # (B,M,E)
        base = Rt.reshape(a.batch, -1).mean(1, keepdim=True).unsqueeze(-1)   # per-scene GRPO baseline
        adv = Rt - base                                     # (B,M,E)
        wts = torch.softmax(adv / a.temp, dim=2).unsqueeze(-1).unsqueeze(-1) # (B,M,E,1,1)
        target = (wts * cand).sum(2).detach()               # (B,M,H,2) AWR target
        loss_awr = ((xy - target) ** 2).mean()
        # GT realism anchor: best mode -> expert
        gt = fut_s[idx].to(dev)[:, :, :2]                   # (B,H,2)
        err = ((xy - gt.unsqueeze(1)) ** 2).mean((2, 3))    # (B,M)
        bestm = err.argmin(1)
        anchor = err.gather(1, bestm.unsqueeze(1)).mean()
        # score CE toward best-reward mode
        ce = F.cross_entropy(scores, Rt.mean(2).argmax(1))
        loss = loss_awr + a.anchor_w * anchor + 0.1 * ce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {loss.item():.4f} awr {loss_awr.item():.4f} anchor {anchor.item():.4f} "
                  f"meanR {R.mean():.3f} maxR {R.max():.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "n_modes": M, "steps": a.steps, "n_scenes": N, "rl": True}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
