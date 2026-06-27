"""Tier-3 Step-1 de-risk trainer (bounded, standalone): train SceneEncoder + WTAHead
on a fixed in-memory subset of real f0_v3 scenes for a fixed step budget, then save a
checkpoint to probe. WHY standalone (not train_policy): the full pipeline does a 44k-
sample eval every epoch over ~400k scenes (~2h/epoch) -- overkill for a viability gate.
The de-risk question is only "do WTA modes specialize on real data?", visible after a
few thousand steps. Reuses the SAME wta_loss + encoder as the real training.
"""
import argparse, glob, time
import torch
from models.scene_encoder import SceneEncoder, SceneEncoderConfig
from models.policy_heads import WTAHead, HeadConfig
from models.f0_dataset import scale_future
from training.train_policy import wta_loss, ENCODER_KEYS

def load_scenes(shard_glob, n):
    shards = sorted(glob.glob(shard_glob))
    enc, fut = {k: [] for k in ENCODER_KEYS}, []
    got = 0
    for sp in shards:
        d = torch.load(sp, map_location="cpu", weights_only=False)
        for s in d["samples"]:
            for k in ENCODER_KEYS: enc[k].append(s[k])
            fut.append(s["ego_future"].float())
            got += 1
            if got >= n: break
        if got >= n: break
    return {k: torch.stack(v) for k, v in enc.items()}, torch.stack(fut)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="/scratch/patodia.pa/av-policy-lab/features/f0_v3/task_*/scene_shard_*.pt")
    ap.add_argument("--n-scenes", type=int, default=16000)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wta-eps", type=float, default=0.05,
                    help="relaxed-WTA: winner weight 1-eps, rest share eps (smaller=sharper modes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, "loading", a.n_scenes, "scenes...", flush=True)
    batch_all, fut_all = load_scenes(a.shard_glob, a.n_scenes)
    fut_all = scale_future(fut_all)                       # train in scaled space
    N = fut_all.shape[0]
    print("loaded", N, "scenes", flush=True)

    enc = SceneEncoder(SceneEncoderConfig()).to(dev).train()
    head = WTAHead(HeadConfig(), n_modes=a.n_modes).to(dev).train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=1e-4)

    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, N, (a.batch,))
        sub = {k: v[idx].to(dev) for k, v in batch_all.items()}
        fut = fut_all[idx].to(dev)
        memory = enc(sub)
        loss = wta_loss(head, memory, None, fut, eps=a.wta_eps)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step}: wta_loss {loss.item():.5f}  ({(time.time()-t0):.0f}s)", flush=True)

    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "n_modes": a.n_modes, "steps": a.steps, "n_scenes": N}, a.out)
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
