"""Experiment 0 (rigor-upgrade, ADR-029 follow-up): can the DiffusionHead represent
a MULTIMODAL conditional at all? Decisive, self-contained, no encoder/shards.

Setup: C toy contexts (random fixed `memory` tensors, bypassing the SceneEncoder).
Each context maps 50/50 to TWO well-separated arc futures (left vs right, endpoints
SPREAD_M apart >> lane width). Train the SAME DiffusionHead with the SAME x0-MSE
objective + CosineSchedule as train_policy.compute_loss. Then DDIM-sample K per
context and measure whether the samples recover BOTH modes or collapse to the mean.

Verdict:
- recover ~2 modes (samples near both arc endpoints): architecture + objective + DDIM
  CAN do multimodality -> the real-data collapse (ADR-029) is a property of single-
  future-per-scene imitation, not a model bug.
- collapse to 1 mode at the midpoint: the x0-MSE/DDIM path itself cannot capture
  multimodality as wired -> fix the objective before any data work.
"""
import argparse, json, math
import numpy as np, torch
from models.policy_heads import DiffusionHead, HeadConfig, CosineSchedule
from models.samplers import ddim_sample
from models.f0_dataset import scale_future, unscale_future, FUTURE_SCALE

def build_arcs(C, H, spread_m, seed):
    """Two mirrored arc futures per context, in METERS, shape (C,2,H,3).
    mode 0 curves +y, mode 1 curves -y; both reach ~reach_m ahead. heading=0."""
    g = torch.Generator().manual_seed(seed)
    reach = 25.0 + 5.0 * torch.rand(C, generator=g)            # forward reach per context
    lat = (spread_m / 2.0) * (1.0 + 0.2 * torch.rand(C, generator=g))  # half lateral spread
    s = torch.linspace(0, 1, H).view(1, H)                      # 0..1 along horizon
    arcs = torch.zeros(C, 2, H, 3)
    for c in range(C):
        x = reach[c] * s.squeeze(0)                             # forward (x) same for both modes
        y_mag = lat[c] * (s.squeeze(0) ** 2)                    # quadratic lateral -> arc
        arcs[c, 0, :, 0] = x; arcs[c, 0, :, 1] = +y_mag        # mode 0: left
        arcs[c, 1, :, 0] = x; arcs[c, 1, :, 1] = -y_mag        # mode 1: right
    return arcs  # meters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--L", type=int, default=8)               # memory tokens per context
    ap.add_argument("--spread-m", type=float, default=24.0)   # endpoint separation (>> 3.5m lane)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=64)              # samples/context at eval
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--eps-m", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HeadConfig()
    H, d = cfg.horizon, cfg.d_model
    head = DiffusionHead(cfg).to(dev).train()
    sched = CosineSchedule(T=cfg.T).to(dev)

    # fixed synthetic contexts (bypass encoder) + scaled bimodal targets
    memory = torch.randn(a.contexts, a.L, d, generator=torch.Generator().manual_seed(7)).to(dev)
    arcs_m = build_arcs(a.contexts, H, a.spread_m, seed=11).to(dev)        # (C,2,H,3) meters
    arcs_scaled = scale_future(arcs_m)                                     # (C,2,H,3) scaled
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)

    for step in range(a.steps):
        ci = torch.randint(0, a.contexts, (a.batch,), device=dev)
        mi = torch.randint(0, 2, (a.batch,), device=dev)                  # 50/50 mode pick
        x0 = arcs_scaled[ci, mi]                                          # (B,H,3) scaled
        mem = memory[ci]                                                  # (B,L,d)
        t = torch.randint(0, sched.alphas_cumprod.shape[0], (a.batch,), device=dev)
        eps = torch.randn_like(x0)
        x_t = sched.q_sample(x0, t, eps)
        pred = head(x_t, t, mem)                                          # x0-pred MSE (= compute_loss)
        loss = torch.nn.functional.mse_loss(pred, x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {loss.item():.5f}", flush=True)

    # eval: K DDIM samples per context
    head.eval()
    with torch.no_grad():
        gen = torch.Generator(device=dev).manual_seed(0)
        samp = ddim_sample(head, sched, memory, goal=None, num_samples=a.K,
                           num_steps=a.ddim_steps, generator=gen)         # (C,K,H,3) scaled
        samp = unscale_future(samp)                                       # meters
    end = samp[..., -1, :2].float().cpu().numpy()                         # (C,K,2) endpoints
    arcA = arcs_m[:, 0, -1, :2].cpu().numpy()                             # (C,2) mode-0 endpoint
    arcB = arcs_m[:, 1, -1, :2].cpu().numpy()
    mid = 0.5 * (arcA + arcB)

    def n_modes(ep):                                                      # union-find at eps_m
        K = len(ep); par = list(range(K))
        def f(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
        for i in range(K):
            for j in range(i+1, K):
                if D[i, j] < a.eps_m: par[f(i)] = f(j)
        return len({f(i) for i in range(K)})

    modes, disp, frac_nearA, frac_nearB, frac_mid = [], [], [], [], []
    for c in range(a.contexts):
        ep = end[c]; D = np.linalg.norm(ep[:, None] - ep[None, :], axis=-1)
        modes.append(n_modes(ep))
        disp.append(D[np.triu_indices(a.K, 1)].mean())
        dA = np.linalg.norm(ep - arcA[c], axis=-1)
        dB = np.linalg.norm(ep - arcB[c], axis=-1)
        dM = np.linalg.norm(ep - mid[c], axis=-1)
        frac_nearA.append(float((dA < a.eps_m).mean()))
        frac_nearB.append(float((dB < a.eps_m).mean()))
        frac_mid.append(float((dM < a.eps_m).mean()))
    modes = np.array(modes)
    res = {
        "device": str(dev), "contexts": a.contexts, "K": a.K, "spread_m": a.spread_m,
        "steps": a.steps, "eps_m": a.eps_m, "final_train_loss": float(loss.item()),
        "modes": {"mean": float(modes.mean()), "frac_ge2": float((modes >= 2).mean()),
                  "max": int(modes.max())},
        "endpoint_dispersion_m": {"median": float(np.median(disp)), "mean": float(np.mean(disp))},
        "frac_samples_near_modeA_median": float(np.median(frac_nearA)),
        "frac_samples_near_modeB_median": float(np.median(frac_nearB)),
        "frac_samples_near_midpoint_median": float(np.median(frac_mid)),
        "both_modes_covered_frac_contexts": float(np.mean(
            [(a_ > 0.05 and b_ > 0.05) for a_, b_ in zip(frac_nearA, frac_nearB)])),
    }
    print(json.dumps(res, indent=2))
    if a.out: json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
