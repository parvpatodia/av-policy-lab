"""F5: train one cell of the 2x2 — {det, diff} x {route, precise}.

WHY one cell per invocation: SLURM jobs are the unit of scheduling and
failure; four independent runs with identical hyperparameters and seeds
are trivially comparable and individually resumable, while a single
four-model script couples their failure modes.

Fairness contract (the experiment's validity rests on this):
  - same seed -> same encoder init in every cell (seeded before build)
  - identical optimizer, LR schedule, batch size, grad clip, epochs
  - the ONLY differences are the head class and goal conditioning.

Checkpointing: latest.pt every epoch (covers the 8 h GPU window — an epoch
here is minutes), best.pt on val-minADE improvement. Resume restores
model/opt/RNG and continues at the next epoch; with num_workers=0 the
resumed run is bit-identical to an uninterrupted one (tested).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.f0_dataset import F0ShardDataset, unscale_future
from models.policy_heads import (
    CosineSchedule,
    DeterministicHead,
    DiffusionHead,
    HeadConfig,
    build_precise_goal,
)
from models.samplers import ddim_sample
from models.scene_encoder import SceneEncoder, SceneEncoderConfig

ENCODER_KEYS = (
    "ego", "agents", "agent_mask", "map_polylines", "map_mask",
    "crosswalks", "crosswalk_mask", "route_polyline", "route_mask",
    "traffic_lights",
)


class EMA:
    """Exponential moving average over encoder+head parameters.

    WHY on both heads: EMA is standard for diffusion policies (Chi et al.,
    arXiv:2303.04137 use it); fairness requires the regressor twin get the
    identical treatment. Validation and best.pt use EMA weights; the raw
    weights keep training. State is checkpointed so resume stays bit-exact.
    """

    def __init__(self, modules: dict, decay: float = 0.999):
        self.decay = decay
        self.modules = modules
        self.shadow = {
            f"{m}.{k}": p.detach().clone()
            for m, mod in modules.items() for k, p in mod.named_parameters()
        }

    @torch.no_grad()
    def update(self):
        for m, mod in self.modules.items():
            for k, p in mod.named_parameters():
                s = self.shadow[f"{m}.{k}"]
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap(self):
        """Exchange live and shadow parameters (call again to restore)."""
        for m, mod in self.modules.items():
            for k, p in mod.named_parameters():
                s = self.shadow[f"{m}.{k}"]
                tmp = p.detach().clone()
                p.copy_(s)
                s.copy_(tmp)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        for k, v in sd["shadow"].items():
            self.shadow[k].copy_(v)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", choices=("det", "diff"), required=True)
    p.add_argument("--goal", choices=("route", "precise"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-stride", type=int, default=10)
    p.add_argument("--patience", type=int, default=15,
                   help="early stop after this many epochs without val minADE improvement")
    p.add_argument("--val-k", type=int, default=8,
                   help="diffusion: candidates per scene for minADE")
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def run_name(args) -> str:
    return f"{args.head}_{args.goal}_seed{args.seed}"


def build_models(args):
    # WHY seed before construction: identical init across the four cells is
    # the capacity-matching guarantee at the weight level, not just the
    # architecture level.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    encoder = SceneEncoder(SceneEncoderConfig())
    head_cfg = HeadConfig()
    head = DeterministicHead(head_cfg) if args.head == "det" else DiffusionHead(head_cfg)
    schedule = CosineSchedule(T=head_cfg.T)
    return encoder, head, schedule


def goal_from_batch(args, fut_scaled: torch.Tensor):
    """Route condition: no goal token. Precise: near/far points from the label."""
    return build_precise_goal(fut_scaled) if args.goal == "precise" else None


def compute_loss(args, head, schedule, memory, fut_scaled):
    goal = goal_from_batch(args, fut_scaled)
    if args.head == "det":
        pred = head(memory, goal=goal)
        return nn.functional.mse_loss(pred, fut_scaled)
    t = torch.randint(0, schedule.alphas_cumprod.shape[0], (fut_scaled.shape[0],),
                      device=fut_scaled.device)
    eps = torch.randn_like(fut_scaled)
    x_t = schedule.q_sample(fut_scaled, t, eps)
    x0_pred = head(x_t, t, memory, goal=goal)
    return nn.functional.mse_loss(x0_pred, fut_scaled)


@torch.no_grad()
def evaluate(args, encoder, head, schedule, loader, device) -> dict:
    """Meter-space ADE/FDE. Diffusion reports min over K DDIM samples."""
    encoder.eval(), head.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed)  # fixed val noise
    ade_sum = fde_sum = 0.0
    n = 0
    for batch in loader:
        # WHY tensor check: v2 shards carry scenario identifiers (str) for F4
        # scoring; strings have no .to() and must pass through untouched.
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        fut_scaled = batch["ego_future"]
        memory = encoder({k: batch[k] for k in ENCODER_KEYS})
        goal = goal_from_batch(args, fut_scaled)
        if args.head == "det":
            pred = head(memory, goal=goal).unsqueeze(1)            # (B,1,H,3)
        else:
            pred = ddim_sample(head, schedule, memory, goal=goal,
                               num_samples=args.val_k,
                               num_steps=args.ddim_steps, generator=gen)
        gt = unscale_future(fut_scaled).unsqueeze(1)               # (B,1,H,3)
        pred = unscale_future(pred)
        dist = (pred[..., :2] - gt[..., :2]).norm(dim=-1)          # (B,K,H)
        ade_sum += dist.mean(dim=-1).min(dim=1).values.sum().item()
        fde_sum += dist[..., -1].min(dim=1).values.sum().item()
        n += fut_scaled.shape[0]
    encoder.train(), head.train()
    return {"minADE": ade_sum / n, "minFDE": fde_sum / n, "n_val": n}


def save_ckpt(path: Path, encoder, head, opt, ema, epoch, global_step, best, args):
    payload = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_minADE": best,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)  # WHY atomic rename: a job killed mid-save must not corrupt latest.pt


def train(args) -> dict:
    device = torch.device(args.device)
    out = args.ckpt_dir / run_name(args)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    encoder, head, schedule = build_models(args)
    encoder.to(device), head.to(device), schedule.to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    ema = EMA({"encoder": encoder, "head": head}, decay=args.ema_decay)

    train_ds = F0ShardDataset(args.data_root, shuffle=True, seed=args.seed,
                              split="train", val_stride=args.val_stride)
    val_ds = F0ShardDataset(args.data_root, shuffle=False, seed=args.seed,
                            split="val", val_stride=args.val_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    start_epoch, global_step, best = 0, 0, math.inf
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
        else:
            print("[resume] pre-EMA checkpoint: shadow re-seeded from live weights")
        start_epoch, global_step = ck["epoch"] + 1, ck["global_step"]
        best = ck["best_minADE"]
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        random.setstate(ck["rng"]["python"])
        print(f"[resume] {latest} -> epoch {start_epoch}, best minADE {best:.3f}")

    bad_epochs = 0
    # WHY bf16 gate: V100s (no bf16 hardware) silently emulate or fail;
    # autocast only where bf16 is native (Ampere+)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    epoch = start_epoch - 1  # WHY: if the run already hit --epochs, the loop
    # body never executes and the return below must still be well-defined
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        t0, loss_sum, n_steps = time.time(), 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                memory = encoder({k: batch[k] for k in ENCODER_KEYS})
                loss = compute_loss(args, head, schedule, memory, batch["ego_future"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), args.grad_clip)
            # linear warmup, then constant — identical across cells, no total-step
            # dependence (IterableDataset has no len)
            warm = min(1.0, (global_step + 1) / max(1, args.warmup_steps))
            for g in opt.param_groups:
                g["lr"] = args.lr * warm
            opt.step()
            ema.update()
            loss_sum += loss.item()
            n_steps += 1
            global_step += 1

        ema.swap()
        val = evaluate(args, encoder, head, schedule, val_loader, device)
        ema.swap()
        row = {
            "epoch": epoch, "step": global_step,
            "train_loss": loss_sum / max(1, n_steps),
            "lr": opt.param_groups[0]["lr"],
            "sec": round(time.time() - t0, 1), **val,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{run_name(args)}] " + json.dumps(row))

        if val["minADE"] < best:
            best, bad_epochs = val["minADE"], 0
            save_ckpt(out / "best.pt", encoder, head, opt, ema, epoch, global_step, best, args)
        else:
            bad_epochs += 1
        save_ckpt(latest, encoder, head, opt, ema, epoch, global_step, best, args)
        if bad_epochs >= args.patience:
            print(f"[early-stop] no val improvement for {args.patience} epochs")
            break

    return {"best_minADE": best, "epochs_run": epoch + 1 - start_epoch,
            "global_step": global_step}


def main(argv=None):
    args = parse_args(argv)
    result = train(args)
    print(json.dumps({"run": run_name(args), **result}))


if __name__ == "__main__":
    main()
