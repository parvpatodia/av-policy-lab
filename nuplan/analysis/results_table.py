"""Assemble the paper's closed-loop results from sharded, multi-seed eval runs.

Layout produced by eval_array.sbatch:
    {RUNS}/{run_tag}/{head}_{goal}_shard{k}/eval/aggregator_metric/*.parquet
with one such tree per seed (run_tag encodes the seed, e.g. v3_seed0).

This module:
  1. merges shard parquets within a (cell, seed) -> {token: CLS}
  2. averages CLS over seeds per token (reduces seed noise BEFORE pairing,
     so Delta_i is a seed-stable per-scenario quantity)
  3. per-cell summary: mean CLS over tokens with a bootstrap CI
  4. baseline rows (idm, log_future) the same way (single run, no seeds)
  5. hands the seed-averaged per-cell token scores to analyze_moderation

WHY seed-average before pairing (not pair-then-average): the moderator F4 is
per-scenario; a seed-stable CLS per (cell, token) gives one Delta_i per
scenario with the seed variance already integrated out, which is what the
per-scenario regression assumes. Across-seed spread is reported separately
as the cell-level CI.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import analyze_moderation as M

CELLS = [("det", "route"), ("det", "precise"), ("diff", "route"), ("diff", "precise")]


def merge_shards(cell_seed_dir: str) -> Dict[str, float]:
    """All shard aggregator parquets under one (cell, seed) -> {token: CLS}."""
    out: Dict[str, float] = {}
    pattern = str(Path(cell_seed_dir) / "**" / "aggregator_metric" / "*.parquet")
    for pq in sorted(glob.glob(pattern, recursive=True)):
        out.update(M.read_cell_scores(pq))
    return out


def cell_scores_over_seeds(runs_root: str, run_tag_fmt: str, seeds: List[int],
                           head: str, goal: str) -> Dict[str, float]:
    """Seed-averaged {token: CLS} for one cell. run_tag_fmt has a {seed} slot."""
    per_seed: List[Dict[str, float]] = []
    for s in seeds:
        d = Path(runs_root) / run_tag_fmt.format(seed=s) / f"{head}_{goal}"
        # eval_array writes shard dirs {head}_{goal}_shard{k}; collect them
        shard_dirs = sorted(glob.glob(str(
            Path(runs_root) / run_tag_fmt.format(seed=s) / f"{head}_{goal}_shard*")))
        merged: Dict[str, float] = {}
        for sd in shard_dirs:
            merged.update(merge_shards(sd))
        if not merged and d.exists():           # unsharded fallback
            merged = merge_shards(str(d))
        if merged:
            per_seed.append(merged)
    if not per_seed:
        return {}
    common = set(per_seed[0])
    for d in per_seed[1:]:
        common &= set(d)
    return {t: float(np.mean([d[t] for d in per_seed])) for t in common}


def summarize(scores: Dict[str, float], B: int = 10000, seed: int = 0) -> dict:
    v = np.array(list(scores.values()))
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "ci95": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return {"n": len(v), "mean": float(v.mean()),
            "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))}


def run(args) -> dict:
    f4 = {t: r["f4"] for t, r in json.load(open(args.f4_scores)).items()
          if r.get("f4") is not None}
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = {}
    print(f"{'cell':16s} {'n':>5} {'mean CLS':>9} {'95% CI':>20}")
    for head, goal in CELLS:
        sc = cell_scores_over_seeds(args.runs_root, args.run_tag_fmt, seeds, head, goal)
        cells[f"{head}_{goal}"] = sc
        s = summarize(sc)
        ci = s["ci95"]
        print(f"{head+'_'+goal:16s} {s['n']:5d} {s['mean']:9.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    for name, base_dir in (("idm", args.idm_dir), ("log_future", args.log_future_dir)):
        if base_dir:
            sc = merge_shards(base_dir)
            s = summarize(sc)
            ci = s["ci95"]
            print(f"{name:16s} {s['n']:5d} {s['mean']:9.4f} "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]  (baseline)")

    # moderation per condition, using the seed-averaged cells
    out = {"cells": {k: summarize(v) for k, v in cells.items()}}
    deltas = {}
    for cond, det_key, diff_key in (("route", "det_route", "diff_route"),
                                     ("precise", "det_precise", "diff_precise")):
        det, diff = cells[det_key], cells[diff_key]
        toks = set(det) & set(diff)
        if not toks:
            continue
        delta = {t: diff[t] - det[t] for t in toks}
        deltas[cond] = delta
        res = M.moderation(delta, f4, cond)
        out[cond] = M.asdict(res)
        print(f"\n== {cond}: mean Delta {res.mean_delta:+.4f} "
              f"CI{res.mean_delta_ci95}  beta1 {res.beta1:+.4f} "
              f"p(1-sided) {res.beta1_p_onesided:.4f}  rho {res.spearman_rho:+.3f}")
    if "route" in deltas and "precise" in deltas:
        cres = M.moderation_contrast(deltas["route"], deltas["precise"], f4)
        out["contrast"] = M.asdict(cres)
        out["route_minus_precise_beta1"] = cres.beta1   # back-compat
        print(f"\n== contrast beta1(route)-beta1(precise) = {cres.beta1:+.4f}  "
              f"HC3 SE {cres.beta1_se_hc3:.4f}  p(1-sided) {cres.beta1_p_onesided:.4f}"
              f"  (hypothesis: > 0)")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--run-tag-fmt", default="v3_seed{seed}",
                   help="dir name per seed; must contain {seed}")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--f4-scores", required=True)
    p.add_argument("--idm-dir"); p.add_argument("--log-future-dir")
    p.add_argument("--out")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
