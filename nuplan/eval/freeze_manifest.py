"""Freeze the closed-loop evaluation scenario manifest (pre-registration).

The eval set must be fixed and committed BEFORE any cell is unblinded, or
the comparison invites cherry-picking. This writes manifest.json containing
the token list, the F4 stratification, a SHA256 content hash, and the git
commit, so the exact eval set is auditable and reproducible.

Stratified sampling over 4 F4 BANDS, not deciles: 60% of scenarios score
F4 == 0 exactly, so equal-count deciles collapse (bottom 6 deciles all
[0,0]) and would starve the eval set of either the zero-anchor or the
signal range. The 4 bands are {zero, low (0,1/3], med (1/3,2/3], high
(2/3,1]} with equal allocation.

WHY equal-allocation (a balanced-X design): selection on the REGRESSOR (F4)
does not bias the OLS moderation slope beta1; balancing X across its range
MAXIMIZES slope precision (leverage at the extremes). The cost is that the
marginal mean Delta and the intercept are no longer population-representative
and must NOT be read as population values; this analysis is slope-focused
(does the diffusion advantage grow with ambiguity), which is the hypothesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def freeze(f4_path: str, n: int, seed: int, out_path: str, exclude=None) -> dict:
    exclude = set(exclude or [])  # keep out (e.g. eval manifest) so a probe is disjoint
    f4_all = json.load(open(f4_path))
    scored = {t: r for t, r in f4_all.items()
              if r.get("f4") is not None and not r.get("excluded")}
    toks = sorted(scored)                      # deterministic order before sampling
    f4 = np.array([scored[t]["f4"] for t in toks])
    rng = np.random.default_rng(seed)

    # 4 bands; zero is its own stratum because it is a 60% point mass
    bands = [
        ("zero", lambda v: v == 0.0),
        ("low",  lambda v: 0.0 < v <= 1 / 3),
        ("med",  lambda v: 1 / 3 < v <= 2 / 3),
        ("high", lambda v: v > 2 / 3),
    ]
    per = max(1, n // len(bands))
    chosen: list[str] = []
    strata = {}
    for name, pred in bands:
        in_bin = [t for t, v in zip(toks, f4) if pred(v) and t not in exclude]
        k = min(per, len(in_bin))
        pick = sorted(rng.choice(in_bin, k, replace=False).tolist()) if in_bin else []
        chosen.extend(pick)
        strata[name] = {"available": len(in_bin), "n": len(pick)}
    chosen = sorted(set(chosen))

    payload = {
        "tokens": chosen,
        "n_requested": n, "n_actual": len(chosen),
        "seed": seed,
        "f4_source": f4_path,
        "strata": strata,
        "f4_by_token": {t: scored[t]["f4"] for t in chosen},
    }
    content = json.dumps(payload["tokens"], sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(content).hexdigest()
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        payload["git_commit"] = "unknown"
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"froze {len(chosen)} tokens -> {out_path}")
    print(f"sha256={payload['sha256'][:16]}  commit={payload['git_commit'][:8]}")
    for k, v in strata.items():
        print(f"  {k}: available={v['available']} chosen={v['n']}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--f4-scores", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None, help="manifest.json whose tokens to exclude (disjoint probe)")
    a = ap.parse_args()
    excl = set(json.load(open(a.exclude))["tokens"]) if a.exclude else None
    freeze(a.f4_scores, a.n, a.seed, a.out, exclude=excl)
