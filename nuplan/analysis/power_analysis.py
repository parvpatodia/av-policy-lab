"""Eval-set sizing for the moderation test (pre-registered, run 2026-06-12).

Monte Carlo over the EMPIRICAL F4 v1.1 distribution (n=5,604): simulate
Delta_i = beta1 * F4_i + N(0, sigma), fit OLS, one-sided t at 0.05.

Result table (2,000 sims/cell): with sd(F4)=0.34, stratified sampling adds
nothing over random (power within 2 points everywhere). n=1000 yields
power >= 0.83 for beta1 >= 0.05 at sigma <= 0.20; n=500 suffices if
sigma <= 0.10. DECISION: run a ~100-scenario pilot eval first to measure
sigma_Delta, then freeze n (500 or 1000) and the manifest BEFORE unblinding.

F4 scoring rule: the moderator is always computed on UNPERTURBED
extractions. v2-derived scores transfer to v3 by scenario token; enrichment
scenarios get a dedicated perturb_prob=0 scoring extraction.
"""
import json
import sys

import numpy as np


def power(f4_all, n, beta1, sigma, sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        f4 = rng.choice(f4_all, n)
        d = beta1 * f4 + rng.normal(0, sigma, n)
        X = np.stack([np.ones(n), f4], axis=1)
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        resid = d - X @ coef
        cov = (resid @ resid) / (n - 2) * np.linalg.inv(X.T @ X)
        hits += (coef[1] / np.sqrt(cov[1, 1])) > 1.645
    return hits / sims


if __name__ == "__main__":
    rows = [r for r in json.load(open(sys.argv[1])).values() if r.get("f4") is not None]
    f4 = np.array([r["f4"] for r in rows])
    print(f"empirical F4: n={len(f4)} mean={f4.mean():.3f} sd={f4.std():.3f}")
    print(f"{'n':>5} {'beta1':>6} {'sigma':>6} {'power':>6}")
    for n in (250, 500, 1000):
        for b in (0.03, 0.05, 0.10):
            for s in (0.10, 0.20):
                print(f"{n:5d} {b:6.2f} {s:6.2f} {power(f4, n, b, s):6.2f}")
