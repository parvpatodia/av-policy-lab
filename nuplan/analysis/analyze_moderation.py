"""F4 moderation analysis: does the diffusion-vs-deterministic CLS gap grow
with scene ambiguity, and only under route-region conditioning?

This is the file that produces the experiment's verdict. It is deliberately
small, pure (no sim, no torch), and fully unit-tested, because every number
in the paper comes through here.

Pipeline:
  1. read each cell's aggregator parquet -> per-scenario CLS keyed by token
     (drop the type-aggregate and final_score rows; keep real scenarios)
  2. pair cells: Delta_i = CLS_i(diff) - CLS_i(det), within a goal condition
  3. join F4_i (model-free moderator, scored on UNPERTURBED extractions)
  4. moderation tests:
       primary   : OLS Delta ~ 1 + F4, HC3 robust SE, one-sided beta1>0
       robustness: Spearman(Delta, F4), Theil-Sen slope
       paired CI : bootstrap 95% CI on mean Delta (per condition)
  5. cross-condition: the hypothesis is beta1(route) > 0 and beta1(precise)~0;
     report both and their difference.

REF: HC3 (MacKinnon & White 1985); Theil-Sen for robustness to bounded,
non-normal Delta; the moderation framing (not mediation) is ADR-013.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# rows the aggregator emits that are NOT individual scenarios
_NON_SCENARIO = {"final_score"}


def read_cell_scores(agg_parquet: str) -> Dict[str, float]:
    """aggregator parquet -> {scenario_token: CLS score}. Per-scenario rows
    only (log_name not null, scenario id is a real token)."""
    import pandas as pd
    df = pd.read_parquet(agg_parquet)
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        tok, log = r["scenario"], r["log_name"]
        # real scenario rows carry a log_name; aggregate rows have None and a
        # name that is a scenario_type or 'final_score'
        if log is None or (isinstance(log, float) and np.isnan(log)):
            continue
        if tok in _NON_SCENARIO:
            continue
        out[str(tok)] = float(r["score"])
    return out


def latest_aggregator(cell_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(str(Path(cell_dir) / "**" / "aggregator_metric" / "*.parquet"),
                            recursive=True))
    return hits[-1] if hits else None


@dataclass
class ModerationResult:
    condition: str
    n: int
    beta0: float
    beta1: float
    beta1_se_hc3: float
    beta1_t: float
    beta1_p_onesided: float
    spearman_rho: float
    theilsen_slope: float
    mean_delta: float
    mean_delta_ci95: Tuple[float, float]


def _ols_hc3(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (beta0, beta1, HC3 SE of beta1) for y = b0 + b1 x."""
    X = np.stack([np.ones_like(x), x], axis=1)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)        # leverages
    # HC3: divide squared residuals by (1 - h_i)^2
    omega = (resid ** 2) / np.clip((1 - h) ** 2, 1e-12, None)
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[1, 1]))


def _student_sf(t: float, dof: int) -> float:
    """One-sided upper-tail P(T > t) for Student-t, no scipy dependency."""
    # regularized incomplete beta via continued fraction is overkill here;
    # use the standard transformation to the incomplete beta I_x(dof/2, 1/2)
    import math
    x = dof / (dof + t * t)

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        d = FPMIN if abs(d) < FPMIN else d
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
            c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
            c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    def betai(a, b, x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b

    ix = betai(dof / 2.0, 0.5, x)            # = 2 * P(T > |t|) total two tails halved
    two_tail = ix
    p_upper = two_tail / 2.0 if t > 0 else 1.0 - two_tail / 2.0
    return p_upper


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 200000) -> float:
    n = len(x)
    slopes = []
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        if x[j] != x[i]:
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else 0.0


def _bootstrap_mean_ci(d: np.ndarray, B: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def moderation(delta: Dict[str, float], f4: Dict[str, float],
               condition: str) -> ModerationResult:
    toks = sorted(set(delta) & set(f4))
    d = np.array([delta[t] for t in toks])
    x = np.array([f4[t] for t in toks])
    b0, b1, se = _ols_hc3(x, d)
    t_stat = b1 / se if se > 0 else 0.0
    p = _student_sf(t_stat, len(toks) - 2)
    return ModerationResult(
        condition=condition, n=len(toks),
        beta0=b0, beta1=b1, beta1_se_hc3=se, beta1_t=t_stat,
        beta1_p_onesided=p,
        spearman_rho=_spearman(x, d),
        theilsen_slope=_theil_sen(x, d),
        mean_delta=float(d.mean()),
        mean_delta_ci95=_bootstrap_mean_ci(d),
    )


def paired_delta(det_dir: str, diff_dir: str) -> Dict[str, float]:
    det = read_cell_scores(latest_aggregator(det_dir))
    diff = read_cell_scores(latest_aggregator(diff_dir))
    toks = set(det) & set(diff)
    return {t: diff[t] - det[t] for t in toks}


def run(args) -> dict:
    f4_all = json.load(open(args.f4_scores))
    f4 = {t: r["f4"] for t, r in f4_all.items() if r.get("f4") is not None}
    out = {}
    for cond, det_dir, diff_dir in (
        ("route", args.det_route, args.diff_route),
        ("precise", args.det_precise, args.diff_precise),
    ):
        if not (det_dir and diff_dir):
            continue
        delta = paired_delta(det_dir, diff_dir)
        res = moderation(delta, f4, cond)
        out[cond] = asdict(res)
        print(f"\n== {cond} (n={res.n})")
        print(f"   mean Delta = {res.mean_delta:+.4f}  95% CI {res.mean_delta_ci95}")
        print(f"   beta1 = {res.beta1:+.4f}  HC3 SE {res.beta1_se_hc3:.4f}  "
              f"t={res.beta1_t:.2f}  p(1-sided)={res.beta1_p_onesided:.4f}")
        print(f"   Spearman rho = {res.spearman_rho:+.3f}  "
              f"Theil-Sen = {res.theilsen_slope:+.4f}")
    if "route" in out and "precise" in out:
        diff_slope = out["route"]["beta1"] - out["precise"]["beta1"]
        out["route_minus_precise_beta1"] = diff_slope
        print(f"\n== hypothesis: beta1(route) - beta1(precise) = {diff_slope:+.4f}")
        print("   (predicted > 0: diffusion helps under ambiguity only when "
              "the goal does not already resolve it)")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f4-scores", required=True)
    p.add_argument("--det-route"); p.add_argument("--diff-route")
    p.add_argument("--det-precise"); p.add_argument("--diff-precise")
    p.add_argument("--out")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
