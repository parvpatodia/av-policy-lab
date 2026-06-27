"""Definitive moderation re-analysis (rigor-upgrade WS-B + WS-E).

WHY this file exists alongside analyze_moderation.py:
  1. WS-E (sensitivity): the composite CLS is near its ceiling (0.85-0.96), so a
     true moderation could be masked by clipping. nuPlan's aggregator parquet
     carries the PDM SUB-COMPONENTS, several of which are NOT ceilinged -- above
     all the CONTINUOUS `ego_expert_L2_error` (meters) and
     `ego_progress_along_expert_route`. Re-running the moderation on these
     separates "multimodality genuinely does not help" (null on a metric that CAN
     move) from "CLS can't express it" (a ceiling artifact).
  2. WS-B (pre-registered inference): RESEARCH_PROTOCOL.md:100-105 pre-registered
     "OLS Delta ~ 1 + F4 with scenario_type FIXED EFFECTS, WILD-CLUSTER-BOOTSTRAP
     SE by scenario_type ... precise-slope ~0 established by TOST equivalence."
     The shipped analyze_moderation.py used plain HC3 with neither. This file
     ships the registered method.

Inference, stated plainly:
  - Fixed effects: add a dummy per scenario_type. The F4 slope b1 is then
    identified ONLY from WITHIN-type F4 variation, so it cannot be confounded by
    "some scenario types are just harder" (between-type differences). F4 was built
    to vary across types, so this matters.
  - Cluster-robust + wild-cluster bootstrap: closed-loop scores of scenarios of
    the same type are correlated; plain HC3 assumes independence and is
    anticonservative. With only a few dozen clusters the asymptotic cluster SE is
    biased, so we use the restricted wild-cluster bootstrap (Rademacher weights),
    which is the standard small-cluster-count fix. REF: Cameron, Gelbach & Miller
    (2008); MacKinnon & Webb (2017).
  - TOST: to assert "precise slope ~ 0" we run two one-sided tests against an
    equivalence margin, instead of treating a non-significant slope as zero
    (the fallacy the protocol explicitly forbade). REF: Lakens (2017).
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

_NON_SCENARIO = {"final_score"}

# Outcomes to test. (column, higher_is_better, ceilinged?) -- comments are the WHY.
OUTCOMES = [
    ("score", True, True),                              # composite CLS (ceilinged baseline)
    ("ego_expert_L2_error", False, False),              # CONTINUOUS meters, no ceiling -- key
    ("ego_progress_along_expert_route", True, False),   # continuous progress fraction
    ("time_to_collision_within_bound", True, True),     # safety sub-score
    ("no_ego_at_fault_collisions", True, True),
    ("drivable_area_compliance", True, True),
    ("ego_is_comfortable", True, True),
    ("ego_is_making_progress", True, True),
]


def read_cell_metric(agg_parquet: str, col: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """aggregator parquet -> ({token: metric value}, {token: scenario_type}) for
    real per-scenario rows only (log_name not null, not 'final_score')."""
    import pandas as pd
    df = pd.read_parquet(agg_parquet)
    vals: Dict[str, float] = {}
    types: Dict[str, str] = {}
    for _, r in df.iterrows():
        tok, log = r["scenario"], r["log_name"]
        if log is None or (isinstance(log, float) and np.isnan(log)):
            continue
        if tok in _NON_SCENARIO:
            continue
        v = r.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals[str(tok)] = float(v)
        types[str(tok)] = str(r.get("scenario_type", "?"))
    return vals, types


def latest_aggregator(cell_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(str(Path(cell_dir) / "**" / "aggregator_metric" / "*.parquet"),
                            recursive=True))
    return hits[-1] if hits else None


# ----------------------------- inference core -----------------------------

def _design(x: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build [1, x, type-dummies(drop first)] design matrix. Returns (X, uniq_groups)."""
    uniq = np.unique(groups)
    # drop the first level for identification (its effect folds into the intercept)
    dummies = np.stack([(groups == g).astype(float) for g in uniq[1:]], axis=1) \
        if len(uniq) > 1 else np.zeros((len(x), 0))
    X = np.column_stack([np.ones_like(x), x, dummies])
    return X, uniq


def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return beta, resid, XtX_inv


def _cluster_se_beta1(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray,
                      groups: np.ndarray, k_idx: int = 1) -> float:
    """CR1 cluster-robust SE for coefficient k_idx, clustered by `groups`."""
    G = len(np.unique(groups)); n, k = X.shape
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    adj = (G / max(1, G - 1)) * ((n - 1) / max(1, n - k))   # CR1 small-sample scale
    cov = adj * (XtX_inv @ meat @ XtX_inv)
    return float(np.sqrt(max(cov[k_idx, k_idx], 0.0)))


def wild_cluster_test(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      B: int = 9999, seed: int = 0) -> dict:
    """Restricted wild-cluster bootstrap for H0: beta1 (slope on x) = 0, with
    scenario_type fixed effects, clustered by scenario_type.

    Returns beta1, cluster-robust SE, t, and bootstrap one/two-sided p.
    Procedure (Cameron-Gelbach-Miller bootstrap-t):
      1. full fit -> beta1_hat, cluster-robust t_hat.
      2. restricted fit (drop x) -> restricted residuals r and fitted yhat0.
      3. resample y* = yhat0 + w_g * r with cluster Rademacher signs w_g; refit
         full; recompute cluster-robust t*. p = freq(|t*| >= |t_hat|).
    """
    rng = np.random.default_rng(seed)
    X, _ = _design(x, groups)
    beta, resid, XtXi = _ols(X, y)
    b1 = float(beta[1])
    se = _cluster_se_beta1(X, resid, XtXi, groups, 1)
    t_hat = b1 / se if se > 0 else 0.0

    # restricted model: y ~ 1 + type FE (no x)
    Xr = np.delete(X, 1, axis=1)
    br, rr, XtXir = _ols(Xr, y)
    yhat0 = Xr @ br

    uniq = np.unique(groups)
    t_star = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(uniq))             # one sign per cluster
        wmap = {g: w[i] for i, g in enumerate(uniq)}
        wv = np.array([wmap[g] for g in groups])
        ystar = yhat0 + wv * rr
        beta_s, resid_s, XtXi_s = _ols(X, ystar)
        se_s = _cluster_se_beta1(X, resid_s, XtXi_s, groups, 1)
        t_star[b] = (beta_s[1] / se_s) if se_s > 0 else 0.0
    p_two = float((np.abs(t_star) >= abs(t_hat)).mean())
    p_upper = float((t_star >= t_hat).mean())                    # H1: beta1 > 0
    return {"beta1": b1, "cluster_se": se, "t": t_hat,
            "p_wcr_twosided": p_two, "p_wcr_onesided_gt": p_upper,
            "n": int(len(y)), "n_clusters": int(len(uniq))}


def tost_equivalence(beta1: float, se: float, dof: int, margin: float) -> dict:
    """Two one-sided tests: is beta1 within +/- margin of 0? Equivalence is
    established (at alpha=.05) iff the 90% CI [b1 +/- 1.645*se] lies inside
    [-margin, +margin]. Reports the larger of the two one-sided p's."""
    from math import erf, sqrt
    def norm_sf(z): return 0.5 * (1 - erf(z / sqrt(2)))
    t_lower = (beta1 + margin) / se if se > 0 else 0.0   # H0: b1 <= -margin
    t_upper = (margin - beta1) / se if se > 0 else 0.0   # H0: b1 >= +margin
    p_lower = norm_sf(t_lower)
    p_upper = norm_sf(t_upper)
    p_tost = max(p_lower, p_upper)
    ci90 = (beta1 - 1.645 * se, beta1 + 1.645 * se)
    equivalent = (ci90[0] > -margin) and (ci90[1] < margin)
    return {"margin": margin, "p_tost": float(p_tost),
            "ci90": [float(ci90[0]), float(ci90[1])], "equivalent": bool(equivalent)}


# ----------------------------- pipeline -----------------------------

def paired_delta_metric(det_dir: str, diff_dir: str, col: str
                        ) -> Tuple[Dict[str, float], Dict[str, str]]:
    det, t1 = read_cell_metric(latest_aggregator(det_dir), col)
    diff, t2 = read_cell_metric(latest_aggregator(diff_dir), col)
    toks = set(det) & set(diff)
    delta = {t: diff[t] - det[t] for t in toks}
    types = {t: t1.get(t, t2.get(t, "?")) for t in toks}
    return delta, types


def analyze_outcome(col: str, det_route, diff_route, det_precise, diff_precise,
                    f4: Dict[str, float], B: int) -> dict:
    dr, tr = paired_delta_metric(det_route, diff_route, col)
    dp, tp = paired_delta_metric(det_precise, diff_precise, col)
    types = {**tp, **tr}
    res = {"outcome": col}
    for cond, dd in (("route", dr), ("precise", dp)):
        toks = sorted(set(dd) & set(f4))
        if len(toks) < 30:
            res[cond] = None; continue
        y = np.array([dd[t] for t in toks])
        x = np.array([f4[t] for t in toks])
        g = np.array([types[t] for t in toks])
        wc = wild_cluster_test(x, y, g, B=B)
        res[cond] = wc
    # headline contrast: C_i = Delta_route - Delta_precise, regress on F4
    ctoks = sorted(set(dr) & set(dp) & set(f4))
    if len(ctoks) >= 30:
        c = np.array([dr[t] - dp[t] for t in ctoks])
        x = np.array([f4[t] for t in ctoks])
        g = np.array([types[t] for t in ctoks])
        wc = wild_cluster_test(x, c, g, B=B)
        # equivalence margin: 0.02 for CLS (pre-reg "subtle"); for other metrics
        # scale by the outcome's robust spread so the margin is comparable
        margin = 0.02 if col == "score" else 0.1 * float(np.std(c) + 1e-9)
        wc["tost"] = tost_equivalence(wc["beta1"], wc["cluster_se"], len(ctoks) - 2, margin)
        # also the OLD naive HC3 (no FE, no cluster) for side-by-side
        from numpy.linalg import inv
        X = np.column_stack([np.ones_like(x), x]); bb = inv(X.T@X)@X.T@c
        rr = c - X@bb; h = np.einsum("ij,jk,ik->i", X, inv(X.T@X), X)
        om = rr**2/np.clip((1-h)**2,1e-12,None); cov = inv(X.T@X)@(X.T*om)@X@inv(X.T@X)
        wc["naive_hc3_beta1"] = float(bb[1]); wc["naive_hc3_se"] = float(np.sqrt(cov[1,1]))
        res["contrast"] = wc
    return res


def run(args) -> dict:
    f4_all = json.load(open(args.f4_scores))
    f4 = {t: r["f4"] for t, r in f4_all.items() if r.get("f4") is not None}
    out = {"reactive": args.tag, "outcomes": {}}
    for col, hib, ceil in OUTCOMES:
        try:
            r = analyze_outcome(col, args.det_route, args.diff_route,
                                args.det_precise, args.diff_precise, f4, args.bootstrap)
        except Exception as e:
            r = {"outcome": col, "error": str(e)}
        r["higher_is_better"] = hib; r["ceilinged"] = ceil
        out["outcomes"][col] = r
        c = r.get("contrast")
        if c:
            print(f"[{col:34s}] contrast beta1={c['beta1']:+.4f} clSE={c['cluster_se']:.4f} "
                  f"t={c['t']:+.2f} p_wcr(2)={c['p_wcr_twosided']:.3f} "
                  f"| naiveHC3 beta1={c['naive_hc3_beta1']:+.4f} se={c['naive_hc3_se']:.4f} "
                  f"| equiv={c['tost']['equivalent']}")
        else:
            print(f"[{col:34s}] (insufficient tokens)")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("wrote", args.out)
    return out


def selftest():
    """Validate the wild-cluster bootstrap: under a TRUE null (no x effect, but
    strong clustered structure) the one-sided p should be ~uniform (≈0.5 on
    average); under a TRUE effect it should be small. 12 clusters."""
    rng = np.random.default_rng(1)
    def make(effect):
        G = 12; per = 60
        g = np.repeat(np.arange(G), per)
        clamp = rng.normal(0, 1.0, G)[g]                 # strong cluster random effect
        x = rng.normal(0, 1, G * per) + 0.3 * clamp      # x correlated within cluster
        y = effect * x + clamp + rng.normal(0, 1, G * per)
        return x, y, g.astype(str)
    # null
    ps = []
    for s in range(40):
        rng = np.random.default_rng(s)
        x, y, g = make(0.0)
        ps.append(wild_cluster_test(x, y, g, B=999, seed=s)["p_wcr_twosided"])
    null_mean = float(np.mean(ps)); null_rej = float(np.mean(np.array(ps) < 0.05))
    # effect
    x, y, g = make(0.5)
    eff = wild_cluster_test(x, y, g, B=2999, seed=0)
    print(f"SELFTEST null: mean p={null_mean:.3f} (expect ~0.5), "
          f"type-I @.05={null_rej:.3f} (expect ~0.05)")
    print(f"SELFTEST effect=0.5: beta1={eff['beta1']:.3f} p_wcr2={eff['p_wcr_twosided']:.4f} "
          f"(expect small)")
    ok = (0.40 < null_mean < 0.60) and (null_rej <= 0.15) and (eff["p_wcr_twosided"] < 0.05)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--f4-scores")
    p.add_argument("--det-route"); p.add_argument("--diff-route")
    p.add_argument("--det-precise"); p.add_argument("--diff-precise")
    p.add_argument("--tag", default="r?")
    p.add_argument("--bootstrap", type=int, default=9999)
    p.add_argument("--out")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)
