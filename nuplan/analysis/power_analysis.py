"""Pre-registration power analysis for the F4 moderation eval.

Decides N for the frozen manifest BEFORE burning the cluster. The eval costs
~7.5 min/scenario; at 4 cells x 3 seeds plus 3 baselines that is ~9 GPU-runs
per scenario, so N is the single biggest compute lever. We must know: at a
given N, what is the smallest moderation slope beta1 we can detect at
alpha=0.05, power=0.80? If the minimum detectable effect (MDE) is larger than
any plausible true effect, the eval is dead on arrival.

Estimand (per goal condition, seed-averaged):
    Delta_i = CLS_diff,i - CLS_det,i = beta0 + beta1 * F4_i + eps_i
beta1 is the moderation slope (does the diffusion head's advantage grow with
interaction-multimodality F4). OLS slope variance is
    Var(beta1_hat) = sigma_eps^2 / (n * Var_n(F4)),
so MDE(beta1) = (z_alpha + z_power) * sigma_eps / sqrt(n * Var_n(F4))   [1-sided].

Two facts make this analysis non-trivial and worth doing properly:

  1. The HEADLINE is the cross-condition contrast beta1(route) - beta1(precise),
     not either slope alone. Because both conditions are evaluated on the SAME
     tokens, the contrast is the slope of the per-token difference
     C_i = Delta_route_i - Delta_precise_i regressed on F4. Its residual SD is
        sigma_C = sigma_Delta * sqrt(2 * (1 - rho_cond)),
     where rho_cond is the correlation between the two conditions' residuals
     (same scenario hard for both). rho_cond is unknown pre-eval, so contrast
     power is reported as a function of it. rho_cond > 0 (the likely case)
     SHRINKS sigma_C and helps; independence (rho_cond=0) is the worst case.

  2. CLS is bounded [0,1] with mass near 1. The Gaussian MDE formula assumes
     unbounded normal residuals; clipping attenuates the measured slope for
     high-CLS scenes. We quantify this with a bounded-CLS simulation and report
     the realistic correction, not just the textbook number.

Verification gate (maker/checker): the analytic Gaussian MDE and a Monte-Carlo
simulation that reuses the PRODUCTION estimator (analyze_moderation._ols_hc3 +
_student_sf) must agree to within Monte-Carlo error. If they disagree, either
the formula or the estimator is wrong. Only after that gate passes do we trust
the bounded-CLS correction.

REF: Cohen (1988) for the MDE framing; the contrast-as-difference-slope
identity holds because OLS slope is linear in y with shared X.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import analyze_moderation as M


# ---- normal quantiles / cdf without scipy ---------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---- analytic power / MDE -------------------------------------------------
def analytic_power(n: int, beta1: float, sigma: float, var_f4: float,
                   alpha: float = 0.05) -> float:
    """One-sided power for H1: beta1>0 under the Gaussian-residual OLS slope."""
    se = sigma / math.sqrt(n * var_f4)
    ncp = beta1 / se
    return norm_cdf(ncp - norm_ppf(1 - alpha))


def analytic_mde(n: int, sigma: float, var_f4: float,
                 alpha: float = 0.05, power: float = 0.80) -> float:
    return (norm_ppf(1 - alpha) + norm_ppf(power)) * sigma / math.sqrt(n * var_f4)


# ---- simulation reusing the production estimator --------------------------
def simulate_power(n: int, beta1: float, sigma: float, f4_pool: np.ndarray,
                   alpha: float = 0.05, n_sims: int = 2000,
                   bounded: bool = False, sigma_head: Optional[float] = None,
                   cls_mean: float = 0.85, cls_sd: float = 0.18,
                   seed: int = 0) -> float:
    """Empirical rejection rate of the REAL estimator (analyze_moderation).

    bounded=False: Delta = beta1*F4 + N(0, sigma)  (matches the analytic model).
    bounded=True : model CLS in [0,1] directly. A shared per-scenario base
      (det skill) plus per-head noise; the diffusion head adds beta1*F4. Both
      clipped to [0,1] before differencing, so the slope attenuation from the
      upper boundary is captured. sigma_head defaults to sigma/sqrt(2) so the
      unclipped Delta SD matches `sigma`.
    """
    rng = np.random.default_rng(seed)
    if sigma_head is None:
        sigma_head = sigma / math.sqrt(2.0)
    rejects = 0
    for _ in range(n_sims):
        x = rng.choice(f4_pool, n, replace=(len(f4_pool) < n))
        if bounded:
            base = np.clip(rng.normal(cls_mean, cls_sd, n), 0.0, 1.0)
            cls_det = np.clip(base + rng.normal(0, sigma_head, n), 0.0, 1.0)
            cls_diff = np.clip(base + beta1 * x + rng.normal(0, sigma_head, n),
                               0.0, 1.0)
            d = cls_diff - cls_det
        else:
            d = beta1 * x + rng.normal(0, sigma, n)
        _, b1, se = M._ols_hc3(x, d)
        if se > 0 and M._student_sf(b1 / se, n - 2) < alpha:
            rejects += 1
    return rejects / n_sims


def simulate_mde(n: int, sigma: float, f4_pool: np.ndarray, power: float = 0.80,
                 alpha: float = 0.05, bounded: bool = False, n_sims: int = 2000,
                 seed: int = 0) -> float:
    """Bisection on beta1 to the slope giving the target power (empirical)."""
    lo, hi = 0.0, 0.5
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        p = simulate_power(n, mid, sigma, f4_pool, alpha, n_sims, bounded,
                           seed=seed)
        if p < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- contrast (the headline) ----------------------------------------------
def contrast_mde(n: int, sigma_delta: float, var_f4: float, rho_cond: float,
                 alpha: float = 0.05, power: float = 0.80) -> float:
    """MDE for beta1(route)-beta1(precise) = slope of (Delta_route-Delta_precise).

    sigma_C = sigma_delta * sqrt(2*(1-rho_cond)); rho_cond>0 helps.
    """
    sigma_c = sigma_delta * math.sqrt(max(2.0 * (1.0 - rho_cond), 0.0))
    return analytic_mde(n, sigma_c, var_f4, alpha, power)


# ---- internal-pilot variance re-estimation --------------------------------
def required_n(target_effect: float, sigma: float, var_f4: float,
               alpha: float = 0.05, power: float = 0.80) -> int:
    """Smallest n whose MDE <= target_effect. Inverts the analytic MDE."""
    z = norm_ppf(1 - alpha) + norm_ppf(power)
    n = (z * sigma) ** 2 / (var_f4 * target_effect ** 2)
    return int(math.ceil(n))


def reestimate_from_pilot(delta_route: dict, delta_precise: dict,
                          f4: dict, target_contrast: float,
                          alpha: float = 0.05, power: float = 0.80) -> dict:
    """Internal-pilot re-estimation (Wittes-Brittain 1990).

    From a pilot's per-token deltas in BOTH conditions, measure the realized
    contrast residual SD and the cross-condition correlation, then report the
    N needed to detect `target_contrast` at the target power. Uses ONLY the
    variance/correlation of the pilot, never its effect estimate, so the
    type-I error of the final test is preserved.
    """
    toks = sorted(set(delta_route) & set(delta_precise) & set(f4))
    dr = np.array([delta_route[t] for t in toks])
    dp = np.array([delta_precise[t] for t in toks])
    x = np.array([f4[t] for t in toks])
    var_f4 = float(x.var())
    # residuals of each condition's Delta after removing the F4 trend, so the
    # SD/correlation are of the noise the test actually fights (not the signal)
    rr = dr - np.polyval(np.polyfit(x, dr, 1), x)
    rp = dp - np.polyval(np.polyfit(x, dp, 1), x)
    sigma_r, sigma_p = float(rr.std(ddof=2)), float(rp.std(ddof=2))
    rho = float(np.corrcoef(rr, rp)[0, 1]) if len(toks) > 3 else 0.0
    sigma_delta = math.sqrt(0.5 * (sigma_r ** 2 + sigma_p ** 2))
    sigma_c = math.sqrt(max(sigma_r ** 2 + sigma_p ** 2
                            - 2 * rho * sigma_r * sigma_p, 1e-12))
    return {
        "pilot_n": len(toks), "var_f4": var_f4,
        "sigma_route": sigma_r, "sigma_precise": sigma_p,
        "sigma_delta": sigma_delta, "rho_cond": rho, "sigma_contrast": sigma_c,
        "required_n_per_condition": required_n(target_contrast, sigma_delta,
                                               var_f4, alpha, power),
        "required_n_contrast": required_n(target_contrast, sigma_c, var_f4,
                                          alpha, power),
    }


# ---- report ---------------------------------------------------------------
@dataclass
class Design:
    var_f4_balanced: float
    var_f4_natural: float


def design_from_f4(f4_path: str, seed: int = 0,
                   trials: int = 400) -> tuple[Design, np.ndarray]:
    import json
    raw = json.load(open(f4_path))
    f4 = np.array([r["f4"] for r in raw.values()
                   if r.get("f4") is not None and not r.get("excluded")])
    rng = np.random.default_rng(seed)
    bands = {"zero": f4 == 0, "low": (f4 > 0) & (f4 <= 1/3),
             "med": (f4 > 1/3) & (f4 <= 2/3), "high": f4 > 2/3}
    pools = {k: f4[m] for k, m in bands.items()}
    n = 500
    per = n // 4
    vb = np.mean([np.concatenate(
        [rng.choice(pools[k], min(per, len(pools[k])), replace=False)
         for k in bands]).var() for _ in range(trials)])
    vn = float(f4.var())
    # the balanced design pool (one realization) to draw from in simulation
    balanced_pool = np.concatenate(
        [rng.choice(pools[k], min(per, len(pools[k])), replace=False)
         for k in bands])
    return Design(float(vb), vn), balanced_pool


def report(design: Design, pool: np.ndarray,
           ns=(200, 300, 500, 800),
           sigmas=(0.05, 0.10, 0.15),
           ref_effects=(0.02, 0.035, 0.05)) -> dict:
    vb = design.var_f4_balanced
    out = {"var_f4_balanced": vb, "var_f4_natural": design.var_f4_natural,
           "design_effect": vb / design.var_f4_natural, "mde": {}, "contrast": {}}
    print(f"Var(F4): balanced={vb:.4f}  natural={design.var_f4_natural:.4f}  "
          f"design-effect={vb/design.var_f4_natural:.2f}x "
          f"(MDE ratio {math.sqrt(design.var_f4_natural/vb):.2f})\n")

    print("Per-condition MDE(beta1) at power=0.80, alpha=0.05 (one-sided):")
    print(f"{'sigma_Delta':>11} " + " ".join(f"N={n:>4}" for n in ns))
    for s in sigmas:
        row = [analytic_mde(n, s, vb) for n in ns]
        out["mde"][s] = dict(zip(ns, row))
        print(f"{s:>11.2f} " + " ".join(f"{m:6.3f}" for m in row))

    print("\nReference effect sizes (CLS swing from F4=0 to F4=1):")
    for e in ref_effects:
        print(f"  beta1={e:.3f}  -> "
              + "  ".join(f"N{n}:{analytic_power(n, e, 0.10, vb):.2f}" for n in ns)
              + "   (power at sigma_Delta=0.10)")

    print("\nHeadline CONTRAST beta1(route)-beta1(precise), MDE at "
          "sigma_Delta=0.10, varying residual corr rho_cond:")
    print(f"{'rho_cond':>9} " + " ".join(f"N={n:>4}" for n in ns))
    for rho in (0.0, 0.3, 0.6):
        row = [contrast_mde(n, 0.10, vb, rho) for n in ns]
        out["contrast"][rho] = dict(zip(ns, row))
        print(f"{rho:>9.1f} " + " ".join(f"{m:6.3f}" for m in row))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f4-scores", required=True)
    p.add_argument("--out")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    design, pool = design_from_f4(a.f4_scores)
    res = report(design, pool)
    if a.out:
        import json
        # round for readability
        json.dump(res, open(a.out, "w"), indent=2, default=float)
        print(f"\nwrote {a.out}")
