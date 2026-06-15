"""Tests for the moderation power analysis.

The central test is the VERIFICATION GATE: the analytic Gaussian power formula
must agree with a Monte-Carlo simulation that runs the actual production
estimator (analyze_moderation). If these diverge, one of them is wrong.
"""
import math

import numpy as np
import pytest

import power_analysis as P


def test_norm_ppf_known_values():
    assert P.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert P.norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-4)
    assert P.norm_ppf(0.80) == pytest.approx(0.841621, abs=1e-4)


def test_norm_cdf_ppf_roundtrip():
    for p in (0.05, 0.2, 0.5, 0.8, 0.99):
        assert P.norm_cdf(P.norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_mde_decreases_with_n_and_increases_with_sigma():
    assert P.analytic_mde(800, 0.1, 0.12) < P.analytic_mde(200, 0.1, 0.12)
    assert P.analytic_mde(500, 0.2, 0.12) > P.analytic_mde(500, 0.1, 0.12)


def test_mde_is_the_effect_at_target_power():
    # by construction, the true effect == MDE should give ~the target power
    mde = P.analytic_mde(500, 0.10, 0.12, power=0.80)
    assert P.analytic_power(500, mde, 0.10, 0.12) == pytest.approx(0.80, abs=1e-3)


def test_verification_gate_analytic_matches_simulation():
    """Gaussian-residual simulation through the REAL estimator must reproduce
    the analytic power within Monte-Carlo error."""
    pool = np.random.default_rng(0).uniform(0, 1, 5000)
    var_f4 = float(pool.var())
    for n, beta1, sigma in ((400, 0.04, 0.10), (600, 0.03, 0.12)):
        a = P.analytic_power(n, beta1, sigma, var_f4)
        s = P.simulate_power(n, beta1, sigma, pool, n_sims=4000, seed=1)
        # MC SE of a proportion ~ sqrt(p(1-p)/4000) <= 0.008; allow 3.5 SE
        assert abs(a - s) < 0.03, f"analytic {a:.3f} vs sim {s:.3f}"


def test_simulation_holds_alpha_under_null():
    """At beta1=0 the rejection rate must sit at alpha (estimator not anti-conservative)."""
    pool = np.random.default_rng(2).uniform(0, 1, 4000)
    s = P.simulate_power(400, 0.0, 0.10, pool, alpha=0.05, n_sims=6000, seed=3)
    assert s < 0.075, f"null rejection {s:.3f} exceeds alpha tolerance"


def test_bounded_cls_attenuates_power_vs_gaussian():
    """Clipping CLS at 1 attenuates the measured slope, so bounded power should
    be <= the unbounded Gaussian power for the same nominal effect."""
    pool = np.random.default_rng(4).uniform(0, 1, 5000)
    g = P.simulate_power(500, 0.06, 0.10, pool, n_sims=3000, bounded=False, seed=5)
    b = P.simulate_power(500, 0.06, 0.10, pool, n_sims=3000, bounded=True, seed=5)
    assert b <= g + 0.02          # bounded never meaningfully easier
    assert b > 0.3                # but still has real power (sanity)


def test_contrast_mde_improves_with_residual_correlation():
    """rho_cond>0 (same scenario hard for both conditions) shrinks the contrast
    residual and improves the MDE; independence is the worst case."""
    indep = P.contrast_mde(500, 0.10, 0.12, rho_cond=0.0)
    corr = P.contrast_mde(500, 0.10, 0.12, rho_cond=0.6)
    assert corr < indep
    # at rho=0 the contrast residual is sqrt(2)x a single condition -> MDE sqrt(2)x
    single = P.analytic_mde(500, 0.10, 0.12)
    assert indep == pytest.approx(single * math.sqrt(2), rel=1e-6)


def test_simulate_mde_recovers_analytic_mde_gaussian():
    pool = np.random.default_rng(6).uniform(0, 1, 5000)
    var_f4 = float(pool.var())
    a = P.analytic_mde(500, 0.10, var_f4)
    s = P.simulate_mde(500, 0.10, pool, n_sims=1500, seed=7)
    assert abs(a - s) < 0.012, f"analytic MDE {a:.3f} vs sim MDE {s:.3f}"


def test_required_n_inverts_mde():
    # the n returned must have MDE just under the target effect
    for eff in (0.02, 0.035, 0.05):
        n = P.required_n(eff, 0.10, 0.12)
        assert P.analytic_mde(n, 0.10, 0.12) <= eff
        assert P.analytic_mde(n - 1, 0.10, 0.12) > eff


def test_reestimate_from_pilot_recovers_known_variance():
    """Construct a pilot with known per-condition residual SD and cross-condition
    correlation; re-estimation must recover them and size N sensibly."""
    rng = np.random.default_rng(0)
    toks = [f"t{i}" for i in range(400)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    sigma, rho = 0.08, 0.5
    shared = {t: rng.normal(0, sigma * math.sqrt(rho)) for t in toks}
    delta_route = {t: 0.05 * f4[t] + shared[t]
                   + rng.normal(0, sigma * math.sqrt(1 - rho)) for t in toks}
    delta_precise = {t: 0.00 * f4[t] + shared[t]
                     + rng.normal(0, sigma * math.sqrt(1 - rho)) for t in toks}
    r = P.reestimate_from_pilot(delta_route, delta_precise, f4, 0.035)
    assert r["sigma_route"] == pytest.approx(sigma, abs=0.015)
    assert r["rho_cond"] == pytest.approx(rho, abs=0.12)
    # correlated residuals -> contrast needs fewer subjects than the rho=0 bound
    n0 = P.required_n(0.035, r["sigma_delta"] * math.sqrt(2), r["var_f4"])
    assert r["required_n_contrast"] < n0
