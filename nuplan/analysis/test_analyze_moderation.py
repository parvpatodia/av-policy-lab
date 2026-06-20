"""Tests for the moderation analyzer. Validate the statistics against known
constructions; no parquet/sim dependency for the math."""
import numpy as np
import pytest

import analyze_moderation as A


def test_ols_hc3_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 2000)
    y = 0.3 + 0.5 * x + rng.normal(0, 0.05, 2000)
    b0, b1, se = A._ols_hc3(x, y)
    assert abs(b0 - 0.3) < 0.02 and abs(b1 - 0.5) < 0.02
    assert 0.0 < se < 0.05


def test_ols_hc3_zero_slope_under_null():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 3000)
    y = rng.normal(0, 0.1, 3000)            # no dependence on x
    _, b1, se = A._ols_hc3(x, y)
    assert abs(b1 / se) < 3.0               # not significant


def test_student_sf_matches_known_values():
    # t=0 -> upper tail 0.5; large positive t -> small p
    assert abs(A._student_sf(0.0, 50) - 0.5) < 1e-6
    assert A._student_sf(5.0, 100) < 0.001
    assert A._student_sf(1.660, 100) == pytest.approx(0.05, abs=0.01)  # ~crit


def test_student_sf_symmetry():
    for dof in (5, 30, 200):
        for t in (0.5, 1.3, 2.7):
            assert A._student_sf(t, dof) + A._student_sf(-t, dof) == pytest.approx(1.0, abs=1e-6)


def test_spearman_monotone():
    x = np.arange(50.0)
    assert A._spearman(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-9)
    assert A._spearman(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_theil_sen_robust_to_outliers():
    x = np.arange(60.0)
    y = 0.4 * x + np.random.default_rng(0).normal(0, 0.5, 60)
    y[5] += 100  # outlier that would wreck OLS
    assert abs(A._theil_sen(x, y) - 0.4) < 0.1


def test_bootstrap_ci_brackets_mean():
    d = np.random.default_rng(0).normal(0.2, 0.1, 500)
    lo, hi = A._bootstrap_mean_ci(d)
    assert lo < 0.2 < hi and (hi - lo) < 0.05


def test_moderation_detects_positive_interaction():
    rng = np.random.default_rng(0)
    toks = [f"t{i}" for i in range(800)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    delta = {t: 0.08 * f4[t] + rng.normal(0, 0.05) for t in toks}
    res = A.moderation(delta, f4, "route")
    assert res.beta1 > 0.05 and res.beta1_p_onesided < 0.01
    assert res.spearman_rho > 0.2


def test_moderation_null_not_significant():
    rng = np.random.default_rng(2)
    toks = [f"t{i}" for i in range(800)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    delta = {t: rng.normal(0, 0.05) for t in toks}     # no F4 dependence
    res = A.moderation(delta, f4, "precise")
    assert res.beta1_p_onesided > 0.05


def test_contrast_equals_difference_of_slopes():
    """The contrast estimator must exactly equal beta1(route)-beta1(precise),
    because OLS slope is linear in y with shared X."""
    rng = np.random.default_rng(0)
    toks = [f"t{i}" for i in range(300)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    dr = {t: 0.06 * f4[t] + rng.normal(0, 0.05) for t in toks}
    dp = {t: 0.01 * f4[t] + rng.normal(0, 0.05) for t in toks}
    br = A.moderation(dr, f4, "route").beta1
    bp = A.moderation(dp, f4, "precise").beta1
    c = A.moderation_contrast(dr, dp, f4)
    assert c.beta1 == pytest.approx(br - bp, abs=1e-9)
    assert c.condition == "route_minus_precise"


def test_contrast_significant_when_route_steeper():
    rng = np.random.default_rng(1)
    toks = [f"t{i}" for i in range(800)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    # shared scenario difficulty -> positively correlated residuals (realistic)
    shared = {t: rng.normal(0, 0.04) for t in toks}
    dr = {t: 0.07 * f4[t] + shared[t] + rng.normal(0, 0.03) for t in toks}
    dp = {t: 0.00 * f4[t] + shared[t] + rng.normal(0, 0.03) for t in toks}
    c = A.moderation_contrast(dr, dp, f4)
    assert c.beta1 > 0.04 and c.beta1_p_onesided < 0.01


def test_contrast_not_significant_when_slopes_equal():
    rng = np.random.default_rng(2)
    toks = [f"t{i}" for i in range(800)]
    f4 = {t: rng.uniform(0, 1) for t in toks}
    dr = {t: 0.04 * f4[t] + rng.normal(0, 0.05) for t in toks}
    dp = {t: 0.04 * f4[t] + rng.normal(0, 0.05) for t in toks}
    c = A.moderation_contrast(dr, dp, f4)
    assert c.beta1_p_onesided > 0.05


def test_read_cell_scores_filters_aggregate_rows(tmp_path):
    import pandas as pd
    rows = [
        {"scenario": "tokA", "log_name": "log1", "score": 0.8},
        {"scenario": "tokB", "log_name": "log2", "score": 0.6},
        {"scenario": "stationary", "log_name": None, "score": 0.7},   # type agg
        {"scenario": "final_score", "log_name": None, "score": 0.7},  # final
    ]
    p = tmp_path / "agg.parquet"
    pd.DataFrame(rows).to_parquet(p)
    scores = A.read_cell_scores(str(p))
    assert set(scores) == {"tokA", "tokB"}
    assert scores["tokA"] == 0.8


def test_paired_delta_intersects_tokens(tmp_path):
    import pandas as pd
    def mk(d, name, rows):
        sub = d / name / "aggregator_metric"
        sub.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(sub / "a.parquet")
        return str(d / name)
    det = mk(tmp_path, "det", [
        {"scenario": "x", "log_name": "l", "score": 0.5},
        {"scenario": "y", "log_name": "l", "score": 0.5},
    ])
    diff = mk(tmp_path, "diff", [
        {"scenario": "x", "log_name": "l", "score": 0.8},
        {"scenario": "z", "log_name": "l", "score": 0.9},
    ])
    delta = A.paired_delta(det, diff)
    assert set(delta) == {"x"} and delta["x"] == pytest.approx(0.3)
