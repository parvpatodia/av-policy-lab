"""Tests for results aggregation: synthetic parquet trees, seed averaging,
moderation wiring."""
import json

import numpy as np
import pandas as pd
import pytest

import results_table as R


def _write_agg(d, rows):
    sub = d / "eval" / "aggregator_metric"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(sub / "a.parquet")


def _rows(tok_scores):
    rows = [{"scenario": t, "log_name": "lg", "score": s} for t, s in tok_scores.items()]
    rows.append({"scenario": "final_score", "log_name": None, "score": 0.0})
    return rows


def test_merge_shards_concatenates(tmp_path):
    _write_agg(tmp_path / "c_shard0", _rows({"a": 0.8, "b": 0.6}))
    _write_agg(tmp_path / "c_shard1", _rows({"c": 0.9}))
    # merge_shards over the parent recurses both shard dirs
    out = {}
    for sd in (tmp_path / "c_shard0", tmp_path / "c_shard1"):
        out.update(R.merge_shards(str(sd)))
    assert out == {"a": 0.8, "b": 0.6, "c": 0.9}


def test_seed_averaging(tmp_path):
    for seed, val in ((0, 0.6), (1, 0.8)):
        d = tmp_path / f"v3_seed{seed}" / "det_route_shard0"
        _write_agg(d, _rows({"a": val, "b": val}))
    sc = R.cell_scores_over_seeds(str(tmp_path), "v3_seed{seed}", [0, 1],
                                  "det", "route")
    assert sc["a"] == pytest.approx(0.7) and sc["b"] == pytest.approx(0.7)


def test_seed_averaging_intersects_tokens(tmp_path):
    _write_agg(tmp_path / "v3_seed0" / "det_route_shard0", _rows({"a": 0.6, "b": 0.6}))
    _write_agg(tmp_path / "v3_seed1" / "det_route_shard0", _rows({"a": 0.8}))  # no b
    sc = R.cell_scores_over_seeds(str(tmp_path), "v3_seed{seed}", [0, 1],
                                  "det", "route")
    assert set(sc) == {"a"} and sc["a"] == pytest.approx(0.7)


def test_summarize_ci_brackets_mean():
    s = R.summarize({f"t{i}": 0.5 + 0.01 * i for i in range(100)})
    assert s["n"] == 100
    lo, hi = s["ci95"]
    assert lo < s["mean"] < hi


def test_summarize_empty():
    s = R.summarize({})
    assert s["n"] == 0 and np.isnan(s["mean"])


def test_run_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    toks = [f"t{i}" for i in range(120)]
    f4 = {t: float(rng.uniform(0, 1)) for t in toks}
    (tmp_path / "f4.json").write_text(
        json.dumps({t: {"f4": v} for t, v in f4.items()}))
    # construct a positive route moderation: diff beats det more at high F4
    for seed in (0, 1):
        base = tmp_path / f"v3_seed{seed}"
        for head, goal in R.CELLS:
            sc = {}
            for t in toks:
                det = 0.7 + rng.normal(0, 0.02)
                if head == "diff" and goal == "route":
                    sc[t] = det + 0.1 * f4[t] + rng.normal(0, 0.02)
                else:
                    sc[t] = det + rng.normal(0, 0.02)
            _write_agg(base / f"{head}_{goal}_shard0", _rows(sc))
    out = R.run(R.parse_args([
        "--runs-root", str(tmp_path), "--run-tag-fmt", "v3_seed{seed}",
        "--seeds", "0,1", "--f4-scores", str(tmp_path / "f4.json"),
        "--out", str(tmp_path / "out.json"),
    ]))
    assert out["route"]["beta1"] > 0.05
    assert out["route"]["beta1_p_onesided"] < 0.05
    # precise condition has no F4 dependence by construction
    assert abs(out["precise"]["beta1"]) < out["route"]["beta1"]
    assert out["route_minus_precise_beta1"] > 0
    assert (tmp_path / "out.json").exists()
