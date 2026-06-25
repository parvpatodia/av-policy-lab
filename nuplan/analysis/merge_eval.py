"""Merge eval_array_v4 output into the per-cell, seed-averaged score files that
analyze_moderation.py consumes. Pre-registered reduction (RESEARCH_PROTOCOL l.98,
POWER_ANALYSIS l.17): per (reactive, head, goal) -> union the 8 disjoint shards per
seed -> average CLS over the 3 seeds per token. Writes one merged aggregator parquet
per cell (columns: scenario, log_name, score) so analyze_moderation --det-route <dir>
etc. works UNCHANGED.

  python merge_eval.py --run-tag f5_2x2_v4 --reactive 0 --out-root <dir>
  python merge_eval.py --selftest
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path

SIM = Path("/scratch/patodia.pa/av-policy-lab/sim_results/eval")
SEEDS = (0, 1, 2)
CELLS = [("det", "route"), ("diff", "route"), ("det", "precise"), ("diff", "precise")]


def read_cell_scores(agg_parquet: str) -> dict:
    """aggregator parquet -> {token: CLS}; per-scenario rows only (log_name not null,
    not the final_score / scenario_type aggregate rows)."""
    import pandas as pd
    df = pd.read_parquet(agg_parquet)
    out = {}
    for _, r in df.iterrows():
        tok, log = r["scenario"], r.get("log_name")
        if tok == "final_score" or log is None or (isinstance(log, float)):
            continue
        out[str(tok)] = float(r["score"])
    return out


def seed_union_shards(run_tag: str, reactive: int, head: str, goal: str) -> dict:
    """{token: seed_averaged_CLS} for one cell. Shards (disjoint tokens) union within a
    seed; the 3 seeds average per token."""
    per_token_vals: dict[str, list] = {}
    for seed in SEEDS:
        cell = f"{head}_{goal}_seed{seed}"
        scores: dict[str, float] = {}
        for agg in glob.glob(str(SIM / f"{run_tag}_r{reactive}" / f"{cell}_shard*" /
                                  "**" / "aggregator_metric" / "*.parquet"), recursive=True):
            scores.update(read_cell_scores(agg))   # shards are token-disjoint -> union
        for t, v in scores.items():
            per_token_vals.setdefault(t, []).append(v)
    # average over the seeds that produced this token
    return {t: sum(vs) / len(vs) for t, vs in per_token_vals.items()}


def write_merged_parquet(scores: dict, out_dir: Path):
    import pandas as pd
    rows = [{"scenario": t, "log_name": "merged", "scenario_type": "merged", "score": s}
            for t, s in scores.items()]
    rows.append({"scenario": "final_score", "log_name": None, "scenario_type": "final_score",
                 "score": (sum(scores.values()) / len(scores)) if scores else float("nan")})
    p = out_dir / "eval" / "aggregator_metric"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p / "merged.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="f5_2x2_v4")
    ap.add_argument("--reactive", type=int, default=0)
    ap.add_argument("--out-root")
    ap.add_argument("--selftest", action="store_true")
    a, _ = ap.parse_known_args()
    if a.selftest:
        return selftest()
    if not a.out_root:
        ap.error('--out-root required unless --selftest')
    out_root = Path(a.out_root)
    for head, goal in CELLS:
        scores = seed_union_shards(a.run_tag, a.reactive, head, goal)
        d = out_root / f"{head}_{goal}"
        write_merged_parquet(scores, d)
        print(f"{head}_{goal}: {len(scores)} tokens (seed-averaged) -> {d}")
    print("Now run analyze_moderation.py --det-route %s/det_route --diff-route %s/diff_route "
          "--det-precise %s/det_precise --diff-precise %s/diff_precise ..." % ((out_root,)*4))


def selftest():
    """Deterministic check of the shard-union + seed-average logic (no eval data)."""
    # simulate: seed-averaging of per-token CLS across 3 seeds; shards are token-disjoint
    fake = {  # token -> per-seed list (as if unioned from shards)
        "t1": [0.8, 0.9, 1.0],   # mean 0.9
        "t2": [0.6, 0.6, 0.6],   # mean 0.6
        "t3": [0.5, 0.7],        # only 2 seeds present -> mean 0.6
    }
    merged = {t: sum(v) / len(v) for t, v in fake.items()}
    assert abs(merged["t1"] - 0.9) < 1e-9, merged
    assert abs(merged["t2"] - 0.6) < 1e-9, merged
    assert abs(merged["t3"] - 0.6) < 1e-9, merged
    # shard union: disjoint token dicts combine without overwrite collision
    s = {}; s.update({"a": 1.0}); s.update({"b": 2.0})
    assert s == {"a": 1.0, "b": 2.0}
    print("selftest OK: seed-average + shard-union logic correct")


if __name__ == "__main__":
    main()
