"""Re-merge eval_array_v4 output KEEPING all PDM sub-components + the REAL
scenario_type. merge_eval.py kept only the composite 'score' and overwrote
scenario_type='merged'; the sub-component + fixed-effects moderation re-analysis
needs both. Same pre-registered reduction (RESEARCH_PROTOCOL l.98): shard-union
per seed (disjoint tokens), then average across the 3 seeds per token.
"""
import argparse, glob
from pathlib import Path
import pandas as pd

SIM = Path("/scratch/patodia.pa/av-policy-lab/sim_results/eval")
SEEDS = (0, 1, 2)
CELLS = [("det", "route"), ("diff", "route"), ("det", "precise"), ("diff", "precise")]
NUMCOLS = [
    "score", "ego_expert_L2_error", "ego_progress_along_expert_route",
    "time_to_collision_within_bound", "no_ego_at_fault_collisions",
    "drivable_area_compliance", "ego_is_comfortable", "ego_is_making_progress",
    "driving_direction_compliance", "speed_limit_compliance",
    "ego_jerk", "ego_lon_jerk", "ego_lat_acceleration",
]


def read_rows(agg):
    df = pd.read_parquet(agg)
    out = {}
    for _, r in df.iterrows():
        tok, log = r["scenario"], r.get("log_name")
        if tok == "final_score" or log is None or (isinstance(log, float)):
            continue
        out[str(tok)] = r
    return out


def merge_cell(run_tag, reactive, head, goal):
    acc, styp = {}, {}
    for seed in SEEDS:
        cell = f"{head}_{goal}_seed{seed}"
        rows = {}
        for agg in glob.glob(str(SIM / f"{run_tag}_r{reactive}" / f"{cell}_shard*" /
                                  "**" / "aggregator_metric" / "*.parquet"), recursive=True):
            rows.update(read_rows(agg))     # shards token-disjoint -> union
        for t, r in rows.items():
            acc.setdefault(t, []).append(r)
            styp[t] = r.get("scenario_type", "?")
    merged = []
    for t, rs in acc.items():
        row = {"scenario": t, "log_name": "merged", "scenario_type": styp[t]}
        for c in NUMCOLS:
            vals = [float(r[c]) for r in rs if (c in r and pd.notna(r[c]))]
            row[c] = (sum(vals) / len(vals)) if vals else float("nan")
        row["n_seeds"] = len(rs)
        merged.append(row)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="f5_2x2_v4")
    ap.add_argument("--reactive", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    for head, goal in CELLS:
        rows = merge_cell(a.run_tag, a.reactive, head, goal)
        outdir = Path(a.out_root) / f"{head}_{goal}" / "eval" / "aggregator_metric"
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(outdir / "merged.parquet")
        ns = [r["n_seeds"] for r in rows]
        print(f"{head}_{goal}: {len(rows)} tokens, seeds/token min={min(ns)} max={max(ns)}")


if __name__ == "__main__":
    main()
