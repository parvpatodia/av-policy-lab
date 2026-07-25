"""ADR-087: re-aggregate EXISTING per-scenario sub-metrics into the CANONICAL nuPlan reactive CLS
(closed_loop_reactive_agents_weighted_average) -- the field-standard PDM score with hard multipliers --
WITHOUT re-running any sim. Fixes the ADR-086 finding that run_cells used default_weighted_average
(multiple_metrics=null, no multipliers -> inflated CLS).

canonical CLS(token) = [ (5*progress + 5*ttc + 4*speed + 2*comfort) / 16 ]
                       * no_ego_at_fault_collisions * drivable_area_compliance
                       * ego_is_making_progress * driving_direction_compliance

Usage:
  canonical_reaggregate.py xcheck <run_dir>      # validate formula vs the run's own canonical aggregator
  canonical_reaggregate.py analyze [permode_exp] [zoo_exp]
"""
import sys, glob, os
import numpy as np
import pandas as pd

SC = "/scratch/patodia.pa/av-policy-lab"
WEIGHTED = {"ego_progress_along_expert_route": 5.0, "time_to_collision_within_bound": 5.0,
            "speed_limit_compliance": 4.0, "ego_is_comfortable": 2.0}
WSUM = sum(WEIGHTED.values())  # 16
MULT = ["no_ego_at_fault_collisions", "drivable_area_compliance",
        "ego_is_making_progress", "driving_direction_compliance"]


def _per_token(metrics_dir, metric):
    """token -> metric_score for one metric parquet (mean if duplicate rows)."""
    p = os.path.join(metrics_dir, metric + ".parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p, columns=None)
    sc = "scenario_name" if "scenario_name" in d.columns else "scenario"
    if "metric_score" not in d.columns:
        return None
    g = d.groupby(sc)["metric_score"].mean()
    return {str(k): float(v) for k, v in g.items()}


def canonical_from_metrics_dir(metrics_dir):
    """token -> canonical CLS for one cell's metrics/ dir. Needs all 8 sub-metrics."""
    cols = {}
    for m in list(WEIGHTED) + MULT:
        t = _per_token(metrics_dir, m)
        if t is None:
            return {}  # incomplete cell
        cols[m] = t
    toks = set(cols[list(WEIGHTED)[0]])
    for m in cols:
        toks &= set(cols[m])
    out = {}
    for t in toks:
        base = sum(WEIGHTED[m] * cols[m][t] for m in WEIGHTED) / WSUM
        mult = 1.0
        for m in MULT:
            mult *= cols[m][t]
        out[t] = base * mult
    return out


def collect_canonical(pattern):
    """merge canonical per-token CLS across shard dirs matching pattern."""
    d = {}
    for cell in sorted(glob.glob(pattern)):
        mdirs = glob.glob(os.path.join(cell, "**", "metrics"), recursive=True)
        if not mdirs:
            continue
        d.update(canonical_from_metrics_dir(sorted(mdirs)[-1]))
    return d


def _run_agg_score(run_dir):
    """token -> the run's OWN final CLS from its aggregator parquet (for cross-check)."""
    fs = glob.glob(os.path.join(run_dir, "**", "aggregator_metric", "*.parquet"), recursive=True)
    if not fs:
        return {}
    d = pd.read_parquet(sorted(fs)[-1])
    sc = "scenario"
    out = {}
    for _, r in d.iterrows():
        s = str(r.get(sc, ""))
        if len(s) == 16 and s.isalnum():   # per-scenario token rows only
            out[s] = float(r["score"])
    return out


def xcheck(run_dir):
    mdirs = glob.glob(os.path.join(run_dir, "**", "metrics"), recursive=True)
    can = canonical_from_metrics_dir(sorted(mdirs)[-1]) if mdirs else {}
    agg = _run_agg_score(run_dir)
    toks = sorted(set(can) & set(agg))
    print(f"xcheck {run_dir}")
    print(f"  tokens: canonical={len(can)} agg={len(agg)} common={len(toks)}")
    if not toks:
        return
    diffs = [abs(can[t] - agg[t]) for t in toks]
    print(f"  max|canonical-agg|={max(diffs):.6f}  mean|diff|={np.mean(diffs):.6f}")
    for t in toks:
        print(f"    {t}: canonical={can[t]:.4f}  run_agg={agg[t]:.4f}  d={can[t]-agg[t]:+.4f}")
    print(f"  canonical final mean={np.mean([can[t] for t in toks]):.4f}  "
          f"agg final mean={np.mean([agg[t] for t in toks]):.4f}")


def analyze(permode_exp, zoo_exp):
    P = f"{SC}/sim_results/eval/{permode_exp}"
    Z = f"{SC}/sim_results/eval/{zoo_exp}"
    modes = {m: collect_canonical(f"{P}/mode{m}_shard*") for m in range(6)}
    print("per-mode token counts (canonical):", {m: len(modes[m]) for m in range(6)})
    common = set(modes[0])
    for m in range(1, 6):
        common &= set(modes[m])
    common = sorted(common)
    print("tokens with all 6 modes:", len(common))
    if not common:
        print("no complete-mode tokens"); return
    oracle = {t: max(modes[m][t] for m in range(6)) for t in common}
    det = collect_canonical(f"{Z}/det_route_shard*")
    wta = collect_canonical(f"{Z}/wta_route_shard*")
    toks = [t for t in common if t in det and t in wta]
    print("tokens with oracle+det+defaultWTA:", len(toks))
    if not toks:
        print("no overlap"); return
    permode_means = {m: float(np.mean([modes[m][t] for t in toks])) for m in range(6)}
    o = np.mean([oracle[t] for t in toks]); dv = np.mean([det[t] for t in toks])
    wv = np.mean([wta[t] for t in toks]); rand = float(np.mean(list(permode_means.values())))
    print("=== CANONICAL CLS (nuPlan reactive, hard multipliers) ===")
    print("per-mode means:", {m: round(v, 4) for m, v in permode_means.items()})
    print(f"oracle {o:.4f}  det {dv:.4f}  default-WTA {wv:.4f}  random {rand:.4f}  (n={len(toks)})")
    print(f"latent value (oracle-det)         {o - dv:+.4f}")
    print(f"default-WTA vs det                {wv - dv:+.4f}")
    print(f"default-WTA vs random (mechanism) {wv - rand:+.4f}")
    print(f"frac scenes some mode beats det:  {np.mean([oracle[t] > det[t] for t in toks]):.3f}")
    print("--- LENIENT baseline (ADR-075): oracle 0.828 det 0.737 latent +0.091; selection ~random ---")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "xcheck":
        xcheck(sys.argv[2])
    else:
        analyze(sys.argv[2] if len(sys.argv) > 2 else "permode_boston_r1",
                sys.argv[3] if len(sys.argv) > 3 else "boston_zoo_r1")
