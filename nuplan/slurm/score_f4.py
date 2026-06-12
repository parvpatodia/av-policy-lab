"""Compute per-scenario F4 scores for the f0_v2 dataset (two passes).

Pass "shards": stream all shards once; the FIRST occurrence of each
scenario_token is its minimum iteration (extract_scenario appends iterations
in order), score the shard-side components there (S_inter, G_stop).

Pass "map": rebuild the exact extraction scenario set (same DB chunking,
same per-type filter), then compute the PRIMARY S_branch/S_lane from the
lane graph restricted to the route roadblock corridor, join, combine, and
write f4_scores.json plus the gate-1 per-type median table.

Run on a compute node:
  srun ... python nuplan/slurm/score_f4.py --pass shards
  srun ... python nuplan/slurm/score_f4.py --pass map [--smoke 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))           # nuplan/ dir: features.* imports
sys.path.insert(0, str(REPO / "features"))

from features.f4_score import DenormConfig, combine, denorm_sample, g_stop, s_inter  # noqa: E402
from features.f4_map_branch import LaneInfo, corridor_branching  # noqa: E402

F0_DIR = Path("/scratch/patodia.pa/av-policy-lab/features/f0_v2")
OUT_DIR = Path("/scratch/patodia.pa/av-policy-lab/features/f4")
DATA_ROOT = "/scratch/patodia.pa/nuplan/data/cache/mini"
MAP_ROOT = "/scratch/patodia.pa/nuplan/maps"
NUM_TASKS = 16
PER_TYPE = 20


def pass_shards() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(F0_DIR.glob("task_*/scene_shard_*.pt"))
    assert shards, f"no shards under {F0_DIR}"
    scores: dict = {}
    for n, sp in enumerate(shards):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        dn = DenormConfig.from_shard_config(data["config"])
        for s in data["samples"]:
            tok = s["scenario_token"]
            if tok in scores:
                continue
            d = denorm_sample(s, dn)
            if not d["route_mask"].any():
                row = {"excluded": True, "v0": d["v0"]}
            else:
                row = {"excluded": False, "s_inter": s_inter(d, dn),
                       "g_stop": g_stop(d, dn), "v0": d["v0"]}
            row.update(scenario_type=s["scenario_type"], log_name=s["log_name"],
                       iteration=int(s["iteration"]))
            scores[tok] = row
        if n % 50 == 0:
            print(f"  shard {n}/{len(shards)}  scenarios so far: {len(scores)}",
                  flush=True)
    out = OUT_DIR / "shard_scores.json"
    out.write_text(json.dumps(scores))
    n_exc = sum(1 for r in scores.values() if r["excluded"])
    print(f"shard pass done: {len(scores)} scenarios ({n_exc} excluded) -> {out}")


# ---------- map pass ----------

def _build_token_index(tokens_needed: set):
    """Rebuild extraction's scenario set task-by-task; index by token."""
    sys.path.insert(0, str(REPO / "features"))
    from scene_features import _build_mini_scenarios  # noqa
    all_dbs = sorted(Path(DATA_ROOT).glob("*.db"))
    chunk = math.ceil(len(all_dbs) / NUM_TASKS)
    index = {}
    for t in range(NUM_TASKS):
        logs = [p.stem for p in all_dbs[t * chunk:(t + 1) * chunk]]
        if not logs:
            continue
        scenarios = _build_mini_scenarios(
            DATA_ROOT, MAP_ROOT, 10_000_000,
            log_names=logs, num_scenarios_per_type=PER_TYPE,
        )
        for sc in scenarios:
            if sc.token in tokens_needed:
                index[sc.token] = sc
        print(f"  task {t}: built {len(scenarios)} scenarios, "
              f"matched so far {len(index)}/{len(tokens_needed)}", flush=True)
        if len(index) == len(tokens_needed):
            break
    return index


def _corridor_graph(scenario) -> tuple:
    """LaneInfo graph over the route roadblock corridor (global frame)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    api = scenario.map_api
    rb_ids = [str(r) for r in scenario.get_route_roadblock_ids()]
    graph = {}
    for rb_id in rb_ids:
        rb = api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK)
        if rb is None:
            rb = api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if rb is None:
            continue
        for lane in rb.interior_edges:
            pts = np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])
            graph[str(lane.id)] = LaneInfo(
                id=str(lane.id), rb_id=rb_id, pts=pts,
                succ=[str(e.id) for e in lane.outgoing_edges],
            )
    return graph, rb_ids


def pass_map(smoke: int = 0) -> None:
    shard_path = OUT_DIR / "shard_scores.json"
    assert shard_path.exists(), "run --pass shards first"
    shard_scores = json.loads(shard_path.read_text())
    tokens = {t for t, r in shard_scores.items() if not r["excluded"]}
    if smoke:
        # take smoke tokens from the alphabetically first logs so the first
        # rebuilt DB chunk already contains them (fast smoke, no full rebuild)
        tokens = set(sorted(tokens, key=lambda t: (shard_scores[t]["log_name"], t))[:smoke])
    print(f"map pass over {len(tokens)} scenarios")
    index = _build_token_index(tokens)
    missing = tokens - set(index)
    if missing:
        print(f"WARNING: {len(missing)} tokens not matched by rebuilt set")

    results = {}
    for i, tok in enumerate(sorted(index)):
        sc, row = index[tok], shard_scores[tok]
        try:
            ego = sc.get_ego_state_at_iteration(row["iteration"])
            ego_xy = np.array([ego.rear_axle.x, ego.rear_axle.y])
            graph, rb_ids = _corridor_graph(sc)
            br = corridor_branching(graph, rb_ids, ego_xy, v0=row["v0"])
        except Exception as exc:
            br = {"excluded": True, "error": f"{type(exc).__name__}: {exc}"}
        if br.get("excluded"):
            results[tok] = {**row, **br, "f4": None}
        else:
            f4 = combine(br["s_branch"], br["s_lane"], row["s_inter"], row["g_stop"])
            results[tok] = {**row, **br, "f4": f4}
        if smoke:
            print(f"  {tok[:10]} {row['scenario_type'][:36]:36s} "
                  f"f4={results[tok].get('f4')} {br}", flush=True)
        elif i % 500 == 0:
            print(f"  {i}/{len(index)}", flush=True)

    out = OUT_DIR / ("f4_scores_smoke.json" if smoke else "f4_scores.json")
    out.write_text(json.dumps(results))
    print(f"wrote {out}")

    # gate-1 table: per-type median F4
    by_type = defaultdict(list)
    for r in results.values():
        if r.get("f4") is not None:
            by_type[r["scenario_type"]].append(r["f4"])
    print(f"\n{'scenario_type':42s} {'n':>5s} {'median_f4':>9s}")
    for t in sorted(by_type, key=lambda k: -float(np.median(by_type[k]))):
        print(f"{t:42s} {len(by_type[t]):5d} {np.median(by_type[t]):9.3f}")


def pass_combine() -> None:
    """v1.1 recombine: fresh shard-side scores + map fields already computed
    by an earlier --pass map run (b_r/n_par/s_branch are version-independent).
    F4 v1.1 = G_stop * (1 - (1-S_branch)(1-S_inter)); S_lane is reported as a
    covariate but excluded from the scalar (ADR-016: it saturates on Vegas
    multi-lane roadblocks and floors the score)."""
    shard_scores = json.loads((OUT_DIR / "shard_scores.json").read_text())
    map_scores = json.loads((OUT_DIR / "f4_scores.json").read_text())
    results = {}
    for tok, row in shard_scores.items():
        m = map_scores.get(tok, {})
        # WHY b_r == 0 excludes: zero corridor terminals means ego could not
        # be localized on the route graph — a measurement failure, not
        # evidence of zero ambiguity (was mis-scored as s_branch=0 in v1.0).
        if (row.get("excluded") or m.get("excluded", True)
                or "s_branch" not in m or m.get("b_r", 0) == 0):
            results[tok] = {**row, "f4": None, "excluded": True}
            continue
        f4 = combine(m["s_branch"], 0.0, row["s_inter"], row["g_stop"])
        results[tok] = {**row, "b_r": m["b_r"], "n_par": m["n_par"],
                        "s_branch": m["s_branch"], "s_lane": m["s_lane"],
                        "f4": f4, "f4_version": "1.1"}
    out = OUT_DIR / "f4_scores_v11.json"
    out.write_text(json.dumps(results))
    n_ok = sum(1 for r in results.values() if r.get("f4") is not None)
    print(f"combine done: {n_ok}/{len(results)} scored -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=("shards", "map", "combine"),
                    required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.which == "shards":
        pass_shards()
    elif a.which == "combine":
        pass_combine()
    else:
        pass_map(a.smoke)


if __name__ == "__main__":
    main()
