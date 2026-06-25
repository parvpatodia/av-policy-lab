"""Closed-loop evaluation runner for the 2x2 cells + baselines (HPC).

One invocation = one (planner, observation-mode, scenario-set) run. Reuses
the battle-tested Hydra composition from eval_production.py (notably the
prod_eval_metrics merge that keeps BOTH L2 and PDM component parquets).

Pre-registration knobs surfaced as flags, never hardcoded:
  --controller two_stage_controller   (official CLS; perfect_tracking for debug)
  --reactive 0|1                      (CLS-NR box replay vs CLS-R IDM agents)
  --tokens-file frozen manifest       (the eval set is committed before unblinding)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path("/home/patodia.pa/av-policy-lab/nuplan")
sys.path.insert(0, str(REPO))
DEVKIT = Path("/home/patodia.pa/nuplan-devkit")
DB_DIR = "/scratch/patodia.pa/nuplan/data/cache/mini"
SIM_OUT = Path("/scratch/patodia.pa/av-policy-lab/sim_results")
HYDRA_BASE = str(DEVKIT / "nuplan" / "planning" / "script")
REPO_CONFIG_DIR = str(REPO / "config")
TUPLAN_CONFIG_DIR = "/home/patodia.pa/tuplan_garage/tuplan_garage/planning/script/config/simulation"

import hydra  # noqa: E402
from collections import namedtuple  # noqa: E402

_HydraPaths = namedtuple("HydraPaths", "common_dir config_name config_path experiment_dir")


def _simulation_hydra_paths(base: str) -> _HydraPaths:
    """Inlined from tutorials.utils (which needlessly imports bokeh)."""
    return _HydraPaths(
        common_dir="file://" + str(Path(base) / "config" / "common"),
        config_name="default_simulation",
        config_path=str(Path(base) / "config" / "simulation"),
        experiment_dir="file://" + str(Path(base) / "experiments"),
    )


def build_cfg(args, exp_name: str):
    paths = _simulation_hydra_paths(HYDRA_BASE)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=paths.config_path)
    overrides = [
        f"group={SIM_OUT}",
        f"experiment_name={exp_name}",
        "job_name=eval",
        "experiment=${experiment_name}/${job_name}",
        "worker=sequential",
        f"ego_controller={args.controller}",
        f"observation={'idm_agents_observation' if args.reactive else 'box_observation'}",
        "simulation_metric=prod_eval_metrics",
        # WHY drop metric_summary_callback: it exists only to render bokeh
        # histogram PDFs and bokeh 2.4 is incompatible with this env numpy;
        # CLS numbers come from metric_file + aggregator parquets.
        "main_callback=[time_callback,metric_file_callback,metric_aggregator_callback]",
        f"hydra.searchpath=[{paths.common_dir}, {paths.experiment_dir}, file://{REPO_CONFIG_DIR}, file://{TUPLAN_CONFIG_DIR}]",
        "output_dir=${group}/${experiment}",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files={DB_DIR}",
        "scenario_filter=all_scenarios",
    ]
    if args.planner == "policy":
        overrides += [
            "planner=policy_planner",
            f"planner.policy_planner.ckpt_path={args.ckpt}",
            f"planner.policy_planner.head_type={args.head}",
            f"planner.policy_planner.goal_mode={args.goal}",
        ]
    elif args.planner == "idm":
        overrides += ["planner=idm_planner"]
    elif args.planner == "log_future":
        overrides += ["planner=log_future_planner"]
    elif args.planner == "pdm_closed":
        overrides += ["planner=pdm_closed_planner"]
    if args.tokens_file:
        data = json.loads(Path(args.tokens_file).read_text())
        # accept either a bare token list or a freeze payload {"tokens": [...]}
        tokens = data["tokens"] if isinstance(data, dict) else data
        # WHY quote each token: some tokens (e.g. 595322e649225137) look like
        # scientific-notation floats; unquoted, hydra parses them as numbers and rejects them.
        toks = "[" + ",".join(f'"{t}"' for t in tokens) + "]"
        overrides += [f"scenario_filter.scenario_tokens={toks}"]
    else:
        overrides += [
            "scenario_filter.shuffle=true",
            f"scenario_filter.limit_total_scenarios={args.n_scenarios}",
        ]
    return hydra.compose(config_name=paths.config_name, overrides=overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=("policy", "idm", "log_future", "pdm_closed"), required=True)
    ap.add_argument("--head", choices=("det", "diff"), default=None)
    ap.add_argument("--goal", choices=("route", "precise"), default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokens-file", default=None)
    ap.add_argument("--n-scenarios", type=int, default=2)
    ap.add_argument("--reactive", type=int, default=0)
    ap.add_argument("--controller", default="two_stage_controller")
    ap.add_argument("--exp-name", required=True)
    args = ap.parse_args()
    if args.planner == "policy":
        assert args.head and args.goal and args.ckpt, "policy needs head/goal/ckpt"

    exp_dir = (SIM_OUT / args.exp_name).resolve()
    if exp_dir.exists() and SIM_OUT.resolve() in exp_dir.parents:
        shutil.rmtree(exp_dir)  # stale parquets must never mix runs

    cfg = build_cfg(args, args.exp_name)
    from nuplan.planning.script.run_simulation import run_simulation as main_sim
    print(f">>> {args.exp_name}: planner={args.planner} reactive={args.reactive} "
          f"controller={args.controller}", flush=True)
    main_sim(cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    pq = list(exp_dir.rglob("*.parquet"))
    print(f"done: {len(pq)} metric parquets under {exp_dir}")
    for q in pq[:8]:
        print("  ", q.relative_to(exp_dir))


if __name__ == "__main__":
    main()
