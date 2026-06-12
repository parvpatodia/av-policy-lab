"""Numerical train/serve feature-parity gate (run on a compute node).

For real mini scenarios, build features twice for the same (scenario,
iteration): (a) the offline path used to make training shards, (b) the
sim-time path through a SimulationHistoryBuffer + PlannerInput, exactly as
PolicyPlanner will see them. Assert every tensor matches.

Any failure here means the closed-loop results would be measuring feature
skew, not policy quality. This gate must PASS before any evaluation run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "features"))

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from features.scene_features import SceneFeatureExtractor, _build_mini_scenarios  # noqa
from serving.policy_planner import features_from_planner_input

DATA_ROOT = "/scratch/patodia.pa/nuplan/data/cache/mini"
MAP_ROOT = "/scratch/patodia.pa/nuplan/maps"
N_SCENARIOS = 3
ITERATIONS = (19, 40)


def planner_input_from_scenario(scenario, iteration: int, n_hist: int):
    """Reconstruct what the simulator would hand the planner at `iteration`."""
    ego_states, observations = [], []
    for it in range(iteration - n_hist + 1, iteration + 1):
        ego_states.append(scenario.get_ego_state_at_iteration(it))
        observations.append(scenario.get_tracked_objects_at_iteration(it))
    buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=n_hist,
        ego_states=ego_states,
        observations=observations,
        sample_interval=scenario.database_interval,
    )
    tl = list(scenario.get_traffic_light_status_at_iteration(iteration))
    sim_iter = SimulationIteration(
        TimePoint(ego_states[-1].time_point.time_us), iteration)
    return PlannerInput(iteration=sim_iter, history=buffer, traffic_light_data=tl)


def main():
    scenarios = _build_mini_scenarios(
        DATA_ROOT, MAP_ROOT, limit=200, num_scenarios_per_type=1,
    )[:N_SCENARIOS]
    assert scenarios, "no scenarios built"
    extractor = SceneFeatureExtractor()
    n_hist = extractor._cfg.history_steps
    n_pass = n_fail = 0
    for sc in scenarios:
        init = PlannerInitialization(
            route_roadblock_ids=list(sc.get_route_roadblock_ids()),
            mission_goal=sc.get_mission_goal(),
            map_api=sc.map_api,
        )
        for it in ITERATIONS:
            offline = extractor.extract_sample(sc, it)
            pin = planner_input_from_scenario(sc, it, n_hist)
            serving, _, _ = features_from_planner_input(extractor, pin, init)
            for key, sv in serving.items():
                ov = offline[key]
                if not np.allclose(np.asarray(ov), np.asarray(sv), atol=1e-5):
                    diff = np.abs(np.asarray(ov, dtype=np.float64)
                                  - np.asarray(sv, dtype=np.float64)).max()
                    print(f"  FAIL {sc.token[:10]} it={it} {key}: max|diff|={diff:.2e}")
                    n_fail += 1
                else:
                    n_pass += 1
            print(f"  {sc.token[:10]} it={it}: checked {len(serving)} tensors")
    print(f"\nPARITY: {n_pass} tensors equal, {n_fail} mismatched")
    if n_fail:
        print("PARITY GATE: FAIL")
        sys.exit(1)
    print("PARITY GATE: PASS")


if __name__ == "__main__":
    main()
