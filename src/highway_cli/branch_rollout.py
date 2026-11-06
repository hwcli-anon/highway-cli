from __future__ import annotations

import copy
import math
import random
from collections import Counter
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .actions import ActionOption


def _round(value: Any, digits: int = 6) -> Any:
    return None if value is None else round(float(value), digits)


def _traffic_metrics(av: Any, target: Any) -> Dict[str, Any]:
    dx = float(target.position[0] - av.position[0])
    dy = float(target.position[1] - av.position[1])
    relative_speed = float(target.velocity - av.velocity)
    same_lane = int(target.lane_index[2]) == int(av.lane_index[2])
    closing_speed = 0.0
    if dx > 0.0:
        closing_speed = -relative_speed
    elif dx < 0.0:
        closing_speed = relative_speed
    ttc = abs(dx) / closing_speed if same_lane and closing_speed > 0.0 else None
    return {
        "center_distance_m": math.hypot(dx, dy),
        "abs_longitudinal_gap_m": abs(dx),
        "ttc_s": ttc,
        "same_lane": same_lane,
    }


def _capture_environment_rngs(env: Any) -> List[Any]:
    """Capture simulator-owned RNGs in addition to process-global RNGs."""

    captured = []
    seen = set()
    for owner in (env, getattr(env, "road", None)):
        rng = getattr(owner, "np_random", None)
        if rng is None or id(rng) in seen:
            continue
        seen.add(id(rng))
        if hasattr(rng, "get_state"):
            captured.append((rng, "legacy", rng.get_state()))
        elif hasattr(rng, "bit_generator"):
            captured.append((rng, "generator", copy.deepcopy(rng.bit_generator.state)))
    return captured


def _restore_environment_rngs(captured: Sequence[Any]) -> None:
    for rng, kind, state in captured:
        if kind == "legacy":
            rng.set_state(state)
        else:
            rng.bit_generator.state = state


def rank_rollout_outcomes(outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank actions by observed safety criticality without inventing a scalar reward."""

    def key(item: Mapping[str, Any]) -> Any:
        controlled_collision = bool(item["av_crash"] and item["controlled_bv_crash"])
        min_ttc = item.get("min_ttc_s")
        return (
            0 if controlled_collision else 1,
            0 if item["av_crash"] else 1,
            float(item["min_center_distance_m"]),
            float(min_ttc) if min_ttc is not None else float("inf"),
            int(item["action_index"]),
        )

    ranked = sorted(outcomes, key=key)
    for rank, item in enumerate(ranked, start=1):
        item["criticality_rank"] = rank
        item["is_best_action"] = rank == 1
    return sorted(ranked, key=lambda item: int(item["action_index"]))


def evaluate_one_step_branches(
    env: Any,
    *,
    action_type: Any,
    cav_action: Any,
    candidate_index: int,
    action_indices: Sequence[int],
    action_options: Mapping[int, ActionOption],
    numpy_module: Any,
    torch_module: Any,
    rng_seed: Any = None,
) -> List[Dict[str, Any]]:
    """Evaluate actions from one state using common-random-number branches.

    Each branch advances one NADE policy interval. The legacy simulator uses
    process-global NumPy, Python and Torch RNGs, so every branch starts from
    identical RNG states and the caller's states are restored at the end.
    """

    numpy_state = numpy_module.random.get_state()
    python_state = random.getstate()
    torch_state = torch_module.get_rng_state()
    environment_rng_states = _capture_environment_rngs(env)
    target_road_index = env.road.vehicles.index(env.controlled_bvs[candidate_index])
    outcomes: List[Dict[str, Any]] = []
    try:
        for action_index in action_indices:
            if rng_seed is None:
                numpy_module.random.set_state(numpy_state)
                random.setstate(python_state)
                torch_module.set_rng_state(torch_state)
            else:
                seed = int(rng_seed)
                numpy_module.random.seed(seed)
                random.seed(seed)
                torch_module.manual_seed(seed)
            branch = copy.deepcopy(env)
            branch_target = branch.road.vehicles[target_road_index]
            samples = [_traffic_metrics(branch.vehicle, branch_target)]
            original_road_step = branch.road.step

            def monitored_road_step(delta_time: float) -> None:
                original_road_step(delta_time)
                samples.append(_traffic_metrics(branch.vehicle, branch_target))

            branch.road.step = monitored_road_step
            bv_actions: List[Any] = [None] * len(branch.controlled_bvs)
            bv_actions[candidate_index] = int(action_index)
            _, terminated, info, _ = branch.step(
                action_type(cav_action=cav_action, bv_action=bv_actions)
            )
            option = action_options[int(action_index)]
            ttc_values = [sample["ttc_s"] for sample in samples if sample["ttc_s"] is not None]
            final = samples[-1]
            outcomes.append(
                {
                    "action_index": int(action_index),
                    "action": option.value,
                    "kind": option.kind,
                    "acceleration_mps2": option.acceleration_mps2,
                    "terminated": bool(terminated),
                    "scene_type": info.get("scene_type"),
                    "av_crash": bool(branch.vehicle.crashed),
                    "controlled_bv_crash": bool(branch_target.crashed),
                    "min_center_distance_m": _round(
                        min(sample["center_distance_m"] for sample in samples)
                    ),
                    "min_abs_longitudinal_gap_m": _round(
                        min(sample["abs_longitudinal_gap_m"] for sample in samples)
                    ),
                    "min_ttc_s": _round(min(ttc_values)) if ttc_values else None,
                    "final_relative_longitudinal_m": _round(
                        branch_target.position[0] - branch.vehicle.position[0]
                    ),
                    "final_center_distance_m": _round(final["center_distance_m"]),
                    "final_av_lane": int(branch.vehicle.lane_index[2]),
                    "final_controlled_bv_lane": int(branch_target.lane_index[2]),
                    "simulator_substeps_observed": len(samples) - 1,
                }
            )
    finally:
        numpy_module.random.set_state(numpy_state)
        random.setstate(python_state)
        torch_module.set_rng_state(torch_state)
        _restore_environment_rngs(environment_rng_states)
    return rank_rollout_outcomes(outcomes)


def evaluate_horizon_branches(
    env: Any,
    *,
    action_type: Any,
    cav_agent: Any,
    select_controlled_action: Any,
    cav_action: Any,
    candidate_index: int,
    action_indices: Sequence[int],
    action_options: Mapping[int, ActionOption],
    numpy_module: Any,
    torch_module: Any,
    policy_steps: int,
    rng_seed: Any = None,
) -> List[Dict[str, Any]]:
    """Evaluate an initial BV action under a fixed closed-loop continuation.

    The candidate action is forced only for the first policy interval. For the
    remaining intervals, the public NADE controller selects BV actions and the
    original AV agent selects CAV actions. All candidate branches use common
    random numbers, and the caller's process-global RNG states are restored.
    """

    if policy_steps < 1:
        raise ValueError("policy_steps must be positive")
    numpy_state = numpy_module.random.get_state()
    python_state = random.getstate()
    torch_state = torch_module.get_rng_state()
    environment_rng_states = _capture_environment_rngs(env)
    target_road_index = env.road.vehicles.index(env.controlled_bvs[candidate_index])
    outcomes: List[Dict[str, Any]] = []
    try:
        for action_index in action_indices:
            if rng_seed is None:
                numpy_module.random.set_state(numpy_state)
                random.setstate(python_state)
                torch_module.set_rng_state(torch_state)
            else:
                seed = int(rng_seed)
                numpy_module.random.seed(seed)
                random.seed(seed)
                torch_module.manual_seed(seed)

            branch = copy.deepcopy(env)
            branch_target = branch.road.vehicles[target_road_index]
            branch_agent = copy.copy(cav_agent)
            branch_agent.env = branch
            samples = [_traffic_metrics(branch.vehicle, branch_target)]
            original_road_step = branch.road.step

            def monitored_road_step(delta_time: float) -> None:
                original_road_step(delta_time)
                samples.append(_traffic_metrics(branch.vehicle, branch_target))

            branch.road.step = monitored_road_step
            initial_bv_actions: List[Any] = [None] * len(branch.controlled_bvs)
            initial_bv_actions[candidate_index] = int(action_index)
            observation_and_indicator, terminated, info, _ = branch.step(
                action_type(cav_action=cav_action, bv_action=initial_bv_actions)
            )
            steps_executed = 1

            while not terminated and steps_executed < policy_steps:
                observation, action_indicator = observation_and_indicator
                continuation = select_controlled_action(
                    observation.bv_observation, branch
                )
                continuation_bv_actions = continuation[0]
                indicator = branch_agent.lane_conflict_safety_check(
                    observation.cav_observation,
                    action_indicator.cav_indicator,
                )
                continuation_cav_action = branch_agent.decision(
                    observation.cav_observation, indicator
                )
                observation_and_indicator, terminated, info, _ = branch.step(
                    action_type(
                        cav_action=continuation_cav_action,
                        bv_action=continuation_bv_actions,
                    )
                )
                steps_executed += 1

            option = action_options[int(action_index)]
            ttc_values = [
                sample["ttc_s"] for sample in samples if sample["ttc_s"] is not None
            ]
            final = samples[-1]
            outcomes.append(
                {
                    "action_index": int(action_index),
                    "action": option.value,
                    "kind": option.kind,
                    "acceleration_mps2": option.acceleration_mps2,
                    "terminated": bool(terminated),
                    "scene_type": info.get("scene_type"),
                    "av_crash": bool(branch.vehicle.crashed),
                    "controlled_bv_crash": bool(branch_target.crashed),
                    "min_center_distance_m": _round(
                        min(sample["center_distance_m"] for sample in samples)
                    ),
                    "min_abs_longitudinal_gap_m": _round(
                        min(sample["abs_longitudinal_gap_m"] for sample in samples)
                    ),
                    "min_ttc_s": _round(min(ttc_values)) if ttc_values else None,
                    "final_relative_longitudinal_m": _round(
                        branch_target.position[0] - branch.vehicle.position[0]
                    ),
                    "final_center_distance_m": _round(final["center_distance_m"]),
                    "final_av_lane": int(branch.vehicle.lane_index[2]),
                    "final_controlled_bv_lane": int(branch_target.lane_index[2]),
                    "policy_steps_requested": int(policy_steps),
                    "policy_steps_executed": steps_executed,
                    "simulator_substeps_observed": len(samples) - 1,
                }
            )
    finally:
        numpy_module.random.set_state(numpy_state)
        random.setstate(python_state)
        torch_module.set_rng_state(torch_state)
        _restore_environment_rngs(environment_rng_states)
    return rank_rollout_outcomes(outcomes)


def _spearman_rank_correlation(first: Mapping[int, int], second: Mapping[int, int]) -> float:
    actions = sorted(first)
    count = len(actions)
    if count < 2:
        return 1.0
    squared_difference = sum(
        (int(first[action]) - int(second[action])) ** 2 for action in actions
    )
    return 1.0 - (6.0 * squared_difference) / (count * (count**2 - 1))


def compare_rollout_rankings(
    short_horizon: Sequence[Mapping[str, Any]],
    long_horizon: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compare action orderings produced by two rollout horizons."""

    short_by_index = {int(item["action_index"]): item for item in short_horizon}
    long_by_index = {int(item["action_index"]): item for item in long_horizon}
    if set(short_by_index) != set(long_by_index):
        raise ValueError("Rollout horizons must contain identical action sets")
    short_ranks = {
        index: int(item["criticality_rank"]) for index, item in short_by_index.items()
    }
    long_ranks = {
        index: int(item["criticality_rank"]) for index, item in long_by_index.items()
    }
    short_best = min(short_by_index.values(), key=lambda item: item["criticality_rank"])
    long_best = min(long_by_index.values(), key=lambda item: item["criticality_rank"])
    top_k = min(3, len(short_by_index))
    short_top = {
        index for index, rank in short_ranks.items() if rank <= top_k
    }
    long_top = {index for index, rank in long_ranks.items() if rank <= top_k}
    return {
        "action_count": len(short_by_index),
        "short_horizon_best_action": short_best["action"],
        "long_horizon_best_action": long_best["action"],
        "top1_match": short_best["action_index"] == long_best["action_index"],
        "short_best_long_horizon_rank": long_ranks[int(short_best["action_index"])],
        "long_best_short_horizon_rank": short_ranks[int(long_best["action_index"])],
        "top3_overlap_count": len(short_top & long_top),
        "top3_jaccard": len(short_top & long_top) / len(short_top | long_top),
        "spearman_rank_correlation": _spearman_rank_correlation(
            short_ranks, long_ranks
        ),
    }


def aggregate_repeated_rollouts(
    repetitions: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    if not repetitions:
        raise ValueError("At least one rollout repetition is required")
    by_action: Dict[int, List[Mapping[str, Any]]] = {}
    rank_maps = []
    top_actions = []
    for repetition in repetitions:
        rank_map = {}
        for item in repetition:
            action_index = int(item["action_index"])
            by_action.setdefault(action_index, []).append(item)
            rank_map[action_index] = int(item["criticality_rank"])
            if int(item["criticality_rank"]) == 1:
                top_actions.append(str(item["action"]))
        rank_maps.append(rank_map)

    aggregated = []
    for action_index, samples in sorted(by_action.items()):
        first = samples[0]
        ranks = [int(sample["criticality_rank"]) for sample in samples]
        distances = [float(sample["min_center_distance_m"]) for sample in samples]
        longitudinal = [
            float(sample["min_abs_longitudinal_gap_m"]) for sample in samples
        ]
        ttc_values = [
            float(sample["min_ttc_s"])
            for sample in samples
            if sample.get("min_ttc_s") is not None
        ]
        controlled_collisions = [
            bool(sample["av_crash"] and sample["controlled_bv_crash"])
            for sample in samples
        ]
        av_collisions = [bool(sample["av_crash"]) for sample in samples]
        aggregated.append(
            {
                "action_index": action_index,
                "action": first["action"],
                "kind": first["kind"],
                "acceleration_mps2": first.get("acceleration_mps2"),
                "rollout_repetitions": len(samples),
                "controlled_collision_rate": mean(controlled_collisions),
                "av_collision_rate": mean(av_collisions),
                "mean_min_center_distance_m": _round(mean(distances)),
                "std_min_center_distance_m": _round(pstdev(distances)),
                "min_center_distance_m": _round(mean(distances)),
                "mean_min_abs_longitudinal_gap_m": _round(mean(longitudinal)),
                "min_abs_longitudinal_gap_m": _round(mean(longitudinal)),
                "mean_min_ttc_s": _round(mean(ttc_values)) if ttc_values else None,
                "min_ttc_s": _round(mean(ttc_values)) if ttc_values else None,
                "mean_repetition_rank": _round(mean(ranks)),
                "std_repetition_rank": _round(pstdev(ranks)),
                "repetition_ranks": ranks,
                "av_crash": any(av_collisions),
                "controlled_bv_crash": any(controlled_collisions),
            }
        )

    aggregated.sort(
        key=lambda item: (
            -float(item["controlled_collision_rate"]),
            -float(item["av_collision_rate"]),
            float(item["mean_min_center_distance_m"]),
            float(item["mean_min_ttc_s"])
            if item["mean_min_ttc_s"] is not None
            else float("inf"),
            int(item["action_index"]),
        )
    )
    for rank, item in enumerate(aggregated, start=1):
        item["criticality_rank"] = rank
        item["is_best_action"] = rank == 1

    correlations = []
    for first_index in range(len(rank_maps)):
        for second_index in range(first_index + 1, len(rank_maps)):
            correlations.append(
                _spearman_rank_correlation(
                    rank_maps[first_index], rank_maps[second_index]
                )
            )
    top_counts = Counter(top_actions)
    consensus_action, consensus_count = top_counts.most_common(1)[0]
    return {
        "action_outcomes": sorted(aggregated, key=lambda item: item["action_index"]),
        "stability": {
            "rollout_repetitions": len(repetitions),
            "top_action_counts": dict(sorted(top_counts.items())),
            "consensus_top_action": consensus_action,
            "consensus_top_action_rate": consensus_count / len(repetitions),
            "aggregate_best_action": aggregated[0]["action"],
            "aggregate_best_matches_consensus": aggregated[0]["action"]
            == consensus_action,
            "mean_pairwise_spearman": mean(correlations) if correlations else 1.0,
            "min_pairwise_spearman": min(correlations) if correlations else 1.0,
        },
    }


def evaluate_repeated_one_step_branches(
    env: Any,
    *,
    action_type: Any,
    cav_action: Any,
    candidate_index: int,
    action_indices: Sequence[int],
    action_options: Mapping[int, ActionOption],
    numpy_module: Any,
    torch_module: Any,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    repetitions = [
        evaluate_one_step_branches(
            env,
            action_type=action_type,
            cav_action=cav_action,
            candidate_index=candidate_index,
            action_indices=action_indices,
            action_options=action_options,
            numpy_module=numpy_module,
            torch_module=torch_module,
            rng_seed=seed,
        )
        for seed in seeds
    ]
    result = aggregate_repeated_rollouts(repetitions)
    result["repetition_outcomes"] = repetitions
    return result


def evaluate_repeated_horizon_branches(
    env: Any,
    *,
    action_type: Any,
    cav_agent: Any,
    select_controlled_action: Any,
    cav_action: Any,
    candidate_index: int,
    action_indices: Sequence[int],
    action_options: Mapping[int, ActionOption],
    numpy_module: Any,
    torch_module: Any,
    policy_steps: int,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    repetitions = [
        evaluate_horizon_branches(
            env,
            action_type=action_type,
            cav_agent=cav_agent,
            select_controlled_action=select_controlled_action,
            cav_action=cav_action,
            candidate_index=candidate_index,
            action_indices=action_indices,
            action_options=action_options,
            numpy_module=numpy_module,
            torch_module=torch_module,
            policy_steps=policy_steps,
            rng_seed=seed,
        )
        for seed in seeds
    ]
    result = aggregate_repeated_rollouts(repetitions)
    result["repetition_outcomes"] = repetitions
    return result
