from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Optional, Set, Tuple

from .actions import nade_action_space
from .background_replay import BackgroundActionReplay
from .branch_rollout import (
    compare_rollout_rankings,
    evaluate_horizon_branches,
    evaluate_one_step_branches,
    evaluate_repeated_horizon_branches,
    evaluate_repeated_one_step_branches,
)
from .cav_policies import build_cav_agent, validate_cav_policy
from .demo import adversarial_specification
from .nade_adapter import audit_nade_native_slots, encode_nade_state
from .proposal import importance_ratio, mixed_proposal
from .providers.base import DecisionProvider
from .representations import (
    FULL_STATE_V1,
    NADE_NATIVE_V1,
    REPRESENTATION_PROFILES,
    validate_profile,
)
from .schema import DecisionResult, DecisionSpecification
from .tracing import append_trace
from .trajectory_trace import (
    capture_environment,
    ensure_vehicle_ids,
    open_trajectory,
    write_trajectory_step,
)


def run_public_nade(
    repo: Path,
    *,
    episodes: int = 1,
    seed: int = 0,
    max_steps: int = 5000,
    cav_policy: str = "av2",
    background_policy: str = "nade",
    state_output: Optional[Path] = None,
    provider: Optional[DecisionProvider] = None,
    decision_trace: Optional[Path] = None,
    trajectory_output: Optional[Path] = None,
    background_action_replay: Optional[Path] = None,
    ndd_mix: float = 0.1,
    representation_profile: str = FULL_STATE_V1,
    benchmark_output: Optional[Path] = None,
    decision_state_output: Optional[Path] = None,
    candidate_state_output: Optional[Path] = None,
    benchmark_selection_keys: Optional[Set[Tuple[str, int, str]]] = None,
    max_benchmark_states: int = 0,
    rollout_repetitions: int = 1,
    validation_horizon_steps: int = 1,
) -> Dict[str, Any]:
    """Run the public NADE/NDE loop without changing its config file."""

    repo = Path(os.path.abspath(os.fspath(repo)))
    if state_output is not None:
        state_output = Path(os.path.abspath(os.fspath(state_output)))
    if decision_trace is not None:
        decision_trace = Path(os.path.abspath(os.fspath(decision_trace)))
    if trajectory_output is not None:
        trajectory_output = Path(os.path.abspath(os.fspath(trajectory_output)))
    replayed_background = None
    if background_action_replay is not None:
        background_action_replay = Path(
            os.path.abspath(os.fspath(background_action_replay))
        )
        replayed_background = BackgroundActionReplay.from_path(
            background_action_replay
        )
    if benchmark_output is not None:
        benchmark_output = Path(os.path.abspath(os.fspath(benchmark_output)))
    if decision_state_output is not None:
        decision_state_output = Path(
            os.path.abspath(os.fspath(decision_state_output))
        )
    if candidate_state_output is not None:
        candidate_state_output = Path(
            os.path.abspath(os.fspath(candidate_state_output))
        )
    if not (repo / "NADE_main.py").is_file():
        raise FileNotFoundError(f"Not a public NADE checkout: {repo}")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if background_policy not in {"nade", "nde"}:
        raise ValueError("background_policy must be 'nade' or 'nde'")
    cav_policy = validate_cav_policy(cav_policy)
    if background_policy == "nde" and provider is not None:
        raise ValueError("NDE uses naturalistic background behavior and no provider")
    if replayed_background is not None and provider is not None:
        raise ValueError("Fixed BV action replay cannot be combined with a provider")
    representation_profile = validate_profile(representation_profile)
    if benchmark_output is not None and max_benchmark_states < 1:
        raise ValueError("max_benchmark_states must be positive when exporting a benchmark")
    if rollout_repetitions < 1:
        raise ValueError("rollout_repetitions must be positive")
    if validation_horizon_steps < 1:
        raise ValueError("validation_horizon_steps must be positive")

    previous_cwd = Path.cwd()
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    try:
        import numpy as np
        import torch
        import global_val
        from CAV_agent.agent import AV_RL_agent
        from NADE_core import (
            _get_Surrogate_CAV_action_probability,
            get_NDD_possi,
            select_controlled_bv_and_action,
        )
        from highway_env.envs.highway_env_NDD import Action, HighwayEnvNDD

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Keep provider sampling independent from the legacy simulator's global
        # RNG so API/model bookkeeping cannot perturb unrelated traffic draws.
        proposal_rng = np.random.RandomState(seed + 104729)
        env_config = {
            "min_distance": 0,
            "max_distance": 1200,
            "min_velocity": 20,
            "max_velocity": 40,
            "min_lane": 0,
            "max_lane": 2,
            "cav_observation_range": global_val.cav_obs_range,
            "bv_observation_range": global_val.bv_obs_range,
            "candidate_controlled_bv_num": 8,
            "cav_observation_num": 10,
            "bv_observation_num": 10,
            "generate_vehicle_mode": "NDD",
            "delete_BV_position": 1200,
            "CAV_model": "AV2",
            "policy_frequency": 1,
        }
        env = HighwayEnvNDD(env_config)
        cav_agent = build_cav_agent(
            cav_policy,
            env,
            actions=global_val.ACTIONS,
            av2_agent_type=AV_RL_agent,
        )
        counts = {"AV-Finish-Test": 0, "AV-Crash": 0}
        episode_weights = []
        weighted_crash_sum = 0.0
        total_steps = 0
        state_count = 0
        provider_decisions = 0
        provider_requests = 0
        fallback_decisions = 0
        provider_api_calls = 0
        provider_cost_usd = 0.0
        provider_input_tokens = 0
        provider_output_tokens = 0
        provider_latency_ms = []
        benchmark_states = 0
        benchmark_matched_keys: Set[Tuple[str, int, str]] = set()
        decision_census_states = 0
        candidate_census_states = 0
        branch_rollouts = 0
        first_state_sha256 = None
        nade_native_fidelity_states = 0
        nade_native_fidelity_exact_states = 0
        nade_native_fidelity_slots = 0
        nade_native_fidelity_matched_slots = 0
        replay_steps_expected = 0
        replay_steps_found = 0
        replay_actions_expected = 0
        replay_actions_matched = 0
        replay_actions_missing = 0
        cav_action_counts: Dict[str, int] = {}
        cav_speed_samples = []
        cav_lane_changes_completed = 0
        cav_lane_commands = 0
        cav_emergency_brake_commands = 0
        started = timer()
        output_handle = None
        benchmark_handle = None
        decision_state_handle = None
        candidate_state_handle = None
        trajectory_handle = None
        if state_output is not None:
            state_output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = state_output.open("w", encoding="utf-8")
        if benchmark_output is not None:
            benchmark_output.parent.mkdir(parents=True, exist_ok=True)
            benchmark_handle = benchmark_output.open("w", encoding="utf-8")
        if decision_state_output is not None:
            decision_state_output.parent.mkdir(parents=True, exist_ok=True)
            decision_state_handle = decision_state_output.open("w", encoding="utf-8")
        if candidate_state_output is not None:
            candidate_state_output.parent.mkdir(parents=True, exist_ok=True)
            candidate_state_handle = candidate_state_output.open("w", encoding="utf-8")
        if trajectory_output is not None:
            trajectory_handle = open_trajectory(trajectory_output)
        try:
            for episode in range(episodes):
                global_val.episode = episode
                obs_and_indicator, _ = env.reset()
                obs, action_indicator = obs_and_indicator
                vehicle_ids = {
                    id(vehicle): ("AV" if index == 0 else f"BV-{index:03d}")
                    for index, vehicle in enumerate(env.road.vehicles)
                }
                vehicle_history: Dict[int, Dict[str, Any]] = {}
                done = False
                step = 0
                episode_weight = 1.0
                info = {"scene_type": None}
                previous_cav_lane = int(env.vehicle.lane_index[2])
                while not done:
                    if step >= max_steps:
                        raise RuntimeError(
                            f"Episode {episode} exceeded max_steps={max_steps}"
                        )
                    ensure_vehicle_ids(env, vehicle_ids)
                    replay_execution_details = []
                    if replayed_background is not None:
                        replay_steps_expected += 1
                        # Execute the original NADE selection path before
                        # overriding its chosen actions.  Besides selecting a
                        # BV, this public implementation consumes RNG state and
                        # updates vehicle flags used by the NDD continuation.
                        # Preserving those side effects is required for exact
                        # same-CAV replay of the uncontrolled background traffic.
                        select_controlled_bv_and_action(obs.bv_observation, env)
                        replay_actions = replayed_background.actions_for(
                            seed, episode, step
                        )
                        if replayed_background.has_step(seed, episode, step):
                            replay_steps_found += 1
                        bv_actions = [None] * len(env.controlled_bvs)
                        nade_weights = [None] * len(env.controlled_bvs)
                        max_criticality = replayed_background.criticality_for(
                            seed, episode, step
                        )
                        controlled_by_id = {
                            vehicle_ids[id(candidate)]: index
                            for index, candidate in enumerate(env.controlled_bvs)
                        }
                        for replay_action in replay_actions:
                            replay_actions_expected += 1
                            candidate_index = controlled_by_id.get(
                                replay_action.vehicle_id
                            )
                            if candidate_index is None:
                                replay_actions_missing += 1
                                continue
                            bv_actions[candidate_index] = replay_action.action_index
                            replay_actions_matched += 1
                            replay_execution_details.append(
                                {
                                    "candidate_index": candidate_index,
                                    "vehicle_id": replay_action.vehicle_id,
                                    "decision_source": "fixed_background_action_replay",
                                    "action_index": replay_action.action_index,
                                    "action": replay_action.action,
                                    "importance_ratio": 1.0,
                                    "fallback": False,
                                }
                            )
                    elif background_policy == "nade":
                        result = select_controlled_bv_and_action(
                            obs.bv_observation, env
                        )
                        bv_actions, nade_weights, max_criticality, _, _ = result
                    else:
                        # Passing None for every candidate lets the public
                        # environment execute its NDD/NDE background policy
                        # without NADE intervention.
                        bv_actions = [None] * len(env.controlled_bvs)
                        nade_weights = [None] * len(env.controlled_bvs)
                        max_criticality = 0.0
                    states = []
                    for candidate_index, candidate in enumerate(env.controlled_bvs):
                        candidate_key = id(candidate)
                        candidate_x = float(candidate.position[0] - env.vehicle.position[0])
                        candidate_relative_speed = float(
                            candidate.velocity - env.vehicle.velocity
                        )
                        previous = vehicle_history.get(candidate_key)
                        history = {
                            "candidate_index": candidate_index,
                            "history_available": previous is not None,
                        }
                        if previous is not None:
                            history.update(
                                {
                                    "previous_relative_longitudinal_m": previous[
                                        "relative_longitudinal_m"
                                    ],
                                    "relative_longitudinal_change_m": round(
                                        candidate_x - previous["relative_longitudinal_m"],
                                        3,
                                    ),
                                    "previous_relative_speed_mps": previous[
                                        "relative_speed_mps"
                                    ],
                                    "relative_speed_change_mps": round(
                                        candidate_relative_speed
                                        - previous["relative_speed_mps"],
                                        3,
                                    ),
                                    "previous_lane_index": previous["lane_index"],
                                    "lane_changed": int(candidate.lane_index[2])
                                    != previous["lane_index"],
                                }
                            )
                        state = encode_nade_state(
                            env,
                            candidate,
                            episode_id=f"nade-{seed}-{episode}",
                            decision_step=step,
                            criticality=float(getattr(candidate, "criticality", 0.0)),
                            recent_history=history,
                            vehicle_ids=vehicle_ids,
                            native_observation=obs.bv_observation[candidate_index],
                        )
                        if representation_profile == NADE_NATIVE_V1:
                            fidelity = audit_nade_native_slots(
                                state,
                                obs.bv_observation[candidate_index],
                                vehicle_ids=vehicle_ids,
                            )
                            nade_native_fidelity_states += 1
                            nade_native_fidelity_exact_states += int(
                                fidelity["exact_match"]
                            )
                            nade_native_fidelity_slots += int(
                                fidelity["total_slots"]
                            )
                            nade_native_fidelity_matched_slots += int(
                                fidelity["matched_slots"]
                            )
                        vehicle_history[candidate_key] = {
                            "relative_longitudinal_m": candidate_x,
                            "relative_speed_mps": candidate_relative_speed,
                            "lane_index": int(candidate.lane_index[2]),
                        }
                        canonical = state.canonical_representation(
                            representation_profile
                        )
                        if first_state_sha256 is None:
                            first_state_sha256 = hashlib.sha256(
                                canonical.encode("utf-8")
                            ).hexdigest()
                        if output_handle is not None:
                            output_handle.write(canonical + "\n")
                        states.append(state)
                        state_count += 1
                    selected_decision_indices = [
                        index
                        for index, action_index in enumerate(bv_actions)
                        if action_index is not None
                    ]
                    selected_execution_details = list(replay_execution_details)
                    if candidate_state_handle is not None:
                        cav_action_probabilities = _get_Surrogate_CAV_action_probability(env)
                        for candidate_index, state in enumerate(states):
                            criticality = state.risk.nade_criticality
                            f1, r1, f0, r0, f2, r2 = obs.bv_observation[
                                candidate_index
                            ]
                            candidate = env.controlled_bvs[candidate_index]
                            if r1 is env.vehicle:
                                interaction_case = "bv_ahead_same_lane"
                            elif r0 is env.vehicle or r2 is env.vehicle:
                                interaction_case = "bv_ahead_adjacent_lane"
                            elif abs(
                                int(candidate.lane_index[2])
                                - int(env.vehicle.lane_index[2])
                            ) == 2:
                                interaction_case = "two_lanes_apart"
                            elif f0 is env.vehicle or f2 is env.vehicle:
                                interaction_case = "bv_behind_adjacent_lane"
                            elif f1 is env.vehicle:
                                interaction_case = "bv_behind_same_lane"
                            else:
                                interaction_case = "av_not_relevant_neighbor"
                            candidate_state_handle.write(
                                json.dumps(
                                    {
                                        "candidate_census_version": "highway-nade-candidate-census-v1",
                                        "collection_seed": seed,
                                        "collection_episode": episode,
                                        "candidate_index": candidate_index,
                                        "nade_selected": candidate_index
                                        in selected_decision_indices,
                                        "positive_criticality": criticality is not None
                                        and criticality > 0.0,
                                        "nade_interaction_case": interaction_case,
                                        "surrogate_av_action_probabilities": {
                                            "left": float(cav_action_probabilities[0]),
                                            "keep_lane": float(cav_action_probabilities[1]),
                                            "right": float(cav_action_probabilities[2]),
                                        },
                                        "state": state.to_dict(),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            candidate_census_states += 1
                    if decision_state_handle is not None:
                        all_actions = nade_action_space()
                        actions_by_index = {
                            action.index: action for action in all_actions
                        }
                        for candidate_index in selected_decision_indices:
                            probability_rng_state = np.random.get_state()
                            try:
                                probability_env = copy.deepcopy(env)
                                probability_observation = (
                                    probability_env.observe_cav_bv()[0]
                                )
                                ndd_pdf = [
                                    float(value)
                                    for value in get_NDD_possi(
                                        probability_env.controlled_bvs[
                                            candidate_index
                                        ],
                                        probability_observation.bv_observation[
                                            candidate_index
                                        ],
                                    )
                                ]
                            finally:
                                np.random.set_state(probability_rng_state)
                            feasible_indices = [
                                index
                                for index, probability in enumerate(ndd_pdf)
                                if probability > 0.0 and index in actions_by_index
                            ]
                            census_state = replace(
                                states[candidate_index],
                                available_actions=[
                                    actions_by_index[index]
                                    for index in feasible_indices
                                ],
                            )
                            sampled_index = int(bv_actions[candidate_index])
                            decision_state_handle.write(
                                json.dumps(
                                    {
                                        "census_version": "highway-nade-decision-census-v1",
                                        "collection_seed": seed,
                                        "collection_episode": episode,
                                        "selection_source": "public_nade_highest_criticality_bv",
                                        "state": census_state.to_dict(),
                                        "nade_sampled_action": actions_by_index[
                                            sampled_index
                                        ].value,
                                        "naturalistic_action_probabilities": {
                                            actions_by_index[index].value: ndd_pdf[index]
                                            for index in feasible_indices
                                        },
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            decision_census_states += 1
                    if provider is None:
                        decision_weight = 1.0
                        for weight in nade_weights:
                            if weight is not None:
                                decision_weight *= float(weight)
                        if replayed_background is None:
                            all_actions = nade_action_space()
                            for candidate_index in selected_decision_indices:
                                action_index = int(bv_actions[candidate_index])
                                selected_execution_details.append(
                                    {
                                        "candidate_index": candidate_index,
                                        "vehicle_id": states[
                                            candidate_index
                                        ].controlled_background_vehicle.vehicle_id,
                                        "decision_source": "public_nade",
                                        "action_index": action_index,
                                        "action": all_actions[action_index].value,
                                        "importance_ratio": (
                                            float(nade_weights[candidate_index])
                                            if nade_weights[candidate_index] is not None
                                            else 1.0
                                        ),
                                        "fallback": False,
                                    }
                                )
                    else:
                        decision_weight = 1.0
                        selected = [
                            index for index, action_index in enumerate(bv_actions)
                            if action_index is not None
                        ]
                        for candidate_index in selected:
                            provider_requests += 1
                            candidate = env.controlled_bvs[candidate_index]
                            numpy_random_state = np.random.get_state()
                            python_random_state = random.getstate()
                            torch_random_state = torch.get_rng_state()
                            ndd_pdf = [
                                float(value)
                                for value in get_NDD_possi(
                                    candidate, obs.bv_observation[candidate_index]
                                )
                            ]
                            all_actions = nade_action_space()
                            feasible_actions = [
                                action for action, probability in zip(all_actions, ndd_pdf)
                                if probability > 0.0
                            ]
                            provider_state = states[candidate_index]
                            base_specification = adversarial_specification(provider_state)
                            specification = DecisionSpecification(
                                name=base_specification.name,
                                instructions=base_specification.instructions,
                                actions=feasible_actions,
                            )
                            original_index = int(bv_actions[candidate_index])
                            try:
                                provider_result = provider.decide(
                                    provider_state, specification
                                )
                                provider_result.validate(specification)
                                usage = provider_result.usage or {}
                                provider_api_calls += int(usage.get("api_calls", 0) or 0)
                                provider_cost_usd += float(usage.get("cost", 0.0) or 0.0)
                                provider_input_tokens += int(
                                    usage.get(
                                        "input_tokens", usage.get("inputTokens", 0)
                                    )
                                    or 0
                                )
                                provider_output_tokens += int(
                                    usage.get(
                                        "output_tokens", usage.get("outputTokens", 0)
                                    )
                                    or 0
                                )
                                provider_latency_ms.append(
                                    float(provider_result.latency_ms)
                                )
                                proposal = mixed_proposal(
                                    provider_result.probabilities,
                                    ndd_pdf,
                                    all_actions,
                                    ndd_mix=ndd_mix,
                                )
                                np.random.set_state(numpy_random_state)
                                random.setstate(python_random_state)
                                torch.set_rng_state(torch_random_state)
                                executed_index = int(
                                    proposal_rng.choice(len(all_actions), p=proposal)
                                )
                                bv_actions[candidate_index] = executed_index
                                ratio = importance_ratio(executed_index, ndd_pdf, proposal)
                                decision_weight *= ratio
                                provider_result.raw_response = dict(
                                    provider_result.raw_response
                                )
                                provider_result.raw_response["execution"] = {
                                    "recommended_action": provider_result.action,
                                    "executed_action": all_actions[executed_index].value,
                                    "executed_action_index": executed_index,
                                    "ndd_probability": ndd_pdf[executed_index],
                                    "proposal_probability": proposal[executed_index],
                                    "importance_ratio": ratio,
                                    "ndd_mix": ndd_mix,
                                    "fallback": False,
                                }
                                provider_decisions += 1
                            except Exception as error:
                                np.random.set_state(numpy_random_state)
                                random.setstate(python_random_state)
                                torch.set_rng_state(torch_random_state)
                                ratio = (
                                    float(nade_weights[candidate_index])
                                    if nade_weights[candidate_index] is not None
                                    else 1.0
                                )
                                decision_weight *= ratio
                                fallback_decisions += 1
                                error_usage = getattr(error, "usage", {}) or {}
                                provider_api_calls += int(
                                    error_usage.get("api_calls", 0) or 0
                                )
                                provider_cost_usd += float(
                                    error_usage.get("cost", 0.0) or 0.0
                                )
                                provider_input_tokens += int(
                                    error_usage.get("input_tokens", 0) or 0
                                )
                                provider_output_tokens += int(
                                    error_usage.get("output_tokens", 0) or 0
                                )
                                error_latency_ms = float(
                                    getattr(error, "latency_ms", 0.0) or 0.0
                                )
                                if error_latency_ms:
                                    provider_latency_ms.append(error_latency_ms)
                                provider_result = DecisionResult(
                                    action=all_actions[original_index].value,
                                    probabilities={
                                        all_actions[original_index].value: 1.0
                                    },
                                    confidence=1.0,
                                    provider=provider.name,
                                    model="fallback-to-nade",
                                    latency_ms=error_latency_ms,
                                    usage=dict(error_usage),
                                    raw_response={
                                        "reliability": getattr(error, "audit", None),
                                        "execution": {
                                            "recommended_action": None,
                                            "executed_action": all_actions[
                                                original_index
                                            ].value,
                                            "executed_action_index": original_index,
                                            "importance_ratio": ratio,
                                            "fallback": True,
                                        }
                                    },
                                    fallback_used=True,
                                    error=f"{type(error).__name__}: {str(error)[:500]}",
                                )
                            if decision_trace is not None:
                                append_trace(
                                    decision_trace,
                                    provider_state,
                                    specification,
                                    provider_result,
                                    representation_profile=getattr(
                                        provider,
                                        "representation_profile",
                                        representation_profile,
                                    ),
                                )
                            execution = provider_result.raw_response.get("execution", {})
                            selected_execution_details.append(
                                {
                                    "candidate_index": candidate_index,
                                    "vehicle_id": provider_state.controlled_background_vehicle.vehicle_id,
                                    "decision_source": (
                                        "fallback_to_nade"
                                        if provider_result.fallback_used
                                        else provider.name
                                    ),
                                    "recommended_action": execution.get(
                                        "recommended_action", provider_result.action
                                    ),
                                    "action_index": int(bv_actions[candidate_index]),
                                    "action": execution.get(
                                        "executed_action",
                                        all_actions[int(bv_actions[candidate_index])].value,
                                    ),
                                    "ndd_probability": execution.get("ndd_probability"),
                                    "proposal_probability": execution.get(
                                        "proposal_probability"
                                    ),
                                    "importance_ratio": execution.get(
                                        "importance_ratio"
                                    ),
                                    "fallback": bool(provider_result.fallback_used),
                                }
                            )
                    indicator = cav_agent.lane_conflict_safety_check(
                        obs.cav_observation, action_indicator.cav_indicator
                    )
                    cav_action = cav_agent.decision(obs.cav_observation, indicator)
                    cav_action_key = str(int(cav_action))
                    cav_action_counts[cav_action_key] = (
                        cav_action_counts.get(cav_action_key, 0) + 1
                    )
                    if int(cav_action) in {0, 1}:
                        cav_lane_commands += 1
                    cav_action_value = global_val.ACTIONS[int(cav_action)]
                    if (
                        cav_action_value not in {"LANE_LEFT", "LANE_RIGHT"}
                        and float(cav_action_value) <= -3.8
                    ):
                        cav_emergency_brake_commands += 1
                    selected_vehicle_ids = [
                        detail["vehicle_id"] for detail in selected_execution_details
                    ]
                    pre_trajectory_state = (
                        capture_environment(
                            env,
                            vehicle_ids,
                            selected_vehicle_ids=selected_vehicle_ids,
                        )
                        if trajectory_handle is not None
                        else None
                    )
                    if benchmark_handle is not None and benchmark_states < max_benchmark_states:
                        selected_for_benchmark = [
                            index
                            for index, action_index in enumerate(bv_actions)
                            if action_index is not None
                            and (
                                benchmark_selection_keys is None
                                or (
                                    states[index].episode_id,
                                    int(states[index].decision_step),
                                    states[index].controlled_background_vehicle.vehicle_id,
                                )
                                in benchmark_selection_keys
                            )
                        ]
                        all_actions = nade_action_space()
                        actions_by_index = {action.index: action for action in all_actions}
                        for candidate_index in selected_for_benchmark:
                            if benchmark_states >= max_benchmark_states:
                                break
                            candidate = env.controlled_bvs[candidate_index]
                            # Public NADE's probability-query path samples from
                            # NumPy and can set vehicle flags even with
                            # modify_flag=False. Query on a clone and restore
                            # RNG state so benchmark labeling cannot perturb the
                            # main trajectory.
                            probability_rng_state = np.random.get_state()
                            try:
                                probability_env = copy.deepcopy(env)
                                probability_observation = (
                                    probability_env.observe_cav_bv()[0]
                                )
                                ndd_pdf = [
                                    float(value)
                                    for value in get_NDD_possi(
                                        probability_env.controlled_bvs[
                                            candidate_index
                                        ],
                                        probability_observation.bv_observation[
                                            candidate_index
                                        ],
                                    )
                                ]
                            finally:
                                np.random.set_state(probability_rng_state)
                            feasible_indices = [
                                index
                                for index, probability in enumerate(ndd_pdf)
                                if probability > 0.0
                                and index in actions_by_index
                                and any(
                                    action.index == index
                                    for action in states[candidate_index].available_actions
                                )
                            ]
                            feasible_actions = [
                                actions_by_index[index] for index in feasible_indices
                            ]
                            benchmark_state = replace(
                                states[candidate_index], available_actions=feasible_actions
                            )
                            stability = None
                            repetition_outcomes = None
                            rollout_seeds = None
                            horizon_outcomes = None
                            horizon_stability = None
                            horizon_repetition_outcomes = None
                            horizon_validation = None
                            seed_base = (
                                seed * 1000003
                                + episode * 10007
                                + step * 101
                                + candidate_index * 17
                            ) % (2**32)
                            if rollout_repetitions == 1:
                                if validation_horizon_steps > 1:
                                    rollout_seeds = [seed_base]
                                outcomes = evaluate_one_step_branches(
                                    env,
                                    action_type=Action,
                                    cav_action=cav_action,
                                    candidate_index=candidate_index,
                                    action_indices=feasible_indices,
                                    action_options=actions_by_index,
                                    numpy_module=np,
                                    torch_module=torch,
                                    rng_seed=(
                                        rollout_seeds[0]
                                        if rollout_seeds is not None
                                        else None
                                    ),
                                )
                            else:
                                rollout_seeds = [
                                    (seed_base + repeat) % (2**32)
                                    for repeat in range(rollout_repetitions)
                                ]
                                repeated = evaluate_repeated_one_step_branches(
                                    env,
                                    action_type=Action,
                                    cav_action=cav_action,
                                    candidate_index=candidate_index,
                                    action_indices=feasible_indices,
                                    action_options=actions_by_index,
                                    numpy_module=np,
                                    torch_module=torch,
                                    seeds=rollout_seeds,
                                )
                                outcomes = repeated["action_outcomes"]
                                stability = repeated["stability"]
                                repetition_outcomes = repeated["repetition_outcomes"]
                            if validation_horizon_steps > 1:
                                validation_seeds = rollout_seeds or [seed_base]
                                if rollout_repetitions == 1:
                                    horizon_outcomes = evaluate_horizon_branches(
                                        env,
                                        action_type=Action,
                                        cav_agent=cav_agent,
                                        select_controlled_action=select_controlled_bv_and_action,
                                        cav_action=cav_action,
                                        candidate_index=candidate_index,
                                        action_indices=feasible_indices,
                                        action_options=actions_by_index,
                                        numpy_module=np,
                                        torch_module=torch,
                                        policy_steps=validation_horizon_steps,
                                        rng_seed=validation_seeds[0],
                                    )
                                else:
                                    repeated_horizon = evaluate_repeated_horizon_branches(
                                        env,
                                        action_type=Action,
                                        cav_agent=cav_agent,
                                        select_controlled_action=select_controlled_bv_and_action,
                                        cav_action=cav_action,
                                        candidate_index=candidate_index,
                                        action_indices=feasible_indices,
                                        action_options=actions_by_index,
                                        numpy_module=np,
                                        torch_module=torch,
                                        policy_steps=validation_horizon_steps,
                                        seeds=validation_seeds,
                                    )
                                    horizon_outcomes = repeated_horizon["action_outcomes"]
                                    horizon_stability = repeated_horizon["stability"]
                                    horizon_repetition_outcomes = repeated_horizon[
                                        "repetition_outcomes"
                                    ]
                                horizon_validation = compare_rollout_rankings(
                                    outcomes, horizon_outcomes
                                )
                            benchmark_handle.write(
                                json.dumps(
                                    {
                                        "benchmark_version": (
                                            "highway-decision-benchmark-v3"
                                            if validation_horizon_steps > 1
                                            else (
                                                "highway-decision-benchmark-v1"
                                                if rollout_repetitions == 1
                                                else "highway-decision-benchmark-v2"
                                            )
                                        ),
                                        "collection_seed": seed,
                                        "collection_episode": episode,
                                        "label_source": (
                                            "one-step-label-with-nade-continuation-validation"
                                            if validation_horizon_steps > 1
                                            else (
                                                "one-policy-interval-nade-branch-rollout"
                                                if rollout_repetitions == 1
                                                else "repeated-one-policy-interval-nade-branch-rollout"
                                            )
                                        ),
                                        "common_random_numbers": True,
                                        "policy_interval_s": 1.0,
                                        "rollout_repetitions": rollout_repetitions,
                                        "rollout_seeds": rollout_seeds,
                                        "validation_horizon_policy_steps": (
                                            validation_horizon_steps
                                            if validation_horizon_steps > 1
                                            else None
                                        ),
                                        "state": benchmark_state.to_dict(),
                                        "representations": {
                                            profile: benchmark_state.to_representation(profile)
                                            for profile in REPRESENTATION_PROFILES
                                        },
                                        "naturalistic_action_probabilities": {
                                            actions_by_index[index].value: ndd_pdf[index]
                                            for index in feasible_indices
                                        },
                                        "action_outcomes": outcomes,
                                        "stability": stability,
                                        "repetition_outcomes": repetition_outcomes,
                                        "horizon_action_outcomes": horizon_outcomes,
                                        "horizon_stability": horizon_stability,
                                        "horizon_repetition_outcomes": (
                                            horizon_repetition_outcomes
                                        ),
                                        "horizon_validation": horizon_validation,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            benchmark_states += 1
                            benchmark_matched_keys.add(
                                (
                                    benchmark_state.episode_id,
                                    int(benchmark_state.decision_step),
                                    benchmark_state.controlled_background_vehicle.vehicle_id,
                                )
                            )
                            branch_rollouts += len(outcomes) * rollout_repetitions
                            if horizon_outcomes is not None:
                                branch_rollouts += (
                                    len(horizon_outcomes) * rollout_repetitions
                                )
                    obs_and_indicator, done, info, _ = env.step(
                        Action(cav_action=cav_action, bv_action=bv_actions)
                    )
                    cav_speed_samples.append(float(env.vehicle.velocity))
                    current_cav_lane = int(env.vehicle.lane_index[2])
                    cav_lane_changes_completed += int(
                        current_cav_lane != previous_cav_lane
                    )
                    previous_cav_lane = current_cav_lane
                    if trajectory_handle is not None:
                        post_trajectory_state = capture_environment(
                            env,
                            vehicle_ids,
                            selected_vehicle_ids=selected_vehicle_ids,
                        )
                        write_trajectory_step(
                            trajectory_handle,
                            seed=seed,
                            episode=episode,
                            decision_step=step,
                            selected_actions=selected_execution_details,
                            cav_action=cav_action,
                            pre_state=pre_trajectory_state,
                            post_state=post_trajectory_state,
                            done=done,
                            scene_type=info.get("scene_type"),
                            max_criticality=max_criticality,
                        )
                    episode_weight *= decision_weight
                    obs, action_indicator = obs_and_indicator
                    step += 1
                    total_steps += 1
                counts[info["scene_type"]] += 1
                episode_weights.append(episode_weight)
                if info["scene_type"] == "AV-Crash":
                    weighted_crash_sum += episode_weight
        finally:
            if output_handle is not None:
                output_handle.close()
            if benchmark_handle is not None:
                benchmark_handle.close()
            if decision_state_handle is not None:
                decision_state_handle.close()
            if candidate_state_handle is not None:
                candidate_state_handle.close()
            if trajectory_handle is not None:
                trajectory_handle.close()
        weight_sum = sum(episode_weights)
        weight_square_sum = sum(weight * weight for weight in episode_weights)
        return {
            "baseline": (
                "public-nde"
                if background_policy == "nde"
                else ("public-nade" if provider is None else "provider-controlled-nade")
            ),
            "provider": (
                provider.name
                if provider is not None
                else background_policy
            ),
            "background_policy": background_policy,
            "cav_policy": cav_policy,
            "background_action_replay": (
                str(background_action_replay)
                if background_action_replay is not None
                else None
            ),
            "background_action_replay_fidelity": (
                {
                    "steps_expected": replay_steps_expected,
                    "steps_found": replay_steps_found,
                    "actions_expected": replay_actions_expected,
                    "actions_matched": replay_actions_matched,
                    "actions_missing": replay_actions_missing,
                    "action_match_rate": (
                        replay_actions_matched / replay_actions_expected
                        if replay_actions_expected
                        else 1.0
                    ),
                }
                if replayed_background is not None
                else None
            ),
            "cav_behavior": {
                "action_counts_by_index": cav_action_counts,
                "lane_command_count": cav_lane_commands,
                "lane_changes_completed": cav_lane_changes_completed,
                "emergency_brake_command_count": cav_emergency_brake_commands,
                "emergency_brake_command_rate": (
                    cav_emergency_brake_commands / total_steps
                    if total_steps
                    else None
                ),
                "speed_min_mps": min(cav_speed_samples) if cav_speed_samples else None,
                "speed_max_mps": max(cav_speed_samples) if cav_speed_samples else None,
                "speed_mean_mps": (
                    sum(cav_speed_samples) / len(cav_speed_samples)
                    if cav_speed_samples
                    else None
                ),
            },
            "episodes": episodes,
            "seed": seed,
            "results": counts,
            "crash_rate_unweighted": counts["AV-Crash"] / episodes,
            "crash_rate_importance_sampling": weighted_crash_sum / episodes,
            "importance_weight_sum": weight_sum,
            "importance_weight_square_sum": weight_square_sum,
            "weighted_crash_sum": weighted_crash_sum,
            "effective_sample_size": (
                weight_sum * weight_sum / weight_square_sum if weight_square_sum else 0.0
            ),
            "decision_steps": total_steps,
            "encoded_candidate_states": state_count,
            "provider_decisions": provider_decisions,
            "provider_requests": provider_requests,
            "fallback_decisions": fallback_decisions,
            "provider_api_calls_successful": provider_api_calls,
            "provider_cost_usd_successful": provider_cost_usd,
            "provider_input_tokens_successful": provider_input_tokens,
            "provider_output_tokens_successful": provider_output_tokens,
            "provider_mean_latency_ms_successful": (
                sum(provider_latency_ms) / len(provider_latency_ms)
                if provider_latency_ms
                else None
            ),
            "provider_usage_scope": (
                "observed response usage; transport failures may lack provider usage"
                if provider is not None
                else None
            ),
            "first_state_sha256": first_state_sha256,
            "elapsed_s": round(timer() - started, 3),
            "state_output": str(state_output) if state_output is not None else None,
            "decision_trace": str(decision_trace) if decision_trace is not None else None,
            "trajectory_output": (
                str(trajectory_output) if trajectory_output is not None else None
            ),
            "ndd_mix": ndd_mix if provider is not None else None,
            "representation_profile": representation_profile,
            "nade_native_reconstruction_fidelity": (
                {
                    "audited_states": nade_native_fidelity_states,
                    "exact_state_matches": nade_native_fidelity_exact_states,
                    "exact_state_match_rate": (
                        nade_native_fidelity_exact_states
                        / nade_native_fidelity_states
                        if nade_native_fidelity_states
                        else None
                    ),
                    "matched_slots": nade_native_fidelity_matched_slots,
                    "audited_slots": nade_native_fidelity_slots,
                    "slot_match_rate": (
                        nade_native_fidelity_matched_slots
                        / nade_native_fidelity_slots
                        if nade_native_fidelity_slots
                        else None
                    ),
                }
                if representation_profile == NADE_NATIVE_V1
                else None
            ),
            "benchmark_output": (
                str(benchmark_output) if benchmark_output is not None else None
            ),
            "benchmark_states": benchmark_states,
            "benchmark_selection_requested": (
                len(benchmark_selection_keys)
                if benchmark_selection_keys is not None
                else None
            ),
            "benchmark_selection_missing": (
                [list(key) for key in sorted(benchmark_selection_keys - benchmark_matched_keys)]
                if benchmark_selection_keys is not None
                else None
            ),
            "decision_state_output": (
                str(decision_state_output)
                if decision_state_output is not None
                else None
            ),
            "decision_census_states": decision_census_states,
            "candidate_state_output": (
                str(candidate_state_output) if candidate_state_output is not None else None
            ),
            "candidate_census_states": candidate_census_states,
            "branch_rollouts": branch_rollouts,
            "rollout_repetitions": rollout_repetitions,
            "validation_horizon_steps": validation_horizon_steps,
        }
    finally:
        os.chdir(str(previous_cwd))
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
