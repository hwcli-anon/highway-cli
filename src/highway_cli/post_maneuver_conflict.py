from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .hierarchical_actions import CHANGE_LEFT, CHANGE_RIGHT, KEEP_LANE
from .interaction_graph import render_interaction_graph_intent
from .schema import ScenarioState, VehicleState


POST_MANEUVER_CONFLICT_V1 = "post-maneuver-conflict-v1"
POST_MANEUVER_CONFLICT_V2 = "post-maneuver-conflict-v2"
STAGE_ROUTED_CONFLICT_V1 = "stage-routed-conflict-v1"
PROJECTION_TIMES_S = (1.0, 2.0, 3.0)
FOLLOW_UP_HORIZON_S = 2.0


def _rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _relation(gap_m: float) -> str:
    if gap_m > 0.5:
        return "ahead"
    if gap_m < -0.5:
        return "behind"
    return "alongside"


def _time_to_overlap(gap_m: float, relative_speed_mps: float) -> Optional[float]:
    if abs(relative_speed_mps) < 1e-12 or gap_m * relative_speed_mps >= 0.0:
        return None
    value = -gap_m / relative_speed_mps
    return value if value >= 0.0 else None


def _minimum_projected_gap(
    gap_m: float, relative_speed_mps: float, horizon_s: float
) -> Tuple[float, float]:
    candidates = [0.0, horizon_s]
    overlap = _time_to_overlap(gap_m, relative_speed_mps)
    if overlap is not None and overlap <= horizon_s:
        candidates.append(overlap)
    time_s = min(
        candidates,
        key=lambda value: abs(gap_m + relative_speed_mps * value),
    )
    return abs(gap_m + relative_speed_mps * time_s), time_s


def _actor_card(
    role: str, actor: VehicleState, controlled: VehicleState
) -> Dict[str, Any]:
    gap = float(actor.relative_longitudinal_m) - float(
        controlled.relative_longitudinal_m
    )
    relative_speed = float(actor.relative_speed_mps) - float(
        controlled.relative_speed_mps
    )
    minimum_gap, minimum_time = _minimum_projected_gap(
        gap, relative_speed, max(PROJECTION_TIMES_S)
    )
    overlap = _time_to_overlap(gap, relative_speed)
    return {
        "actor_role": role,
        "relation_to_controlled_bv": _relation(gap),
        "signed_gap_m": _rounded(gap),
        "relative_speed_to_controlled_bv_mps": _rounded(relative_speed),
        "closing": gap * relative_speed < 0.0,
        "time_to_longitudinal_overlap_s": _rounded(overlap),
        "projected_signed_gap_m": {
            f"t_plus_{int(time_s)}s": _rounded(gap + relative_speed * time_s)
            for time_s in PROJECTION_TIMES_S
        },
        "minimum_abs_gap_within_3s_m": _rounded(minimum_gap),
        "time_of_minimum_gap_s": _rounded(minimum_time),
    }


def _target_lane_actors(
    state: ScenarioState, target_lane: int
) -> List[Tuple[str, VehicleState]]:
    actors: List[Tuple[str, VehicleState]] = []
    if int(state.autonomous_vehicle.lane_index) == target_lane:
        actors.append(("AV", state.autonomous_vehicle))
    actors.extend(
        ("NEIGHBOR", vehicle)
        for vehicle in state.nearby_vehicles
        if int(vehicle.lane_index) == target_lane
    )
    return actors


def _nearest_target_lane_traffic(
    state: ScenarioState, target_lane: int
) -> Dict[str, Any]:
    controlled = state.controlled_background_vehicle
    cards = [
        _actor_card(role, actor, controlled)
        for role, actor in _target_lane_actors(state, target_lane)
    ]
    front = min(
        (card for card in cards if float(card["signed_gap_m"]) > 0.5),
        key=lambda card: float(card["signed_gap_m"]),
        default=None,
    )
    rear = min(
        (card for card in cards if float(card["signed_gap_m"]) < -0.5),
        key=lambda card: abs(float(card["signed_gap_m"])),
        default=None,
    )
    alongside = [
        card for card in cards if abs(float(card["signed_gap_m"])) <= 0.5
    ]
    return {
        "nearest_front_actor": front,
        "nearest_rear_actor": rear,
        "alongside_actor_roles": sorted(
            str(card["actor_role"]) for card in alongside
        ),
        "observed_actor_count_in_target_lane": len(cards),
    }


def _av_relation_after_maneuver(
    state: ScenarioState, target_lane: int
) -> Dict[str, Any]:
    av = state.autonomous_vehicle
    controlled = state.controlled_background_vehicle
    same_lane = int(av.lane_index) == target_lane
    card = _actor_card("AV", av, controlled)
    gap = -float(controlled.relative_longitudinal_m)
    if same_lane:
        controlled_relation = _relation(-gap)
        template = {
            "ahead": "controlled_bv_ahead_in_av_path",
            "behind": "controlled_bv_behind_in_av_path",
            "alongside": "controlled_bv_merges_alongside_av",
        }[controlled_relation]
    else:
        template = "no_direct_same_lane_av_conflict_after_maneuver"
    return {
        "same_lane_as_av_after_maneuver": same_lane,
        "interaction_template": template,
        "controlled_bv_relation_to_av": _relation(
            float(controlled.relative_longitudinal_m)
        ),
        "av_relative_motion_if_same_lane": card if same_lane else None,
    }


def _available_longitudinal_accelerations(state: ScenarioState) -> List[float]:
    return sorted(
        {
            float(action.acceleration_mps2 or 0.0)
            for action in state.available_actions
            if action.kind == "longitudinal"
        }
    )


def _follow_up_pressure_envelope(
    state: ScenarioState, target_lane: int
) -> Dict[str, Any]:
    accelerations = _available_longitudinal_accelerations(state)
    if not accelerations:
        return {"available": False}
    controlled = state.controlled_background_vehicle
    gap = float(controlled.relative_longitudinal_m)
    relative_speed = float(controlled.relative_speed_mps)
    neutral_available = min(accelerations, key=abs)
    values = sorted({accelerations[0], neutral_available, accelerations[-1]})
    scenarios = []
    for acceleration in values:
        projected_gap = (
            gap
            + relative_speed * FOLLOW_UP_HORIZON_S
            + 0.5 * acceleration * FOLLOW_UP_HORIZON_S**2
        )
        scenarios.append(
            {
                "controlled_bv_acceleration_mps2": _rounded(acceleration),
                "controlled_bv_gap_ahead_of_av_after_2s_m": _rounded(projected_gap),
                "controlled_bv_speed_after_2s_mps": _rounded(
                    max(0.0, float(controlled.speed_mps) + acceleration * FOLLOW_UP_HORIZON_S)
                ),
            }
        )
    return {
        "available": True,
        "only_directly_relevant_when_target_lane_is_av_lane": True,
        "target_lane_is_av_lane": target_lane
        == int(state.autonomous_vehicle.lane_index),
        "horizon_s": FOLLOW_UP_HORIZON_S,
        "representative_available_accelerations": scenarios,
    }


def _conflict_level(card: Dict[str, Any]) -> str:
    av_relation = card["post_maneuver_av_relation"]
    if not av_relation["same_lane_as_av_after_maneuver"]:
        return "indirect"
    motion = av_relation["av_relative_motion_if_same_lane"]
    minimum_gap = float(motion["minimum_abs_gap_within_3s_m"])
    overlap = motion["time_to_longitudinal_overlap_s"]
    if minimum_gap <= 2.5 or (overlap is not None and float(overlap) <= 1.0):
        return "critical"
    if minimum_gap <= 7.5 or (overlap is not None and float(overlap) <= 3.0):
        return "high"
    return "direct_but_separated"


def _maneuver_card(
    state: ScenarioState,
    maneuver: str,
    lane_offset: int,
    supported_actions: Sequence[str],
) -> Dict[str, Any]:
    controlled_lane = int(state.controlled_background_vehicle.lane_index)
    target_lane = controlled_lane + lane_offset
    within_road = 0 <= target_lane < int(state.lane_count)
    executable = bool(supported_actions) and within_road
    card: Dict[str, Any] = {
        "maneuver": maneuver,
        "supported_original_nade_actions": list(supported_actions),
        "target_lane_index": target_lane,
        "target_lane_within_road": within_road,
        "executable": executable,
    }
    if not within_road:
        card.update(
            {
                "post_maneuver_av_relation": None,
                "target_lane_local_traffic": None,
                "follow_up_longitudinal_pressure_envelope": None,
                "conflict_level": "infeasible",
            }
        )
        return card
    card.update(
        {
            "post_maneuver_av_relation": _av_relation_after_maneuver(
                state, target_lane
            ),
            "target_lane_local_traffic": _nearest_target_lane_traffic(
                state, target_lane
            ),
            "follow_up_longitudinal_pressure_envelope": _follow_up_pressure_envelope(
                state, target_lane
            ),
        }
    )
    card["conflict_level"] = _conflict_level(card) if executable else "unavailable"
    return card


def render_post_maneuver_conflict(state: ScenarioState) -> Dict[str, Any]:
    result = render_interaction_graph_intent(state)
    actions = state.available_actions
    longitudinal = [action.value for action in actions if action.kind == "longitudinal"]
    left = [action.value for action in actions if action.value == "LANE_LEFT"]
    right = [action.value for action in actions if action.value == "LANE_RIGHT"]
    result["post_maneuver_conflict_interface"] = {
        "purpose": (
            "Compare observable local traffic relations after each maneuver before "
            "selecting an interaction intent and an original NADE action."
        ),
        "cards_are_counterfactual_current_state_projections_not_rollout_labels": True,
        "maneuver_cards": [
            _maneuver_card(state, KEEP_LANE, 0, longitudinal),
            _maneuver_card(state, CHANGE_LEFT, -1, left),
            _maneuver_card(state, CHANGE_RIGHT, 1, right),
        ],
    }
    result["projection_assumptions"] = {
        "projection_is_not_simulator_rollout": True,
        "uses_only_current_observed_state": True,
        "lane_change_is_counterfactual_lane_assignment_not_lateral_dynamics": True,
        "other_vehicles_do_not_respond": True,
        "longitudinal_model": "constant_velocity_except_explicit_follow_up_envelope",
        "projection_times_s": list(PROJECTION_TIMES_S),
        "follow_up_envelope_horizon_s": FOLLOW_UP_HORIZON_S,
    }
    return result


def _av_safety_relevance(card: Dict[str, Any], av_lane: int) -> str:
    if not card["executable"]:
        return "not_executable"
    relation = card["post_maneuver_av_relation"]
    if relation["same_lane_as_av_after_maneuver"]:
        level = card["conflict_level"]
        return {
            "critical": "direct_critical_av_interaction",
            "high": "direct_high_av_interaction",
            "direct_but_separated": "direct_separated_av_interaction",
        }.get(level, "direct_av_interaction")
    lane_distance = abs(int(card["target_lane_index"]) - int(av_lane))
    return (
        "indirect_adjacent_to_av_lane"
        if lane_distance == 1
        else "no_direct_av_path"
    )


def render_post_maneuver_conflict_v2(state: ScenarioState) -> Dict[str, Any]:
    """AV-centered semantic organization of the v1 counterfactual evidence."""

    result = render_post_maneuver_conflict(state)
    interface = result["post_maneuver_conflict_interface"]
    interface["decision_scope"] = {
        "adversarial_safety_target": "AV",
        "direct_av_interaction_has_priority": True,
        "neighbor_traffic_role": (
            "Feasibility and plausibility context only; ordinary neighbors are not "
            "the adversarial safety target."
        ),
        "when_no_direct_av_path": (
            "Do not prefer a lane change merely because it approaches an ordinary "
            "neighbor; compare whether it creates a justified future AV interaction."
        ),
    }
    av_lane = int(state.autonomous_vehicle.lane_index)
    for card in interface["maneuver_cards"]:
        card["av_safety_relevance"] = _av_safety_relevance(card, av_lane)
        card["av_conflict_level"] = card.pop("conflict_level")
        traffic = card.pop("target_lane_local_traffic")
        card["traffic_feasibility_constraints"] = (
            None
            if traffic is None
            else {
                "interpretation": (
                    "Use these ordinary-traffic gaps to judge feasibility and "
                    "plausibility, not as an independent attack objective."
                ),
                **traffic,
            }
        )
    return result


def render_stage_routed_conflict(state: ScenarioState) -> Dict[str, Any]:
    """Expose AV-centered maneuver evidence with an explicit stage contract."""

    result = render_post_maneuver_conflict_v2(state)
    result["stage_evidence_routing"] = {
        "interaction_intent_stage": (
            "Use AV-centered post-maneuver conflict cards to compare KEEP_LANE, "
            "CHANGE_LEFT, and CHANGE_RIGHT."
        ),
        "leaf_action_stage": (
            "After selecting the interaction intent, use the original interaction "
            "graph and current-state longitudinal relations to choose the exact "
            "compatible NADE leaf action."
        ),
        "motivation": (
            "Maneuver comparison and fine-grained acceleration selection require "
            "different evidence granularity."
        ),
    }
    return result
