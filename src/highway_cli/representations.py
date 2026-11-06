from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .actions import ActionOption
from .hierarchical_actions import (
    CHANGE_LEFT,
    CHANGE_RIGHT,
    KEEP_LANE,
    hierarchical_action_structure,
    maneuver_for_action,
)
from .interaction_graph import (
    INTERACTION_GRAPH_INTENT_V1,
    render_interaction_graph_intent,
)
from .post_maneuver_conflict import (
    POST_MANEUVER_CONFLICT_V1,
    POST_MANEUVER_CONFLICT_V2,
    STAGE_ROUTED_CONFLICT_V1,
    render_post_maneuver_conflict,
    render_post_maneuver_conflict_v2,
    render_stage_routed_conflict,
)
from .schema import ScenarioState, VehicleState


FULL_STATE_V1 = "full-state-v1"
KINEMATIC_V1 = "kinematic-v1"
RELATIONAL_V1 = "relational-v1"
RISK_AWARE_V1 = "risk-aware-v1"
ACTION_CONDITIONED_V1 = "action-conditioned-v1"
TEMPORAL_RELATIONAL_V1 = "temporal-relational-v1"
TEMPORAL_ACTION_CONDITIONED_V1 = "temporal-action-conditioned-v1"
HIERARCHICAL_RELATIONAL_V1 = "hierarchical-relational-v1"
SYMMETRIC_MANEUVER_EVIDENCE_V1 = "symmetric-maneuver-evidence-v1"
SYMMETRIC_MANEUVER_EVIDENCE_V2 = "symmetric-maneuver-evidence-v2"
NADE_NATIVE_V1 = "nade-native-v1"

REPRESENTATION_PROFILES = (
    FULL_STATE_V1,
    KINEMATIC_V1,
    RELATIONAL_V1,
    RISK_AWARE_V1,
    ACTION_CONDITIONED_V1,
    TEMPORAL_RELATIONAL_V1,
    TEMPORAL_ACTION_CONDITIONED_V1,
    HIERARCHICAL_RELATIONAL_V1,
    SYMMETRIC_MANEUVER_EVIDENCE_V1,
    SYMMETRIC_MANEUVER_EVIDENCE_V2,
    NADE_NATIVE_V1,
    INTERACTION_GRAPH_INTENT_V1,
    POST_MANEUVER_CONFLICT_V1,
    POST_MANEUVER_CONFLICT_V2,
    STAGE_ROUTED_CONFLICT_V1,
)


def validate_profile(profile: str) -> str:
    if profile not in REPRESENTATION_PROFILES:
        raise ValueError(
            f"Unknown representation profile {profile!r}; "
            f"expected one of {', '.join(REPRESENTATION_PROFILES)}"
        )
    return profile


def _rounded(value: Any, digits: int = 3) -> Any:
    return None if value is None else round(float(value), digits)


def _actor_kinematics(vehicle: VehicleState) -> Dict[str, Any]:
    return {
        "vehicle_id": vehicle.vehicle_id,
        "lane_index": int(vehicle.lane_index),
        "relative_longitudinal_m": _rounded(vehicle.relative_longitudinal_m),
        "relative_lateral_m": _rounded(vehicle.relative_lateral_m),
        "speed_mps": _rounded(vehicle.speed_mps),
        "relative_speed_mps": _rounded(vehicle.relative_speed_mps),
        "acceleration_mps2": _rounded(vehicle.acceleration_mps2),
    }


def _relation(vehicle: VehicleState, av_lane: int) -> Dict[str, Any]:
    lane_delta = int(vehicle.lane_index) - int(av_lane)
    if lane_delta == 0:
        lateral_relation = "same_lane"
    elif lane_delta < 0:
        lateral_relation = f"left_{abs(lane_delta)}"
    else:
        lateral_relation = f"right_{lane_delta}"

    gap = float(vehicle.relative_longitudinal_m)
    if gap > 0.5:
        longitudinal_relation = "ahead"
        closing_speed = -float(vehicle.relative_speed_mps)
    elif gap < -0.5:
        longitudinal_relation = "behind"
        closing_speed = float(vehicle.relative_speed_mps)
    else:
        longitudinal_relation = "alongside"
        closing_speed = abs(float(vehicle.relative_speed_mps))

    return {
        **_actor_kinematics(vehicle),
        "relative_lane": lateral_relation,
        "longitudinal_relation": longitudinal_relation,
        "semantic_slot": f"{lateral_relation}_{longitudinal_relation}",
        "closing_speed_mps": _rounded(closing_speed),
        "is_closing": closing_speed > 0.0,
    }


def _base(state: ScenarioState) -> Dict[str, Any]:
    return {
        "road": {
            "lane_count": int(state.lane_count),
            "speed_limit_mps": _rounded(state.speed_limit_mps),
            "coordinate_system": "AV-centered; longitudinal +forward; lane index increases right",
            "units": "SI",
        },
        "available_actions": [action.to_dict() for action in state.available_actions],
    }


def _kinematic(state: ScenarioState) -> Dict[str, Any]:
    result = _base(state)
    result.update(
        {
            "autonomous_vehicle": _actor_kinematics(state.autonomous_vehicle),
            "controlled_background_vehicle": _actor_kinematics(
                state.controlled_background_vehicle
            ),
            "nearby_vehicles": [
                _actor_kinematics(vehicle) for vehicle in state.nearby_vehicles
            ],
        }
    )
    return result


def _relational(state: ScenarioState) -> Dict[str, Any]:
    result = _base(state)
    av = state.autonomous_vehicle
    related = [_relation(vehicle, av.lane_index) for vehicle in state.nearby_vehicles]
    related.sort(
        key=lambda item: (
            item["relative_lane"],
            abs(float(item["relative_longitudinal_m"])),
            item["vehicle_id"],
        )
    )
    result.update(
        {
            "autonomous_vehicle": {
                "lane_index": int(av.lane_index),
                "speed_mps": _rounded(av.speed_mps),
                "acceleration_mps2": _rounded(av.acceleration_mps2),
            },
            "controlled_background_vehicle": _relation(
                state.controlled_background_vehicle, av.lane_index
            ),
            "nearby_vehicles": related,
        }
    )
    return result


def _risk_aware(state: ScenarioState) -> Dict[str, Any]:
    result = _relational(state)
    risk = state.risk.normalized()
    result["risk"] = {
        "ttc_s": risk.ttc_s,
        "time_headway_s": risk.time_headway_s,
        "predicted_min_distance_m": risk.predicted_min_distance_m,
        "lane_conflict": risk.lane_conflict,
        "nade_criticality": risk.nade_criticality,
    }
    result["recent_history"] = dict(sorted(state.recent_history.items()))
    return result


def _nade_native_actor(
    vehicle: VehicleState,
    controlled_bv: VehicleState,
    *,
    is_autonomous_vehicle: bool,
) -> Dict[str, Any]:
    """Render one actor in the controlled-BV coordinate frame.

    Public NADE's ``_get_veh_obs`` exposes vehicle objects in six fixed
    front/rear lane slots.  This adapter preserves those observable
    kinematics while deliberately omitting identifiers and derived risk
    features.
    """

    return {
        "is_autonomous_vehicle": bool(is_autonomous_vehicle),
        "lane_index": int(vehicle.lane_index),
        "lane_offset_from_controlled_bv": int(vehicle.lane_index)
        - int(controlled_bv.lane_index),
        "relative_longitudinal_to_controlled_bv_m": _rounded(
            float(vehicle.relative_longitudinal_m)
            - float(controlled_bv.relative_longitudinal_m)
        ),
        "speed_mps": _rounded(vehicle.speed_mps),
        "relative_speed_to_controlled_bv_mps": _rounded(
            float(vehicle.speed_mps) - float(controlled_bv.speed_mps)
        ),
    }


def _nade_native_slot_selection(
    state: ScenarioState, observation_range_m: float = 120.0
) -> Dict[str, Tuple[VehicleState, bool]]:
    slot_order = (
        "front_same_lane",
        "rear_same_lane",
        "front_left_lane",
        "rear_left_lane",
        "front_right_lane",
        "rear_right_lane",
    )
    if state.nade_native_observation is not None:
        if len(state.nade_native_observation) != len(slot_order):
            raise ValueError("NADE native observation must contain six slots")
        return {
            slot: (vehicle, vehicle.role == "autonomous_vehicle")
            for slot, vehicle in zip(slot_order, state.nade_native_observation)
            if vehicle is not None
        }

    controlled = state.controlled_background_vehicle
    slot_names = {
        (0, "front"): "front_same_lane",
        (0, "rear"): "rear_same_lane",
        (-1, "front"): "front_left_lane",
        (-1, "rear"): "rear_left_lane",
        (1, "front"): "front_right_lane",
        (1, "rear"): "rear_right_lane",
    }
    selected: Dict[str, Tuple[float, VehicleState, bool]] = {}
    actors = [(state.autonomous_vehicle, True)] + [
        (vehicle, False) for vehicle in state.nearby_vehicles
    ]
    for vehicle, is_av in actors:
        lane_offset = int(vehicle.lane_index) - int(controlled.lane_index)
        if lane_offset not in {-1, 0, 1}:
            continue
        gap = float(vehicle.relative_longitudinal_m) - float(
            controlled.relative_longitudinal_m
        )
        if abs(gap) >= observation_range_m or abs(gap) < 1e-9:
            continue
        direction = "front" if gap > 0.0 else "rear"
        slot = slot_names[(lane_offset, direction)]
        candidate = (abs(gap), vehicle, is_av)
        current = selected.get(slot)
        if current is None or candidate[0] < current[0]:
            selected[slot] = candidate

    result: Dict[str, Tuple[VehicleState, bool]] = {}
    for slot in slot_names.values():
        entry = selected.get(slot)
        if entry is not None:
            _, vehicle, is_av = entry
            result[slot] = (vehicle, is_av)
    return result


def _nade_native(state: ScenarioState, observation_range_m: float = 120.0) -> Dict[str, Any]:
    """Approximate the public NADE controller's raw six-slot observation.

    The source state is AV-centred, so positions are translated into the
    controlled background vehicle's frame before assigning the nearest front
    and rear actor in the same, left, and right lanes.  No interaction graph,
    intent, temporal history, TTC, criticality, or rollout outcome is added.
    """

    controlled = state.controlled_background_vehicle
    selected = _nade_native_slot_selection(state, observation_range_m)
    slot_order = (
        "front_same_lane",
        "rear_same_lane",
        "front_left_lane",
        "rear_left_lane",
        "front_right_lane",
        "rear_right_lane",
    )
    slots: Dict[str, Any] = {}
    for slot in slot_order:
        entry = selected.get(slot)
        slots[slot] = (
            _nade_native_actor(
                entry[0], controlled, is_autonomous_vehicle=entry[1]
            )
            if entry is not None
            else None
        )
    return {
        "source_contract": "public-NADE controlled-BV six-slot observation",
        "road": {
            "lane_count": int(state.lane_count),
            "observation_range_m": _rounded(observation_range_m),
            "coordinate_system": (
                "controlled-BV-centered; longitudinal +forward; "
                "lane index increases right"
            ),
            "units": "SI",
        },
        "controlled_background_vehicle": {
            "lane_index": int(controlled.lane_index),
            "speed_mps": _rounded(controlled.speed_mps),
        },
        "autonomous_vehicle": _nade_native_actor(
            state.autonomous_vehicle,
            controlled,
            is_autonomous_vehicle=True,
        ),
        "surrounding_vehicle_slots": slots,
        "available_actions": [action.to_dict() for action in state.available_actions],
    }


def _temporal_context(state: ScenarioState) -> Dict[str, Any]:
    history = state.recent_history
    available = bool(history.get("history_available", False))
    result: Dict[str, Any] = {
        "history_available": available,
        "observation_interval_s": 1.0,
    }
    if not available:
        return result

    current_gap = float(state.controlled_background_vehicle.relative_longitudinal_m)
    previous_gap = float(history["previous_relative_longitudinal_m"])
    absolute_gap_change = abs(current_gap) - abs(previous_gap)
    if absolute_gap_change < -0.05:
        distance_trend = "closing"
    elif absolute_gap_change > 0.05:
        distance_trend = "opening"
    else:
        distance_trend = "stable"
    previous_lane = int(history["previous_lane_index"])
    current_lane = int(state.controlled_background_vehicle.lane_index)
    result.update(
        {
            "previous_relative_longitudinal_m": _rounded(previous_gap),
            "signed_gap_change_m": _rounded(
                history.get("relative_longitudinal_change_m", current_gap - previous_gap)
            ),
            "absolute_gap_change_m": _rounded(absolute_gap_change),
            "distance_trend": distance_trend,
            "previous_relative_speed_mps": _rounded(
                history["previous_relative_speed_mps"]
            ),
            "relative_speed_change_mps": _rounded(
                history.get("relative_speed_change_mps", 0.0)
            ),
            "estimated_relative_acceleration_mps2": _rounded(
                history.get("relative_speed_change_mps", 0.0)
            ),
            "previous_lane_index": previous_lane,
            "current_lane_index": current_lane,
            "lane_changed": bool(history.get("lane_changed", previous_lane != current_lane)),
            "lane_transition": f"{previous_lane}->{current_lane}",
        }
    )
    return result


def _temporal_relational(state: ScenarioState) -> Dict[str, Any]:
    result = _relational(state)
    result["temporal_context"] = _temporal_context(state)
    return result


def _temporal_action_conditioned(state: ScenarioState) -> Dict[str, Any]:
    result = _temporal_relational(state)
    risk = state.risk.normalized()
    result["risk"] = {
        "ttc_s": risk.ttc_s,
        "time_headway_s": risk.time_headway_s,
        "predicted_min_distance_m": risk.predicted_min_distance_m,
        "lane_conflict": risk.lane_conflict,
    }
    result["candidate_action_assessments"] = [
        _action_assessment(state, action, horizon_s=3.0)
        for action in state.available_actions
    ]
    return result


def _hierarchical_relational(state: ScenarioState) -> Dict[str, Any]:
    result = _relational(state)
    # The flat list is deliberately replaced, not duplicated: the experimental
    # treatment is a maneuver-first action view with the exact same NADE leaves.
    result.pop("available_actions", None)
    result["action_hierarchy"] = hierarchical_action_structure(
        state.available_actions
    )
    lateral_actions = [
        action for action in state.available_actions if action.kind == "lateral"
    ]
    result["lateral_maneuver_context"] = [
        _action_assessment(state, action, horizon_s=3.0)
        for action in lateral_actions
    ]
    return result


def _lane_clearance_evidence(
    state: ScenarioState,
    target_lane: int,
    horizon_s: float,
    *,
    include_av: bool = True,
) -> Dict[str, Any]:
    bv = state.controlled_background_vehicle
    if target_lane < 0 or target_lane >= state.lane_count:
        return {
            "current_front_clearance_m": None,
            "current_rear_clearance_m": None,
            "projected_front_clearance_m": None,
            "projected_rear_clearance_m": None,
            "vehicles_in_target_lane": 0,
        }
    current_gaps = []
    projected_gaps = []
    traffic = [*state.nearby_vehicles]
    if include_av:
        traffic.insert(0, state.autonomous_vehicle)
    for vehicle in traffic:
        if int(vehicle.lane_index) != target_lane:
            continue
        current_gap = float(vehicle.relative_longitudinal_m) - float(
            bv.relative_longitudinal_m
        )
        relative_speed_to_bv = float(vehicle.relative_speed_mps) - float(
            bv.relative_speed_mps
        )
        current_gaps.append(current_gap)
        projected_gaps.append(current_gap + relative_speed_to_bv * horizon_s)

    def clearances(gaps: List[float]) -> Tuple[Any, Any]:
        front = min((gap for gap in gaps if gap >= 0.0), default=None)
        rear = max((gap for gap in gaps if gap < 0.0), default=None)
        return _rounded(front), _rounded(abs(rear) if rear is not None else None)

    current_front, current_rear = clearances(current_gaps)
    projected_front, projected_rear = clearances(projected_gaps)
    return {
        "current_front_clearance_m": current_front,
        "current_rear_clearance_m": current_rear,
        "projected_front_clearance_m": projected_front,
        "projected_rear_clearance_m": projected_rear,
        "vehicles_in_target_lane": len(current_gaps),
    }


def _maneuver_evidence_cards(
    state: ScenarioState, horizon_s: float = 3.0
) -> List[Dict[str, Any]]:
    actions_by_maneuver: Dict[str, List[str]] = {
        KEEP_LANE: [],
        CHANGE_LEFT: [],
        CHANGE_RIGHT: [],
    }
    for action in state.available_actions:
        actions_by_maneuver[maneuver_for_action(action.value)].append(action.value)

    bv = state.controlled_background_vehicle
    av = state.autonomous_vehicle
    lane_offsets = {KEEP_LANE: 0, CHANGE_LEFT: -1, CHANGE_RIGHT: 1}
    cards = []
    for maneuver in (KEEP_LANE, CHANGE_LEFT, CHANGE_RIGHT):
        lane_offset = lane_offsets[maneuver]
        target_lane = int(bv.lane_index) + lane_offset
        within_road = 0 <= target_lane < state.lane_count
        available = bool(actions_by_maneuver[maneuver])
        projected_gap = float(bv.relative_longitudinal_m) + float(
            bv.relative_speed_mps
        ) * horizon_s
        minimum_gap, time_of_minimum = _sampled_minimum_abs_gap(
            float(bv.relative_longitudinal_m),
            float(bv.relative_speed_mps),
            0.0,
            horizon_s,
        )
        cards.append(
            {
                "maneuver": maneuver,
                "available": available,
                "availability_reason": (
                    "supported_by_current_action_set"
                    if available
                    else "not_supported_by_current_action_set"
                ),
                "leaf_actions": actions_by_maneuver[maneuver],
                "target_lane_index": target_lane,
                "target_lane_within_road": within_road,
                "target_lane_is_av_lane": within_road
                and target_lane == int(av.lane_index),
                "projection_model": "constant_velocity_no_lane_response",
                "projection_horizon_s": horizon_s,
                "av_interaction_evidence": {
                    "current_signed_longitudinal_gap_m": _rounded(
                        bv.relative_longitudinal_m
                    ),
                    "current_relative_speed_mps": _rounded(bv.relative_speed_mps),
                    "projected_signed_longitudinal_gap_m": _rounded(projected_gap),
                    "predicted_min_abs_longitudinal_gap_m": _rounded(minimum_gap),
                    "time_of_predicted_min_gap_s": _rounded(time_of_minimum),
                    "shares_target_lane_with_av": within_road
                    and target_lane == int(av.lane_index),
                },
                "target_lane_traffic_evidence": _lane_clearance_evidence(
                    state, target_lane, horizon_s
                ),
            }
        )
    return cards


def _symmetric_maneuver_evidence(state: ScenarioState) -> Dict[str, Any]:
    result = _temporal_relational(state)
    result.pop("available_actions", None)
    result["action_hierarchy"] = hierarchical_action_structure(
        state.available_actions
    )
    result["maneuver_evidence_assumptions"] = {
        "purpose": "Compare maneuver-level interaction evidence before selecting an exact leaf action.",
        "projection_is_not_rollout_label": True,
        "projection_model": "constant_velocity_no_lane_response",
        "projection_horizon_s": 3.0,
    }
    result["maneuver_evidence_cards"] = _maneuver_evidence_cards(state, horizon_s=3.0)
    return result


def _maneuver_evidence_cards_v2(
    state: ScenarioState, horizon_s: float = 3.0
) -> List[Dict[str, Any]]:
    """Build label-free maneuver cards that separate geometry from conflict."""

    actions_by_maneuver: Dict[str, List[str]] = {
        KEEP_LANE: [],
        CHANGE_LEFT: [],
        CHANGE_RIGHT: [],
    }
    for action in state.available_actions:
        actions_by_maneuver[maneuver_for_action(action.value)].append(action.value)

    bv = state.controlled_background_vehicle
    av = state.autonomous_vehicle
    current_gap = float(bv.relative_longitudinal_m)
    relative_speed = float(bv.relative_speed_mps)
    projected_gap = current_gap + relative_speed * horizon_s
    minimum_gap, time_of_minimum = _sampled_minimum_abs_gap(
        current_gap, relative_speed, 0.0, horizon_s
    )
    overlap_threshold_m = 5.0
    longitudinal_overlap = minimum_gap <= overlap_threshold_m
    lane_offsets = {KEEP_LANE: 0, CHANGE_LEFT: -1, CHANGE_RIGHT: 1}
    cards = []
    for maneuver in (KEEP_LANE, CHANGE_LEFT, CHANGE_RIGHT):
        target_lane = int(bv.lane_index) + lane_offsets[maneuver]
        within_road = 0 <= target_lane < state.lane_count
        available = bool(actions_by_maneuver[maneuver])
        same_as_av = within_road and target_lane == int(av.lane_index)
        av_reachable = within_road and abs(target_lane - int(av.lane_index)) <= 1
        cards.append(
            {
                "maneuver": maneuver,
                "available": available,
                "availability_reason": (
                    "supported_by_current_action_set"
                    if available
                    else "not_supported_by_current_action_set"
                ),
                "leaf_actions": actions_by_maneuver[maneuver],
                "target_lane_index": target_lane,
                "target_lane_within_road": within_road,
                "projection_horizon_s": horizon_s,
                "longitudinal_alignment_evidence": {
                    "current_signed_gap_m": _rounded(current_gap),
                    "current_relative_speed_mps": _rounded(relative_speed),
                    "projected_signed_gap_m": _rounded(projected_gap),
                    "predicted_min_abs_gap_m": _rounded(minimum_gap),
                    "time_of_predicted_min_gap_s": _rounded(time_of_minimum),
                },
                "av_lane_conflict_evidence": {
                    "av_current_lane_index": int(av.lane_index),
                    "target_lane_is_av_current_lane": same_as_av,
                    "av_can_reach_target_lane_with_one_lane_change": av_reachable,
                    "longitudinal_overlap_threshold_m": overlap_threshold_m,
                    "longitudinal_overlap_within_horizon": longitudinal_overlap,
                    "direct_conflict_under_constant_lane_projection": bool(
                        same_as_av and longitudinal_overlap
                    ),
                    "potential_conflict_if_av_changes_one_lane": bool(
                        not same_as_av and av_reachable and longitudinal_overlap
                    ),
                },
                "non_av_target_lane_traffic_evidence": _lane_clearance_evidence(
                    state, target_lane, horizon_s, include_av=False
                ),
            }
        )
    return cards


def _symmetric_maneuver_evidence_v2(state: ScenarioState) -> Dict[str, Any]:
    relational = _temporal_relational(state)
    # Cards already summarize target-lane traffic; avoid duplicating the full list.
    relational.pop("available_actions", None)
    relational.pop("nearby_vehicles", None)
    relational["action_hierarchy"] = hierarchical_action_structure(
        state.available_actions
    )
    relational["maneuver_evidence_assumptions"] = {
        "purpose": "Compare label-free maneuver-level interaction evidence before selecting an exact leaf action.",
        "projection_is_not_rollout_label": True,
        "projection_model": "constant_velocity_with_av_one_lane_reachability_envelope",
        "projection_horizon_s": 3.0,
        "other_vehicles_do_not_respond_in_projection": True,
    }
    relational["maneuver_evidence_cards"] = _maneuver_evidence_cards_v2(
        state, horizon_s=3.0
    )
    return relational


def _sampled_minimum_abs_gap(
    gap_m: float,
    relative_speed_mps: float,
    acceleration_mps2: float,
    horizon_s: float,
) -> Tuple[float, float]:
    samples = 8
    values: List[Tuple[float, float]] = []
    for index in range(samples + 1):
        time_s = horizon_s * index / samples
        gap = gap_m + relative_speed_mps * time_s + 0.5 * acceleration_mps2 * time_s**2
        values.append((abs(gap), time_s))
    return min(values, key=lambda item: item[0])


def _target_lane_clearance(
    state: ScenarioState, target_lane: int, horizon_s: float
) -> Tuple[Any, Any]:
    projected = []
    for vehicle in state.nearby_vehicles:
        if vehicle.lane_index != target_lane:
            continue
        gap = vehicle.relative_longitudinal_m + vehicle.relative_speed_mps * horizon_s
        projected.append(float(gap))
    front = min((gap for gap in projected if gap >= 0.0), default=None)
    rear = max((gap for gap in projected if gap < 0.0), default=None)
    return _rounded(front), _rounded(abs(rear) if rear is not None else None)


def _action_assessment(
    state: ScenarioState, action: ActionOption, horizon_s: float = 2.0
) -> Dict[str, Any]:
    bv = state.controlled_background_vehicle
    av_lane = state.autonomous_vehicle.lane_index
    assessment: Dict[str, Any] = {
        "action": action.value,
        "kind": action.kind,
        "horizon_s": horizon_s,
        "feasible": True,
    }
    if action.kind == "longitudinal":
        acceleration = float(action.acceleration_mps2 or 0.0)
        future_gap = (
            bv.relative_longitudinal_m
            + bv.relative_speed_mps * horizon_s
            + 0.5 * acceleration * horizon_s**2
        )
        min_gap, time_of_min = _sampled_minimum_abs_gap(
            bv.relative_longitudinal_m,
            bv.relative_speed_mps,
            acceleration,
            horizon_s,
        )
        assessment.update(
            {
                "acceleration_mps2": acceleration,
                "predicted_speed_mps": _rounded(max(0.0, bv.speed_mps + acceleration * horizon_s)),
                "predicted_relative_longitudinal_m": _rounded(future_gap),
                "predicted_min_abs_gap_m": _rounded(min_gap),
                "time_of_min_gap_s": _rounded(time_of_min),
                "same_lane_as_av": bv.lane_index == av_lane,
            }
        )
        return assessment

    lane_change = -1 if action.value == "LANE_LEFT" else 1
    target_lane = int(bv.lane_index) + lane_change
    feasible = 0 <= target_lane < state.lane_count
    front_gap, rear_gap = _target_lane_clearance(state, target_lane, horizon_s)
    assessment.update(
        {
            "target_lane_index": target_lane,
            "feasible": feasible,
            "target_lane_front_gap_m": front_gap,
            "target_lane_rear_gap_m": rear_gap,
            "target_lane_is_av_lane": target_lane == av_lane,
        }
    )
    return assessment


def render_state(state: ScenarioState, profile: str = RISK_AWARE_V1) -> Dict[str, Any]:
    validate_profile(profile)
    if profile == FULL_STATE_V1:
        result = state.to_dict()
    elif profile == KINEMATIC_V1:
        result = _kinematic(state)
    elif profile == RELATIONAL_V1:
        result = _relational(state)
    elif profile == TEMPORAL_RELATIONAL_V1:
        result = _temporal_relational(state)
    elif profile == TEMPORAL_ACTION_CONDITIONED_V1:
        result = _temporal_action_conditioned(state)
    elif profile == HIERARCHICAL_RELATIONAL_V1:
        result = _hierarchical_relational(state)
    elif profile == SYMMETRIC_MANEUVER_EVIDENCE_V1:
        result = _symmetric_maneuver_evidence(state)
    elif profile == SYMMETRIC_MANEUVER_EVIDENCE_V2:
        result = _symmetric_maneuver_evidence_v2(state)
    elif profile == NADE_NATIVE_V1:
        result = _nade_native(state)
    elif profile == INTERACTION_GRAPH_INTENT_V1:
        result = render_interaction_graph_intent(state)
    elif profile == POST_MANEUVER_CONFLICT_V1:
        result = render_post_maneuver_conflict(state)
    elif profile == POST_MANEUVER_CONFLICT_V2:
        result = render_post_maneuver_conflict_v2(state)
    elif profile == STAGE_ROUTED_CONFLICT_V1:
        result = render_stage_routed_conflict(state)
    else:
        result = _risk_aware(state)
        if profile == ACTION_CONDITIONED_V1:
            result["candidate_action_assessments"] = [
                _action_assessment(state, action) for action in state.available_actions
            ]
    result["representation_profile"] = profile
    return result


def canonical_representation(state: ScenarioState, profile: str) -> str:
    return json.dumps(
        render_state(state, profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def profile_summary(state: ScenarioState, profile: str) -> Dict[str, Any]:
    payload = canonical_representation(state, profile)
    return {
        "profile": profile,
        "utf8_bytes": len(payload.encode("utf-8")),
        "characters": len(payload),
    }
