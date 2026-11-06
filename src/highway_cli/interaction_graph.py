from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .actions import ActionOption
from .hierarchical_actions import (
    CHANGE_LEFT,
    CHANGE_RIGHT,
    KEEP_LANE,
    hierarchical_action_structure,
)
from .schema import ScenarioState, VehicleState


INTERACTION_GRAPH_INTENT_V1 = "interaction-graph-intent-v1"
PROJECTION_HORIZON_S = 3.0


def _rounded(value: Any, digits: int = 3) -> Any:
    return None if value is None else round(float(value), digits)


def _minimum_abs_linear_gap(
    gap_m: float, relative_speed_mps: float, horizon_s: float
) -> Tuple[float, float]:
    if abs(relative_speed_mps) < 1e-12:
        return abs(gap_m), 0.0
    crossing = -gap_m / relative_speed_mps
    candidates = [0.0, horizon_s]
    if 0.0 <= crossing <= horizon_s:
        candidates.append(crossing)
    time_s = min(
        candidates,
        key=lambda value: abs(gap_m + relative_speed_mps * value),
    )
    return abs(gap_m + relative_speed_mps * time_s), time_s


def _longitudinal_relation(gap_m: float) -> str:
    if gap_m > 0.5:
        return "ahead"
    if gap_m < -0.5:
        return "behind"
    return "alongside"


def _lane_relation(lane_delta: int) -> str:
    if lane_delta == 0:
        return "same_lane"
    if lane_delta == -1:
        return "adjacent_left"
    if lane_delta == 1:
        return "adjacent_right"
    return "nonadjacent_left" if lane_delta < 0 else "nonadjacent_right"


def _closing_speed(gap_m: float, relative_speed_mps: float) -> float:
    if gap_m > 0.5:
        return -relative_speed_mps
    if gap_m < -0.5:
        return relative_speed_mps
    return abs(relative_speed_mps)


def _actor_node(node_id: str, vehicle: VehicleState, role: str) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": "actor",
        "role": role,
        "lane_index": int(vehicle.lane_index),
        "av_centered_longitudinal_m": _rounded(vehicle.relative_longitudinal_m),
        "speed_mps": _rounded(vehicle.speed_mps),
        "av_relative_speed_mps": _rounded(vehicle.relative_speed_mps),
        "acceleration_mps2": _rounded(vehicle.acceleration_mps2),
    }


def _interaction_edge(
    source_id: str,
    source: VehicleState,
    target_id: str,
    target: VehicleState,
    horizon_s: float,
) -> Dict[str, Any]:
    signed_gap = float(target.relative_longitudinal_m) - float(
        source.relative_longitudinal_m
    )
    relative_speed = float(target.relative_speed_mps) - float(
        source.relative_speed_mps
    )
    lane_delta = int(target.lane_index) - int(source.lane_index)
    minimum_gap, time_of_minimum = _minimum_abs_linear_gap(
        signed_gap, relative_speed, horizon_s
    )
    closing_speed = _closing_speed(signed_gap, relative_speed)
    return {
        "source": source_id,
        "target": target_id,
        "edge_type": "CONTROLLED_BV_INTERACTION",
        "lane_delta": lane_delta,
        "lane_relation": _lane_relation(lane_delta),
        "longitudinal_relation": _longitudinal_relation(signed_gap),
        "signed_longitudinal_gap_m": _rounded(signed_gap),
        "target_relative_speed_to_source_mps": _rounded(relative_speed),
        "closing_speed_mps": _rounded(closing_speed),
        "is_closing": closing_speed > 0.0,
        "constant_velocity_projection": {
            "horizon_s": horizon_s,
            "projected_signed_gap_m": _rounded(
                signed_gap + relative_speed * horizon_s
            ),
            "minimum_abs_gap_m": _rounded(minimum_gap),
            "time_of_minimum_gap_s": _rounded(time_of_minimum),
        },
        "same_lane_now": lane_delta == 0,
        "reachable_by_one_lane_change": abs(lane_delta) <= 1,
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
    change = abs(current_gap) - abs(previous_gap)
    result.update(
        {
            "previous_av_centered_longitudinal_m": _rounded(previous_gap),
            "absolute_gap_change_m": _rounded(change),
            "distance_trend": (
                "closing" if change < -0.05 else "opening" if change > 0.05 else "stable"
            ),
            "previous_av_relative_speed_mps": _rounded(
                history["previous_relative_speed_mps"]
            ),
            "av_relative_speed_change_mps": _rounded(
                history.get("relative_speed_change_mps", 0.0)
            ),
            "previous_lane_index": int(history["previous_lane_index"]),
            "current_lane_index": int(
                state.controlled_background_vehicle.lane_index
            ),
            "lane_changed": bool(history.get("lane_changed", False)),
        }
    )
    return result


def _action_values(actions: Sequence[ActionOption], predicate) -> List[str]:
    return [action.value for action in actions if predicate(action)]


def _intent_catalog(state: ScenarioState) -> List[Dict[str, Any]]:
    actions = state.available_actions
    bv = state.controlled_background_vehicle
    av = state.autonomous_vehicle
    signed_gap = float(bv.relative_longitudinal_m)
    same_lane = int(bv.lane_index) == int(av.lane_index)
    longitudinal = _action_values(actions, lambda action: action.kind == "longitudinal")
    negative = _action_values(
        actions,
        lambda action: action.kind == "longitudinal"
        and float(action.acceleration_mps2 or 0.0) < 0.0,
    )
    positive = _action_values(
        actions,
        lambda action: action.kind == "longitudinal"
        and float(action.acceleration_mps2 or 0.0) > 0.0,
    )
    neutral = _action_values(
        actions,
        lambda action: action.kind == "longitudinal"
        and abs(float(action.acceleration_mps2 or 0.0)) < 1e-12,
    )
    lateral = {
        CHANGE_LEFT: _action_values(actions, lambda action: action.value == "LANE_LEFT"),
        CHANGE_RIGHT: _action_values(
            actions, lambda action: action.value == "LANE_RIGHT"
        ),
    }

    def lateral_intent(intent_id: str, maneuver: str, offset: int) -> Dict[str, Any]:
        target_lane = int(bv.lane_index) + offset
        within_road = 0 <= target_lane < state.lane_count
        return {
            "intent_id": intent_id,
            "purpose": "Create or strengthen a lateral interaction by changing one lane.",
            "available": bool(lateral[maneuver]),
            "compatible_maneuvers": [maneuver] if lateral[maneuver] else [],
            "compatible_leaf_actions": lateral[maneuver],
            "observable_applicability": {
                "target_lane_index": target_lane,
                "target_lane_within_road": within_road,
                "target_lane_is_av_lane": within_road
                and target_lane == int(av.lane_index),
                "av_within_one_lane_of_target": within_road
                and abs(target_lane - int(av.lane_index)) <= 1,
            },
        }

    return [
        {
            "intent_id": "LONGITUDINAL_GAP_SHAPING",
            "purpose": "Use longitudinal control to reduce or maintain interaction distance.",
            "available": bool(longitudinal),
            "compatible_maneuvers": [KEEP_LANE] if longitudinal else [],
            "compatible_leaf_actions": longitudinal,
            "observable_applicability": {
                "same_lane_as_av": same_lane,
                "controlled_bv_longitudinal_relation_to_av": _longitudinal_relation(
                    signed_gap
                ),
                "direct_current_lane_conflict": same_lane,
            },
        },
        {
            "intent_id": "DECELERATION_PRESSURE",
            "purpose": "Decelerate to create pressure when the controlled vehicle leads another actor.",
            "available": bool(negative),
            "compatible_maneuvers": [KEEP_LANE] if negative else [],
            "compatible_leaf_actions": negative,
            "observable_applicability": {
                "same_lane_as_av": same_lane,
                "controlled_bv_ahead_of_av": signed_gap > 0.5,
                "brake_check_geometry_present": same_lane and signed_gap > 0.5,
            },
        },
        {
            "intent_id": "ACCELERATION_PRESSURE",
            "purpose": "Accelerate to create pressure when the controlled vehicle follows another actor.",
            "available": bool(positive),
            "compatible_maneuvers": [KEEP_LANE] if positive else [],
            "compatible_leaf_actions": positive,
            "observable_applicability": {
                "same_lane_as_av": same_lane,
                "controlled_bv_behind_av": signed_gap < -0.5,
                "rear_closing_geometry_present": same_lane and signed_gap < -0.5,
            },
        },
        {
            "intent_id": "MAINTAIN_CURRENT_PRESSURE",
            "purpose": "Preserve the current longitudinal interaction without acceleration.",
            "available": bool(neutral),
            "compatible_maneuvers": [KEEP_LANE] if neutral else [],
            "compatible_leaf_actions": neutral,
            "observable_applicability": {
                "same_lane_as_av": same_lane,
                "controlled_bv_longitudinal_relation_to_av": _longitudinal_relation(
                    signed_gap
                ),
            },
        },
        lateral_intent("LEFT_LANE_INTERACTION", CHANGE_LEFT, -1),
        lateral_intent("RIGHT_LANE_INTERACTION", CHANGE_RIGHT, 1),
    ]


def render_interaction_graph_intent(state: ScenarioState) -> Dict[str, Any]:
    av = state.autonomous_vehicle
    bv = state.controlled_background_vehicle
    ordered_neighbors = sorted(
        state.nearby_vehicles,
        key=lambda item: (
            int(item.lane_index),
            float(item.relative_longitudinal_m),
            float(item.speed_mps),
        ),
    )
    actor_nodes = [
        _actor_node("AV", av, "autonomous_vehicle"),
        _actor_node("CONTROLLED_BV", bv, "controlled_background_vehicle"),
    ]
    neighbor_pairs = []
    for index, vehicle in enumerate(ordered_neighbors):
        node_id = f"NEIGHBOR_{index}"
        actor_nodes.append(_actor_node(node_id, vehicle, "neighbor"))
        neighbor_pairs.append((node_id, vehicle))

    lane_nodes = [
        {
            "node_id": f"LANE_{lane}",
            "node_type": "lane",
            "lane_index": lane,
            "is_av_lane": lane == int(av.lane_index),
            "is_controlled_bv_lane": lane == int(bv.lane_index),
        }
        for lane in range(state.lane_count)
    ]
    lane_edges = []
    for lane in range(state.lane_count - 1):
        lane_edges.extend(
            [
                {
                    "source": f"LANE_{lane}",
                    "target": f"LANE_{lane + 1}",
                    "edge_type": "RIGHT_ADJACENT_LANE",
                },
                {
                    "source": f"LANE_{lane + 1}",
                    "target": f"LANE_{lane}",
                    "edge_type": "LEFT_ADJACENT_LANE",
                },
            ]
        )
    interactions = [
        _interaction_edge("CONTROLLED_BV", bv, "AV", av, PROJECTION_HORIZON_S)
    ]
    interactions.extend(
        _interaction_edge(
            "CONTROLLED_BV", bv, node_id, vehicle, PROJECTION_HORIZON_S
        )
        for node_id, vehicle in neighbor_pairs
    )
    interactions.sort(
        key=lambda item: (
            abs(int(item["lane_delta"])),
            abs(float(item["signed_longitudinal_gap_m"])),
            str(item["target"]),
        )
    )
    return {
        "road": {
            "lane_count": int(state.lane_count),
            "speed_limit_mps": _rounded(state.speed_limit_mps),
            "coordinate_system": "AV-centered; longitudinal +forward; lane index increases right",
            "units": "SI",
        },
        "scene_graph": {
            "graph_scope": (
                "All observed actors are nodes; interaction edges are centered on the "
                "controlled background vehicle because it is the decision actor."
            ),
            "actor_nodes": actor_nodes,
            "lane_nodes": lane_nodes,
            "lane_edges": lane_edges,
            "controlled_bv_interaction_edges": interactions,
        },
        "temporal_context": _temporal_context(state),
        "intent_interface": {
            "intents_are_semantic_hypotheses_not_rollout_labels": True,
            "intent_candidates": _intent_catalog(state),
            "all_final_actions_remain_original_nade_actions": True,
        },
        "action_hierarchy": hierarchical_action_structure(state.available_actions),
        "projection_assumptions": {
            "projection_is_not_simulator_rollout": True,
            "model": "constant_velocity_no_response",
            "horizon_s": PROJECTION_HORIZON_S,
        },
    }


def audit_interaction_graph_intent(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    details = []
    actor_counts: List[int] = []
    edge_counts: List[int] = []
    intent_counts: Counter[str] = Counter()
    graph_reference_failures = 0
    actor_coverage_failures = 0
    action_coverage_failures = 0
    raw_vehicle_id_occurrences = 0

    for record in records:
        state = ScenarioState.from_dict(dict(record["state"]))
        payload = render_interaction_graph_intent(state)
        graph = payload["scene_graph"]
        actor_nodes = graph["actor_nodes"]
        lane_nodes = graph["lane_nodes"]
        node_ids = {
            item["node_id"] for item in [*actor_nodes, *lane_nodes]
        }
        edges = [
            *graph["lane_edges"],
            *graph["controlled_bv_interaction_edges"],
        ]
        references_valid = all(
            edge["source"] in node_ids and edge["target"] in node_ids
            for edge in edges
        )
        expected_actors = 2 + len(state.nearby_vehicles)
        actor_coverage = len(actor_nodes) == expected_actors
        original_actions = {action.value for action in state.available_actions}
        intent_actions = {
            action
            for intent in payload["intent_interface"]["intent_candidates"]
            for action in intent["compatible_leaf_actions"]
        }
        action_coverage = intent_actions == original_actions
        rendered = str(payload)
        raw_ids = {
            state.autonomous_vehicle.vehicle_id,
            state.controlled_background_vehicle.vehicle_id,
            *(vehicle.vehicle_id for vehicle in state.nearby_vehicles),
        }
        raw_id_found = any(
            vehicle_id not in {"AV"} and vehicle_id in rendered
            for vehicle_id in raw_ids
        )
        for intent in payload["intent_interface"]["intent_candidates"]:
            if intent["available"]:
                intent_counts[str(intent["intent_id"])] += 1
        actor_counts.append(len(actor_nodes))
        edge_counts.append(len(edges))
        graph_reference_failures += int(not references_valid)
        actor_coverage_failures += int(not actor_coverage)
        action_coverage_failures += int(not action_coverage)
        raw_vehicle_id_occurrences += int(raw_id_found)
        details.append(
            {
                "episode_id": state.episode_id,
                "decision_step": state.decision_step,
                "controlled_vehicle_id": state.controlled_background_vehicle.vehicle_id,
                "actor_nodes": len(actor_nodes),
                "expected_actor_nodes": expected_actors,
                "edges": len(edges),
                "graph_references_valid": references_valid,
                "all_observed_actors_represented": actor_coverage,
                "intent_union_covers_original_actions": action_coverage,
                "raw_vehicle_identifier_visible": raw_id_found,
            }
        )

    count = len(details)
    return {
        "audit_version": "interaction-graph-intent-audit-v1",
        "status": "offline_structure_audit_not_model_evaluation",
        "examples": count,
        "graph_reference_failure_examples": graph_reference_failures,
        "actor_coverage_failure_examples": actor_coverage_failures,
        "action_coverage_failure_examples": action_coverage_failures,
        "raw_vehicle_identifier_examples": raw_vehicle_id_occurrences,
        "mean_actor_nodes": mean(actor_counts) if actor_counts else None,
        "mean_edges": mean(edge_counts) if edge_counts else None,
        "available_intent_counts": dict(sorted(intent_counts.items())),
        "passes_structural_audit": not any(
            (
                graph_reference_failures,
                actor_coverage_failures,
                action_coverage_failures,
                raw_vehicle_id_occurrences,
            )
        ),
        "details": details,
    }
