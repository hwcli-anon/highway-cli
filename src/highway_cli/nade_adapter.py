from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from .actions import nade_action_space
from .representations import _nade_native_slot_selection
from .schema import RiskFeatures, ScenarioState, VehicleState


NADE_NATIVE_SLOT_ORDER = (
    "front_same_lane",
    "rear_same_lane",
    "front_left_lane",
    "rear_left_lane",
    "front_right_lane",
    "rear_right_lane",
)


def _lane(vehicle: Any) -> int:
    lane_index = getattr(vehicle, "lane_index", 0)
    if isinstance(lane_index, (tuple, list)):
        return int(lane_index[-1])
    return int(lane_index)


def _position(vehicle: Any) -> tuple:
    position = getattr(vehicle, "position", (0.0, 0.0))
    return float(position[0]), float(position[1])


def _acceleration(vehicle: Any) -> Optional[float]:
    action = getattr(vehicle, "action", None)
    if isinstance(action, dict) and "acceleration" in action:
        return float(action["acceleration"])
    return None


def _vehicle_state(
    vehicle: Any,
    av: Any,
    role: str,
    vehicle_ids: Optional[Dict[int, str]] = None,
) -> VehicleState:
    x, y = _position(vehicle)
    av_x, av_y = _position(av)
    speed = float(getattr(vehicle, "velocity", 0.0))
    av_speed = float(getattr(av, "velocity", 0.0))
    identifier = "AV" if vehicle is av else None
    if identifier is None and vehicle_ids is not None:
        identifier = vehicle_ids.get(id(vehicle))
    if identifier is None:
        identifier = str(getattr(vehicle, "id", id(vehicle)))
    return VehicleState(
        vehicle_id=identifier,
        role=role,
        lane_index=_lane(vehicle),
        relative_longitudinal_m=x - av_x,
        relative_lateral_m=y - av_y,
        speed_mps=speed,
        relative_speed_mps=speed - av_speed,
        acceleration_mps2=_acceleration(vehicle),
    )


def _risk(av: Any, controlled_bv: Any, criticality: Optional[float]) -> RiskFeatures:
    av_x, _ = _position(av)
    bv_x, _ = _position(controlled_bv)
    av_speed = float(getattr(av, "velocity", 0.0))
    bv_speed = float(getattr(controlled_bv, "velocity", 0.0))
    gap = bv_x - av_x
    relative_speed = bv_speed - av_speed
    same_lane = _lane(av) == _lane(controlled_bv)
    ttc = None
    if same_lane and gap * relative_speed < 0 and abs(relative_speed) > 1e-9:
        ttc = abs(gap / relative_speed)
    headway = None
    if same_lane and gap > 0 and av_speed > 1e-9:
        headway = gap / av_speed
    predicted_min_distance = None
    if same_lane:
        horizon_s = 3.0
        candidates = [abs(gap), abs(gap + relative_speed * horizon_s)]
        if abs(relative_speed) > 1e-9:
            zero_crossing = -gap / relative_speed
            if 0.0 <= zero_crossing <= horizon_s:
                candidates.append(0.0)
        predicted_min_distance = min(candidates)
    return RiskFeatures(
        ttc_s=ttc,
        time_headway_s=headway,
        predicted_min_distance_m=predicted_min_distance,
        lane_conflict=same_lane,
        nade_criticality=criticality,
    )


def encode_nade_state(
    env: Any,
    controlled_bv: Any,
    *,
    episode_id: str,
    decision_step: int,
    criticality: Optional[float] = None,
    recent_history: Optional[Dict[str, Any]] = None,
    nearby_limit: int = 10,
    vehicle_ids: Optional[Dict[int, str]] = None,
    native_observation: Optional[Iterable[Any]] = None,
) -> ScenarioState:
    """Convert a live public-NADE environment state to highway-state-v1.

    The encoder deliberately depends on the NADE object's public attributes
    rather than importing the legacy package, so it can be unit-tested without
    loading the multi-gigabyte NADE tables.
    """

    av = env.vehicle
    av_x, av_y = _position(av)
    neighbors = [
        vehicle
        for vehicle in getattr(env.road, "vehicles", [])
        if vehicle is not av and vehicle is not controlled_bv
    ]
    neighbors.sort(
        key=lambda vehicle: math.hypot(
            _position(vehicle)[0] - av_x, _position(vehicle)[1] - av_y
        )
    )
    nearby = [
        _vehicle_state(vehicle, av, "neighbor", vehicle_ids)
        for vehicle in neighbors[:nearby_limit]
    ]
    lane_count = int(getattr(env, "max_lane", 2) - getattr(env, "min_lane", 0) + 1)
    encoded_native_observation = None
    if native_observation is not None:
        raw_native_observation = list(native_observation)
        if len(raw_native_observation) != len(NADE_NATIVE_SLOT_ORDER):
            raise ValueError("NADE native observation must contain six slots")
        encoded_native_observation = [
            (
                _vehicle_state(
                    vehicle,
                    av,
                    "autonomous_vehicle" if vehicle is av else "native_neighbor",
                    vehicle_ids,
                )
                if vehicle is not None
                else None
            )
            for vehicle in raw_native_observation
        ]
    return ScenarioState(
        episode_id=episode_id,
        decision_step=decision_step,
        simulation_time_s=float(getattr(env, "time", decision_step)),
        lane_count=lane_count,
        speed_limit_mps=float(getattr(env, "max_velocity", 40.0)),
        autonomous_vehicle=_vehicle_state(av, av, "autonomous_vehicle", vehicle_ids),
        controlled_background_vehicle=_vehicle_state(
            controlled_bv, av, "controlled_background_vehicle", vehicle_ids
        ),
        nearby_vehicles=nearby,
        risk=_risk(av, controlled_bv, criticality),
        recent_history=recent_history or {},
        available_actions=[
            action
            for action in nade_action_space()
            if not (
                (action.value == "LANE_LEFT" and _lane(controlled_bv) <= 0)
                or (
                    action.value == "LANE_RIGHT"
                    and _lane(controlled_bv) >= int(getattr(env, "max_lane", 2))
                )
            )
        ],
        nade_native_observation=encoded_native_observation,
    )


def audit_nade_native_slots(
    state: ScenarioState,
    raw_observation: Iterable[Any],
    *,
    vehicle_ids: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compare reconstructed slots with public NADE's live `_get_veh_obs`.

    Identifiers are used only for this offline fidelity check and never enter
    the provider-facing representation.
    """

    raw = list(raw_observation)
    if len(raw) != len(NADE_NATIVE_SLOT_ORDER):
        raise ValueError("NADE native observation must contain six slots")
    selected = _nade_native_slot_selection(state)

    def identifier(vehicle: Any) -> Optional[str]:
        if vehicle is None:
            return None
        if vehicle_ids is not None and id(vehicle) in vehicle_ids:
            return vehicle_ids[id(vehicle)]
        return str(getattr(vehicle, "id", id(vehicle)))

    expected = {
        slot: identifier(vehicle)
        for slot, vehicle in zip(NADE_NATIVE_SLOT_ORDER, raw)
    }
    reconstructed = {
        slot: (selected[slot][0].vehicle_id if slot in selected else None)
        for slot in NADE_NATIVE_SLOT_ORDER
    }
    matches = {
        slot: expected[slot] == reconstructed[slot]
        for slot in NADE_NATIVE_SLOT_ORDER
    }
    return {
        "expected": expected,
        "reconstructed": reconstructed,
        "slot_matches": matches,
        "matched_slots": sum(matches.values()),
        "total_slots": len(NADE_NATIVE_SLOT_ORDER),
        "exact_match": all(matches.values()),
    }
