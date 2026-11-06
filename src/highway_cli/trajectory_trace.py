from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


TRAJECTORY_VERSION = "highway-trajectory-v1"


def _lane(vehicle: Any) -> int:
    lane_index = getattr(vehicle, "lane_index", 0)
    if isinstance(lane_index, (tuple, list)):
        return int(lane_index[-1])
    return int(lane_index)


def _position(vehicle: Any) -> tuple[float, float]:
    position = getattr(vehicle, "position", (0.0, 0.0))
    return float(position[0]), float(position[1])


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _acceleration(vehicle: Any) -> Optional[float]:
    action = getattr(vehicle, "action", None)
    if isinstance(action, Mapping):
        return _optional_float(action.get("acceleration"))
    return None


def ensure_vehicle_ids(env: Any, vehicle_ids: MutableMapping[int, str]) -> None:
    """Assign stable run-local identifiers to vehicles created after reset."""

    used = set(vehicle_ids.values())
    next_index = 1
    for vehicle in getattr(getattr(env, "road", None), "vehicles", []):
        key = id(vehicle)
        if key in vehicle_ids:
            continue
        if vehicle is getattr(env, "vehicle", None):
            vehicle_ids[key] = "AV"
            used.add("AV")
            continue
        while f"BV-{next_index:03d}" in used:
            next_index += 1
        identifier = f"BV-{next_index:03d}"
        vehicle_ids[key] = identifier
        used.add(identifier)
        next_index += 1


def capture_environment(
    env: Any,
    vehicle_ids: MutableMapping[int, str],
    *,
    selected_vehicle_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    ensure_vehicle_ids(env, vehicle_ids)
    av = env.vehicle
    av_x, av_y = _position(av)
    selected = set(selected_vehicle_ids)
    controlled = {id(vehicle) for vehicle in getattr(env, "controlled_bvs", [])}
    vehicles: List[Dict[str, Any]] = []
    for vehicle in getattr(env.road, "vehicles", []):
        identifier = vehicle_ids[id(vehicle)]
        x, y = _position(vehicle)
        if vehicle is av:
            role = "autonomous_vehicle"
        elif identifier in selected:
            role = "selected_background_vehicle"
        elif id(vehicle) in controlled:
            role = "candidate_background_vehicle"
        else:
            role = "background_vehicle"
        vehicles.append(
            {
                "vehicle_id": identifier,
                "role": role,
                "x_m": x,
                "y_m": y,
                "relative_longitudinal_m": x - av_x,
                "relative_lateral_m": y - av_y,
                "lane_index": _lane(vehicle),
                "speed_mps": _optional_float(getattr(vehicle, "velocity", None)),
                "acceleration_mps2": _acceleration(vehicle),
                "heading_rad": _optional_float(getattr(vehicle, "heading", None)),
                "crashed": bool(getattr(vehicle, "crashed", False)),
            }
        )
    vehicles.sort(key=lambda item: item["vehicle_id"])
    return {
        "simulation_time_s": _optional_float(getattr(env, "time", None)),
        "vehicles": vehicles,
        "crashed_vehicle_ids": [
            item["vehicle_id"] for item in vehicles if item["crashed"]
        ],
    }


def write_trajectory_step(
    handle: Any,
    *,
    seed: int,
    episode: int,
    decision_step: int,
    selected_actions: List[Dict[str, Any]],
    cav_action: Any,
    pre_state: Dict[str, Any],
    post_state: Dict[str, Any],
    done: bool,
    scene_type: Optional[str],
    max_criticality: Any,
) -> None:
    record = {
        "trajectory_version": TRAJECTORY_VERSION,
        "seed": int(seed),
        "episode": int(episode),
        "episode_id": f"nade-{seed}-{episode}",
        "decision_step": int(decision_step),
        "max_nade_criticality": _optional_float(max_criticality),
        "selected_actions": selected_actions,
        "cav_action": cav_action,
        "pre": pre_state,
        "post": post_state,
        "done": bool(done),
        "scene_type": scene_type,
    }
    handle.write(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    handle.flush()


def open_trajectory(path: Path) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")
