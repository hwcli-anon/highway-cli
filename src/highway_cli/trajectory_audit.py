from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .trajectory_trace import TRAJECTORY_VERSION


def read_trajectory(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("trajectory_version") != TRAJECTORY_VERSION:
                raise ValueError(f"Unsupported trajectory row in {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Trajectory is empty: {path}")
    return rows


def _by_id(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["vehicle_id"]: item for item in snapshot["vehicles"]}


def _finite(values: Iterable[Optional[float]]) -> List[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def _ttc(av: Dict[str, Any], other: Dict[str, Any]) -> Optional[float]:
    if av["lane_index"] != other["lane_index"]:
        return None
    gap = float(other["x_m"]) - float(av["x_m"])
    relative_speed = float(other["speed_mps"]) - float(av["speed_mps"])
    if gap * relative_speed >= 0.0 or abs(relative_speed) < 1e-9:
        return None
    return abs(gap / relative_speed)


def _collision_partner(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    final = rows[-1]["post"]
    crashed = [identifier for identifier in final["crashed_vehicle_ids"] if identifier != "AV"]
    if crashed:
        post = _by_id(final)
        av = post.get("AV")
        if av is not None:
            return min(
                crashed,
                key=lambda identifier: math.hypot(
                    float(post[identifier]["x_m"]) - float(av["x_m"]),
                    float(post[identifier]["y_m"]) - float(av["y_m"]),
                ),
            )
        return crashed[0]
    return None


def _collision_type(
    rows: Sequence[Dict[str, Any]], partner_id: Optional[str]
) -> str:
    if rows[-1].get("scene_type") != "AV-Crash":
        return "no_collision"
    if partner_id is None:
        return "unknown_partner"
    final_pre = _by_id(rows[-1]["pre"])
    av = final_pre.get("AV")
    partner = final_pre.get(partner_id)
    if av is None or partner is None:
        return "unknown_geometry"

    recent = rows[-3:]
    av_lanes = []
    partner_lanes = []
    for row in recent:
        state = _by_id(row["pre"])
        if "AV" in state:
            av_lanes.append(state["AV"]["lane_index"])
        if partner_id in state:
            partner_lanes.append(state[partner_id]["lane_index"])
    lane_change = len(set(av_lanes)) > 1 or len(set(partner_lanes)) > 1
    lateral_gap = abs(float(partner["y_m"]) - float(av["y_m"]))
    if lane_change or av["lane_index"] != partner["lane_index"] or lateral_gap > 2.0:
        return "lane_change_or_side_collision"

    longitudinal_gap = float(partner["x_m"]) - float(av["x_m"])
    if longitudinal_gap > 0.5:
        return "av_rear_ended_background_vehicle"
    if longitudinal_gap < -0.5:
        return "background_vehicle_rear_ended_av"
    return "longitudinal_overlap_collision"


def analyze_episode(rows: Sequence[Dict[str, Any]], *, source: Optional[Path] = None) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["decision_step"]))
    first = ordered[0]
    final = ordered[-1]
    crashed = final.get("scene_type") == "AV-Crash"
    partner_id = _collision_partner(ordered) if crashed else None
    selected_vehicle_ids = {
        action["vehicle_id"]
        for row in ordered
        for action in row.get("selected_actions", [])
    }

    all_distances = []
    same_lane_longitudinal_distances = []
    partner_ttcs = []
    partner_relative_speeds = []
    for row in ordered:
        state = _by_id(row["pre"])
        av = state.get("AV")
        if av is None:
            continue
        for identifier, other in state.items():
            if identifier == "AV":
                continue
            all_distances.append(
                math.hypot(
                    float(other["x_m"]) - float(av["x_m"]),
                    float(other["y_m"]) - float(av["y_m"]),
                )
            )
            if other["lane_index"] == av["lane_index"]:
                same_lane_longitudinal_distances.append(
                    abs(float(other["x_m"]) - float(av["x_m"]))
                )
        if partner_id is not None and partner_id in state:
            partner = state[partner_id]
            partner_ttcs.append(_ttc(av, partner))
            partner_relative_speeds.append(
                abs(float(partner["speed_mps"]) - float(av["speed_mps"]))
            )

    action_sequences: Dict[str, List[str]] = {}
    acceleration_values = []
    for row in ordered:
        for action in row.get("selected_actions", []):
            action_sequences.setdefault(action["vehicle_id"], []).append(str(action["action"]))
            try:
                acceleration_values.append(float(action["action"]))
            except (TypeError, ValueError):
                pass
    switches = sum(
        previous != current
        for sequence in action_sequences.values()
        for previous, current in zip(sequence, sequence[1:])
    )
    action_count = sum(len(sequence) for sequence in action_sequences.values())
    extreme_count = sum(value <= -3.0 or value >= 1.5 for value in acceleration_values)

    pre_final = _by_id(final["pre"])
    final_relative_speed = None
    final_relative_longitudinal = None
    final_partner_position = None
    if partner_id is not None and "AV" in pre_final and partner_id in pre_final:
        final_relative_speed = abs(
            float(pre_final[partner_id]["speed_mps"])
            - float(pre_final["AV"]["speed_mps"])
        )
        final_relative_longitudinal = float(
            pre_final[partner_id]["relative_longitudinal_m"]
        )
        if final_relative_longitudinal > 1.0:
            final_partner_position = "ahead_of_av"
        elif final_relative_longitudinal < -1.0:
            final_partner_position = "behind_av"
        else:
            final_partner_position = "overlapping_av"

    return {
        "source": str(source) if source is not None else None,
        "seed": int(first["seed"]),
        "episode": int(first["episode"]),
        "episode_id": first["episode_id"],
        "steps": len(ordered),
        "crash": bool(crashed),
        "collision_type": _collision_type(ordered, partner_id),
        "collision_partner_id": partner_id,
        "collision_partner_was_selected": (
            partner_id in selected_vehicle_ids if partner_id is not None else None
        ),
        "collision_decision_step": int(final["decision_step"]) if crashed else None,
        "collision_time_s": final["post"].get("simulation_time_s") if crashed else None,
        "min_center_distance_m": min(all_distances) if all_distances else None,
        "min_same_lane_longitudinal_distance_m": (
            min(same_lane_longitudinal_distances)
            if same_lane_longitudinal_distances
            else None
        ),
        "min_partner_ttc_s": min(_finite(partner_ttcs)) if _finite(partner_ttcs) else None,
        "max_partner_relative_speed_mps": (
            max(partner_relative_speeds) if partner_relative_speeds else None
        ),
        "pre_collision_relative_speed_mps": final_relative_speed,
        "pre_collision_relative_longitudinal_m": final_relative_longitudinal,
        "pre_collision_partner_position": final_partner_position,
        "selected_action_count": action_count,
        "selected_action_switches": switches,
        "selected_action_switch_rate": switches / max(1, action_count - len(action_sequences)),
        "extreme_longitudinal_action_count": extreme_count,
        "extreme_longitudinal_action_rate": (
            extreme_count / len(acceleration_values) if acceleration_values else None
        ),
        "selected_vehicle_count": len(selected_vehicle_ids),
        "max_nade_criticality": max(
            _finite(row.get("max_nade_criticality") for row in ordered), default=None
        ),
    }


def summarize_trajectories(paths: Sequence[Path]) -> Dict[str, Any]:
    episodes = [analyze_episode(read_trajectory(path), source=path) for path in paths]
    collisions = [episode for episode in episodes if episode["crash"]]
    return {
        "trajectory_audit_version": "highway-trajectory-audit-v1",
        "episodes": len(episodes),
        "collisions": len(collisions),
        "collision_rate": len(collisions) / len(episodes) if episodes else None,
        "collision_type_counts": dict(
            sorted(Counter(item["collision_type"] for item in episodes).items())
        ),
        "selected_actor_collision_count": sum(
            item["collision_partner_was_selected"] is True for item in collisions
        ),
        "pre_collision_partner_position_counts": dict(
            sorted(
                Counter(
                    item["pre_collision_partner_position"]
                    for item in collisions
                    if item["pre_collision_partner_position"] is not None
                ).items()
            )
        ),
        "mean_pre_collision_relative_speed_mps": (
            mean(
                _finite(
                    item["pre_collision_relative_speed_mps"] for item in collisions
                )
            )
            if _finite(item["pre_collision_relative_speed_mps"] for item in collisions)
            else None
        ),
        "mean_selected_action_switch_rate": (
            mean(item["selected_action_switch_rate"] for item in episodes)
            if episodes
            else None
        ),
        "mean_extreme_longitudinal_action_rate": (
            mean(
                _finite(item["extreme_longitudinal_action_rate"] for item in episodes)
            )
            if _finite(item["extreme_longitudinal_action_rate"] for item in episodes)
            else None
        ),
        "episode_details": episodes,
    }


def write_episode_csv(path: Path, episodes: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(episodes[0]) if episodes else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(episodes)
