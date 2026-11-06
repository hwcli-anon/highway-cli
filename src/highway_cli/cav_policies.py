from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple


AV2_POLICY = "av2"
IDM_MOBIL_POLICY = "idm-mobil"
CAV_POLICIES = (AV2_POLICY, IDM_MOBIL_POLICY)


def validate_cav_policy(policy: str) -> str:
    if policy not in CAV_POLICIES:
        raise ValueError(
            f"Unsupported CAV policy: {policy}. Expected one of {CAV_POLICIES}"
        )
    return policy


def nearest_longitudinal_action(
    acceleration: float,
    actions: Mapping[int, str],
    action_indicator: Optional[Sequence[Any]] = None,
) -> int:
    """Map a continuous acceleration to the nearest allowed NADE action."""

    candidates = []
    for index, value in actions.items():
        if value in {"LANE_LEFT", "LANE_RIGHT"}:
            continue
        if action_indicator is not None and not bool(action_indicator[index]):
            continue
        candidates.append((abs(float(value) - acceleration), int(index)))
    if not candidates:
        raise ValueError("No feasible longitudinal CAV action")
    return min(candidates)[1]


class IDMMobilCAVAgent:
    """Deterministic rule-based CAV using IDM longitudinal control and MOBIL lanes.

    The public NADE environment creates the ego vehicle as an ``MDPVehicle`` and
    expects an index in its original 33-action space.  This adapter reuses that
    interface while computing a rule-based action from the live road state.
    """

    COMFORT_ACC_MAX = 2.0
    COMFORT_ACC_MIN = -4.0
    DISTANCE_WANTED = 5.0
    TIME_WANTED = 1.5
    DESIRED_VELOCITY = 35.0
    DELTA = 4.0
    POLITENESS = 0.5
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0

    def __init__(self, env: Any, *, actions: Mapping[int, str]):
        self.env = env
        self.actions = dict(actions)

    def lane_conflict_safety_check(
        self, original_obs_df: Any, action_indicator_before: Sequence[Any]
    ) -> Any:
        # The environment indicator already enforces road and maneuver safety.
        # MOBIL performs an additional target-lane follower braking check.
        del original_obs_df
        try:
            return action_indicator_before.copy()
        except AttributeError:
            return list(action_indicator_before)

    def decision(
        self, original_obs_df: Any, action_indicator: Optional[Sequence[Any]] = None
    ) -> int:
        del original_obs_df
        vehicle = self.env.vehicle
        indicator = self._normalized_indicator(action_indicator)

        lane_action = self._mobil_action(vehicle, indicator)
        if lane_action is not None:
            return lane_action

        front_vehicle, _ = vehicle.road.neighbour_vehicles(vehicle)
        acceleration = max(
            self.COMFORT_ACC_MIN,
            min(self.COMFORT_ACC_MAX, self._acceleration(vehicle, front_vehicle)),
        )
        try:
            return nearest_longitudinal_action(
                acceleration, self.actions, action_indicator=indicator
            )
        except ValueError:
            # Match the legacy AV2 emergency behavior: if the safety mask blocks
            # every action, use the strongest braking action.
            return min(
                (
                    (float(value), int(index))
                    for index, value in self.actions.items()
                    if value not in {"LANE_LEFT", "LANE_RIGHT"}
                ),
                key=lambda item: item[0],
            )[1]

    def _normalized_indicator(
        self, action_indicator: Optional[Sequence[Any]]
    ) -> Sequence[bool]:
        size = max(self.actions) + 1
        if action_indicator is None:
            return [True] * size
        return [
            bool(action_indicator[index]) if index < len(action_indicator) else False
            for index in range(size)
        ]

    def _acceleration(self, vehicle: Any, front_vehicle: Any = None) -> float:
        if vehicle is None:
            return 0.0
        speed = float(vehicle.velocity)
        acceleration = self.COMFORT_ACC_MAX * (
            1.0 - math.pow(speed / self.DESIRED_VELOCITY, self.DELTA)
        )
        if front_vehicle is not None:
            distance = max(
                1e-5,
                float(vehicle.lane_distance_to(front_vehicle))
                - float(getattr(vehicle, "LENGTH", 5.0)),
            )
            gap = self._desired_gap(vehicle, front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * math.pow(gap / distance, 2)
        return acceleration

    def _desired_gap(self, vehicle: Any, front_vehicle: Any) -> float:
        braking_product = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        relative_speed = float(vehicle.velocity) - float(front_vehicle.velocity)
        dynamic = (
            float(vehicle.velocity) * self.TIME_WANTED
            + float(vehicle.velocity)
            * relative_speed
            / (2.0 * math.sqrt(braking_product))
        )
        return self.DISTANCE_WANTED + max(0.0, dynamic)

    def _mobil_action(
        self, vehicle: Any, indicator: Sequence[bool]
    ) -> Optional[int]:
        # Do not issue a second lane-change command while one is in progress.
        if vehicle.lane_index != vehicle.target_lane_index:
            return None

        current_lane = vehicle.lane_index
        current_id = int(current_lane[2])
        best: Optional[Tuple[float, int]] = None
        for lane_index in vehicle.road.network.side_lanes(current_lane):
            lane_id = int(lane_index[2])
            if lane_id < current_id:
                action_index = 0
            elif lane_id > current_id:
                action_index = 1
            else:
                continue
            if action_index >= len(indicator) or not indicator[action_index]:
                continue
            gain = self._mobil_gain(vehicle, lane_index)
            if gain is None or gain <= self.LANE_CHANGE_MIN_ACC_GAIN:
                continue
            if best is None or gain > best[0]:
                best = (gain, action_index)
        return best[1] if best is not None else None

    def _mobil_gain(self, vehicle: Any, lane_index: Any) -> Optional[float]:
        road = vehicle.road
        new_preceding, new_following = road.neighbour_vehicles(vehicle, lane_index)
        self_predicted = self._acceleration(vehicle, new_preceding)
        if self_predicted < self.COMFORT_ACC_MIN:
            return None

        new_following_current = self._acceleration(new_following, new_preceding)
        new_following_predicted = self._acceleration(new_following, vehicle)
        if new_following_predicted < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return None

        old_preceding, old_following = road.neighbour_vehicles(vehicle)
        self_current = self._acceleration(vehicle, old_preceding)
        old_following_current = self._acceleration(old_following, vehicle)
        old_following_predicted = self._acceleration(old_following, old_preceding)
        return self_predicted - self_current + self.POLITENESS * (
            new_following_predicted
            - new_following_current
            + old_following_predicted
            - old_following_current
        )


def build_cav_agent(
    policy: str,
    env: Any,
    *,
    actions: Mapping[int, str],
    av2_agent_type: Any,
) -> Any:
    policy = validate_cav_policy(policy)
    if policy == AV2_POLICY:
        return av2_agent_type(env)
    return IDMMobilCAVAgent(env, actions=actions)
