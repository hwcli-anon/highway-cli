from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .actions import ActionOption


SCHEMA_VERSION = "highway-state-v1"


def _rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    role: str
    lane_index: int
    relative_longitudinal_m: float
    relative_lateral_m: float
    speed_mps: float
    relative_speed_mps: float
    acceleration_mps2: Optional[float] = None

    def normalized(self) -> "VehicleState":
        return VehicleState(
            vehicle_id=str(self.vehicle_id),
            role=str(self.role),
            lane_index=int(self.lane_index),
            relative_longitudinal_m=_rounded(self.relative_longitudinal_m),
            relative_lateral_m=_rounded(self.relative_lateral_m),
            speed_mps=_rounded(self.speed_mps),
            relative_speed_mps=_rounded(self.relative_speed_mps),
            acceleration_mps2=_rounded(self.acceleration_mps2),
        )


@dataclass(frozen=True)
class RiskFeatures:
    ttc_s: Optional[float] = None
    time_headway_s: Optional[float] = None
    predicted_min_distance_m: Optional[float] = None
    lane_conflict: Optional[bool] = None
    nade_criticality: Optional[float] = None

    def normalized(self) -> "RiskFeatures":
        return RiskFeatures(
            ttc_s=_rounded(self.ttc_s),
            time_headway_s=_rounded(self.time_headway_s),
            predicted_min_distance_m=_rounded(self.predicted_min_distance_m),
            lane_conflict=self.lane_conflict,
            nade_criticality=_rounded(self.nade_criticality, 6),
        )


@dataclass(frozen=True)
class ScenarioState:
    episode_id: str
    decision_step: int
    simulation_time_s: float
    lane_count: int
    speed_limit_mps: float
    autonomous_vehicle: VehicleState
    controlled_background_vehicle: VehicleState
    nearby_vehicles: List[VehicleState] = field(default_factory=list)
    risk: RiskFeatures = field(default_factory=RiskFeatures)
    recent_history: Dict[str, Any] = field(default_factory=dict)
    available_actions: List[ActionOption] = field(default_factory=list)
    nade_native_observation: Optional[List[Optional[VehicleState]]] = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioState":
        """Reconstruct a state saved by :meth:`to_dict`."""

        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported state schema: {data.get('schema_version')}")

        def vehicle(value: Dict[str, Any]) -> VehicleState:
            return VehicleState(
                vehicle_id=value["vehicle_id"],
                role=value["role"],
                lane_index=value["lane_index"],
                relative_longitudinal_m=value["relative_longitudinal_m"],
                relative_lateral_m=value["relative_lateral_m"],
                speed_mps=value["speed_mps"],
                relative_speed_mps=value["relative_speed_mps"],
                acceleration_mps2=value.get("acceleration_mps2"),
            )

        return cls(
            episode_id=data["episode_id"],
            decision_step=data["time"]["decision_step"],
            simulation_time_s=data["time"]["simulation_time_s"],
            lane_count=data["road"]["lane_count"],
            speed_limit_mps=data["road"]["speed_limit_mps"],
            autonomous_vehicle=vehicle(data["autonomous_vehicle"]),
            controlled_background_vehicle=vehicle(
                data["controlled_background_vehicle"]
            ),
            nearby_vehicles=[vehicle(item) for item in data.get("nearby_vehicles", [])],
            risk=RiskFeatures(**data.get("risk", {})),
            recent_history=dict(data.get("recent_history", {})),
            available_actions=[
                ActionOption(
                    index=item["index"],
                    value=item["value"],
                    description=item["description"],
                    kind=item["kind"],
                    acceleration_mps2=item.get("acceleration_mps2"),
                )
                for item in data.get("available_actions", [])
            ],
            nade_native_observation=(
                [
                    vehicle(item) if item is not None else None
                    for item in data["nade_native_observation"]
                ]
                if data.get("nade_native_observation") is not None
                else None
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Produce deterministic provider-neutral state.

        Nearby vehicles are sorted spatially, and all numeric values are rounded
        before serialization so provider comparisons use byte-stable inputs.
        """

        av = self.autonomous_vehicle.normalized()
        pov = self.controlled_background_vehicle.normalized()
        nearby = sorted(
            (vehicle.normalized() for vehicle in self.nearby_vehicles),
            key=lambda vehicle: (
                vehicle.lane_index,
                vehicle.relative_longitudinal_m,
                vehicle.vehicle_id,
            ),
        )
        result = {
            "schema_version": self.schema_version,
            "episode_id": str(self.episode_id),
            "time": {
                "decision_step": int(self.decision_step),
                "simulation_time_s": _rounded(self.simulation_time_s),
            },
            "road": {
                "lane_count": int(self.lane_count),
                "speed_limit_mps": _rounded(self.speed_limit_mps),
                "coordinate_system": "AV-centered; longitudinal +forward; lateral +right",
                "units": "SI",
            },
            "autonomous_vehicle": asdict(av),
            "controlled_background_vehicle": asdict(pov),
            "nearby_vehicles": [asdict(vehicle) for vehicle in nearby],
            "risk": asdict(self.risk.normalized()),
            "recent_history": dict(sorted(self.recent_history.items())),
            "available_actions": [action.to_dict() for action in self.available_actions],
        }
        if self.nade_native_observation is not None:
            result["nade_native_observation"] = [
                asdict(vehicle.normalized()) if vehicle is not None else None
                for vehicle in self.nade_native_observation
            ]
        return result

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def to_representation(self, profile: str) -> Dict[str, Any]:
        from .representations import render_state

        return render_state(self, profile)

    def canonical_representation(self, profile: str) -> str:
        from .representations import canonical_representation

        return canonical_representation(self, profile)


@dataclass(frozen=True)
class DecisionSpecification:
    name: str
    instructions: str
    actions: List[ActionOption]


@dataclass
class DecisionResult:
    action: str
    probabilities: Dict[str, float]
    confidence: Optional[float]
    provider: str
    model: str
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    error: Optional[str] = None

    def validate(self, specification: DecisionSpecification) -> None:
        allowed = {action.value for action in specification.actions}
        if self.action not in allowed:
            raise ValueError(f"Provider selected unavailable action: {self.action}")
        unknown = set(self.probabilities) - allowed
        if unknown:
            raise ValueError(f"Provider returned probabilities for unknown actions: {unknown}")
        for action, probability in self.probabilities.items():
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"Invalid probability for {action}: {probability}")
        total = sum(self.probabilities.values())
        if self.probabilities and not 0.999 <= total <= 1.001:
            raise ValueError(f"Action probabilities sum to {total}, expected 1")

