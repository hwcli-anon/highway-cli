from __future__ import annotations

from .actions import nade_action_space
from .schema import DecisionSpecification, RiskFeatures, ScenarioState, VehicleState


def demo_state() -> ScenarioState:
    actions = [action for action in nade_action_space() if action.value in {"LANE_RIGHT", "-4.0", "-2.0", "0.0", "1.0"}]
    return ScenarioState(
        episode_id="demo-001",
        decision_step=7,
        simulation_time_s=7.0,
        lane_count=3,
        speed_limit_mps=40.0,
        autonomous_vehicle=VehicleState(
            vehicle_id="AV",
            role="autonomous_vehicle",
            lane_index=0,
            relative_longitudinal_m=0.0,
            relative_lateral_m=0.0,
            speed_mps=30.0,
            relative_speed_mps=0.0,
            acceleration_mps2=0.0,
        ),
        controlled_background_vehicle=VehicleState(
            vehicle_id="BV-3",
            role="controlled_background_vehicle",
            lane_index=1,
            relative_longitudinal_m=12.4,
            relative_lateral_m=4.0,
            speed_mps=27.5,
            relative_speed_mps=-2.5,
            acceleration_mps2=-0.2,
        ),
        nearby_vehicles=[
            VehicleState("BV-7", "neighbor", 1, -18.0, 4.0, 31.0, 1.0, 0.0),
            VehicleState("BV-1", "neighbor", 0, 35.0, 0.0, 29.0, -1.0, 0.0),
        ],
        risk=RiskFeatures(
            ttc_s=2.4,
            time_headway_s=0.41,
            predicted_min_distance_m=2.1,
            lane_conflict=True,
            nade_criticality=0.018,
        ),
        recent_history={"pov_distance_trend": "closing", "window_s": 2.0},
        available_actions=actions,
    )


def adversarial_specification(state: ScenarioState) -> DecisionSpecification:
    return DecisionSpecification(
        name="background_action",
        instructions=(
            "Choose one physically feasible action for the controlled background vehicle. "
            "Prefer an action that can expose a safety weakness of the autonomous vehicle "
            "while remaining plausible as human driving behavior. Use only the listed actions."
        ),
        actions=state.available_actions,
    )

