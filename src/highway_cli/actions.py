from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ActionOption:
    """One high-level action exposed to a decision provider."""

    index: int
    value: str
    description: str
    kind: str
    acceleration_mps2: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "index": self.index,
            "value": self.value,
            "description": self.description,
            "kind": self.kind,
        }
        if self.acceleration_mps2 is not None:
            data["acceleration_mps2"] = self.acceleration_mps2
        return data


def nade_action_space() -> List[ActionOption]:
    """Return the exact 33-action layout used by the public NADE repository.

    NADE uses two lateral actions followed by longitudinal accelerations from
    -4.0 to 2.0 m/s^2 in 0.2 m/s^2 increments. Decimal arithmetic prevents
    representation drift while constructing labels.
    """

    actions = [
        ActionOption(0, "LANE_LEFT", "Change one lane to the left.", "lateral"),
        ActionOption(1, "LANE_RIGHT", "Change one lane to the right.", "lateral"),
    ]
    value = Decimal("-4.0")
    stop = Decimal("2.0")
    step = Decimal("0.2")
    index = 2
    while value <= stop:
        acceleration = float(value)
        label = str(acceleration)
        actions.append(
            ActionOption(
                index=index,
                value=label,
                description=(
                    "Apply a longitudinal acceleration of "
                    f"{acceleration:.1f} m/s^2 for the next policy interval."
                ),
                kind="longitudinal",
                acceleration_mps2=acceleration,
            )
        )
        value += step
        index += 1
    return actions


def action_map() -> Dict[str, ActionOption]:
    return {action.value: action for action in nade_action_space()}

