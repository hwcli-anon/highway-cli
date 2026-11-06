from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class ReplayedBackgroundAction:
    vehicle_id: str
    action_index: int
    action: str


class BackgroundActionReplay:
    """Immutable step-indexed BV actions extracted from a trajectory trace."""

    def __init__(
        self,
        actions: Mapping[Tuple[int, int, int], List[ReplayedBackgroundAction]],
        criticality: Mapping[Tuple[int, int, int], float],
    ) -> None:
        self._actions = {key: list(value) for key, value in actions.items()}
        self._criticality = dict(criticality)

    @classmethod
    def from_path(cls, path: Path) -> "BackgroundActionReplay":
        path = Path(path)
        actions: Dict[Tuple[int, int, int], List[ReplayedBackgroundAction]] = {}
        criticality: Dict[Tuple[int, int, int], float] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    int(row["seed"]),
                    int(row.get("episode", 0)),
                    int(row["decision_step"]),
                )
                if key in actions:
                    raise ValueError(
                        f"Duplicate background replay step {key} at line {line_number}"
                    )
                step_actions = []
                seen_vehicle_ids = set()
                for item in row.get("selected_actions", []):
                    vehicle_id = str(item["vehicle_id"])
                    if vehicle_id in seen_vehicle_ids:
                        raise ValueError(
                            f"Duplicate vehicle {vehicle_id} at replay step {key}"
                        )
                    seen_vehicle_ids.add(vehicle_id)
                    step_actions.append(
                        ReplayedBackgroundAction(
                            vehicle_id=vehicle_id,
                            action_index=int(item["action_index"]),
                            action=str(item.get("action", item["action_index"])),
                        )
                    )
                actions[key] = step_actions
                value = row.get("max_nade_criticality")
                criticality[key] = float(value) if value is not None else 0.0
        if not actions:
            raise ValueError(f"Background action replay is empty: {path}")
        return cls(actions, criticality)

    def actions_for(
        self, seed: int, episode: int, decision_step: int
    ) -> List[ReplayedBackgroundAction]:
        return list(self._actions.get((seed, episode, decision_step), []))

    def has_step(self, seed: int, episode: int, decision_step: int) -> bool:
        return (seed, episode, decision_step) in self._actions

    def criticality_for(self, seed: int, episode: int, decision_step: int) -> float:
        return self._criticality.get((seed, episode, decision_step), 0.0)

