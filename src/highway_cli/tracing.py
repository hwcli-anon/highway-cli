from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import DecisionResult, DecisionSpecification, ScenarioState
from .representations import FULL_STATE_V1


def append_trace(
    path: Path,
    state: ScenarioState,
    specification: DecisionSpecification,
    result: DecisionResult,
    representation_profile: str = FULL_STATE_V1,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "state_json": state.canonical_json(),
        "state": state.to_dict(),
        "representation_profile": representation_profile,
        "model_input": state.to_representation(representation_profile),
        "specification": {
            "name": specification.name,
            "instructions": specification.instructions,
            "actions": [action.to_dict() for action in specification.actions],
        },
        "result": asdict(result),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

