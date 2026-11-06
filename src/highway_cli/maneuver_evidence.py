from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Set, Tuple

from .hierarchical_actions import CHANGE_LEFT, CHANGE_RIGHT, KEEP_LANE
from .representations import (
    SYMMETRIC_MANEUVER_EVIDENCE_V1,
    SYMMETRIC_MANEUVER_EVIDENCE_V2,
)
from .schema import ScenarioState


def _dict_key_paths(value: Any, path: Tuple[str, ...] = ()) -> Set[Tuple[str, ...]]:
    paths: Set[Tuple[str, ...]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            paths.add(child_path)
            paths.update(_dict_key_paths(child, child_path))
    return paths


def audit_maneuver_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    profile: str = SYMMETRIC_MANEUVER_EVIDENCE_V1,
) -> Dict[str, Any]:
    if profile not in {
        SYMMETRIC_MANEUVER_EVIDENCE_V1,
        SYMMETRIC_MANEUVER_EVIDENCE_V2,
    }:
        raise ValueError(f"Unsupported maneuver evidence profile: {profile}")
    rows = list(records)
    availability = Counter()
    byte_sizes = []
    symmetric_examples = 0
    availability_match_examples = 0
    deterministic_examples = 0
    details = []
    expected_order = [KEEP_LANE, CHANGE_LEFT, CHANGE_RIGHT]

    for row in rows:
        state = ScenarioState.from_dict(row["state"])
        payload = state.to_representation(profile)
        repeated = state.to_representation(profile)
        cards = payload["maneuver_evidence_cards"]
        names = [card["maneuver"] for card in cards]
        shapes = [_dict_key_paths(card) for card in cards]
        symmetric = names == expected_order and len(set(map(frozenset, shapes))) == 1
        symmetric_examples += int(symmetric)

        action_values = {action.value for action in state.available_actions}
        expected_available = {
            KEEP_LANE: any(value not in ("LANE_LEFT", "LANE_RIGHT") for value in action_values),
            CHANGE_LEFT: "LANE_LEFT" in action_values,
            CHANGE_RIGHT: "LANE_RIGHT" in action_values,
        }
        actual_available = {card["maneuver"]: bool(card["available"]) for card in cards}
        availability_matches = actual_available == expected_available
        availability_match_examples += int(availability_matches)
        deterministic = payload == repeated
        deterministic_examples += int(deterministic)
        for maneuver, present in actual_available.items():
            availability[f"{maneuver}:{'available' if present else 'unavailable'}"] += 1
        encoded = state.canonical_representation(profile)
        byte_sizes.append(len(encoded.encode("utf-8")))
        details.append(
            {
                "episode_id": state.episode_id,
                "decision_step": state.decision_step,
                "controlled_vehicle_id": state.controlled_background_vehicle.vehicle_id,
                "symmetric_card_schema": symmetric,
                "availability_matches_action_space": availability_matches,
                "deterministic": deterministic,
                "available_maneuvers": [
                    name for name in expected_order if actual_available[name]
                ],
            }
        )

    count = len(rows)
    return {
        "audit_version": "symmetric-maneuver-evidence-audit-v2",
        "representation_profile": profile,
        "status": "offline_structure_audit_not_model_evaluation",
        "examples": count,
        "all_examples_have_symmetric_card_schema": symmetric_examples == count,
        "all_examples_match_original_action_availability": availability_match_examples
        == count,
        "all_examples_are_deterministic": deterministic_examples == count,
        "availability_counts": dict(sorted(availability.items())),
        "utf8_bytes": {
            "min": min(byte_sizes) if byte_sizes else None,
            "mean": mean(byte_sizes) if byte_sizes else None,
            "max": max(byte_sizes) if byte_sizes else None,
        },
        "details": details,
    }
