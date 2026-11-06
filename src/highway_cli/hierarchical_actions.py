from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .actions import ActionOption


KEEP_LANE = "KEEP_LANE"
CHANGE_LEFT = "CHANGE_LEFT"
CHANGE_RIGHT = "CHANGE_RIGHT"
MANEUVERS = (KEEP_LANE, CHANGE_LEFT, CHANGE_RIGHT)


def maneuver_for_action(action: str) -> str:
    """Map one original NADE action to its lossless first-stage maneuver."""

    if action == "LANE_LEFT":
        return CHANGE_LEFT
    if action == "LANE_RIGHT":
        return CHANGE_RIGHT
    return KEEP_LANE


def hierarchical_action_structure(
    actions: Sequence[ActionOption],
) -> Dict[str, Any]:
    """Describe a balanced maneuver-first view of the original NADE actions.

    The hierarchy changes only how a model sees and selects actions. Every leaf
    is still an original NADE action, so execution and rollout scoring remain
    exactly comparable with the flat-action baseline.
    """

    grouped: Dict[str, List[ActionOption]] = {name: [] for name in MANEUVERS}
    for action in actions:
        grouped[maneuver_for_action(action.value)].append(action)

    descriptions = {
        KEEP_LANE: "Keep the current lane, then choose a longitudinal acceleration.",
        CHANGE_LEFT: "Change one lane to the left using the original NADE action.",
        CHANGE_RIGHT: "Change one lane to the right using the original NADE action.",
    }
    first_stage = [
        {
            "maneuver": maneuver,
            "description": descriptions[maneuver],
            "available": bool(grouped[maneuver]),
            "leaf_action_count": len(grouped[maneuver]),
        }
        for maneuver in MANEUVERS
        if grouped[maneuver]
    ]
    second_stage = {
        maneuver: [action.to_dict() for action in grouped[maneuver]]
        for maneuver in MANEUVERS
        if grouped[maneuver]
    }
    return {
        "selection_order": ["maneuver", "original_nade_action"],
        "first_stage_maneuvers": first_stage,
        "second_stage_actions": second_stage,
        "leaf_actions_are_original_nade_actions": True,
    }


def _ranked_outcomes(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    outcomes = row.get("horizon_action_outcomes") or row.get("action_outcomes") or []
    return sorted(outcomes, key=lambda item: int(item["criticality_rank"]))


def _top_action(outcomes: Sequence[Mapping[str, Any]]) -> str:
    if not outcomes:
        raise ValueError("A benchmark row has no rollout outcomes")
    return str(min(outcomes, key=lambda item: int(item["criticality_rank"]))["action"])


def _consensus_rate(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    return max(Counter(values).values()) / len(values)


def analyze_hierarchical_records(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audit whether maneuver-first selection is lossless and worth testing.

    This is a development-set diagnostic. It does not estimate model quality and
    intentionally performs no API calls.
    """

    details = []
    best_maneuvers: Counter = Counter()
    flat_counts: List[int] = []
    first_stage_counts: List[int] = []
    longitudinal_counts: List[int] = []
    lateral_counts: List[int] = []
    exact_consensus_rates: List[float] = []
    maneuver_consensus_rates: List[float] = []
    mapping_complete = 0

    for row in records:
        state = row["state"]
        state_actions = [str(item["value"]) for item in state["available_actions"]]
        grouped_values = {
            maneuver: [
                action for action in state_actions if maneuver_for_action(action) == maneuver
            ]
            for maneuver in MANEUVERS
        }
        nonempty_groups = {key: value for key, value in grouped_values.items() if value}
        flattened = [action for values in nonempty_groups.values() for action in values]
        complete = len(flattened) == len(set(flattened)) and set(flattened) == set(
            state_actions
        )
        mapping_complete += int(complete)

        outcomes = _ranked_outcomes(row)
        best_action = str(
            row.get("horizon_stability", {}).get("aggregate_best_action")
            or _top_action(outcomes)
        )
        best_maneuver = maneuver_for_action(best_action)
        best_maneuvers[best_maneuver] += 1
        best_rank_by_maneuver = {}
        for maneuver, values in nonempty_groups.items():
            ranks = [
                int(item["criticality_rank"])
                for item in outcomes
                if str(item["action"]) in values
            ]
            if ranks:
                best_rank_by_maneuver[maneuver] = min(ranks)

        repetition_outcomes = row.get("horizon_repetition_outcomes") or []
        repetition_actions = [_top_action(rep) for rep in repetition_outcomes if rep]
        repetition_maneuvers = [maneuver_for_action(value) for value in repetition_actions]
        if repetition_actions:
            exact_consensus_rates.append(_consensus_rate(repetition_actions))
            maneuver_consensus_rates.append(_consensus_rate(repetition_maneuvers))

        flat_counts.append(len(state_actions))
        first_stage_counts.append(len(nonempty_groups))
        longitudinal_counts.append(len(grouped_values[KEEP_LANE]))
        lateral_counts.append(
            len(grouped_values[CHANGE_LEFT]) + len(grouped_values[CHANGE_RIGHT])
        )
        details.append(
            {
                "episode_id": str(state["episode_id"]),
                "decision_step": int(state["time"]["decision_step"]),
                "controlled_vehicle_id": str(
                    state["controlled_background_vehicle"]["vehicle_id"]
                ),
                "flat_action_count": len(state_actions),
                "first_stage_maneuver_count": len(nonempty_groups),
                "leaf_action_counts": {
                    key: len(value) for key, value in nonempty_groups.items()
                },
                "best_action": best_action,
                "best_maneuver": best_maneuver,
                "best_rank_by_maneuver": best_rank_by_maneuver,
                "lossless_mapping": complete,
                "repetition_top_actions": repetition_actions,
                "repetition_top_maneuvers": repetition_maneuvers,
                "exact_action_consensus_rate": (
                    _consensus_rate(repetition_actions) if repetition_actions else None
                ),
                "maneuver_consensus_rate": (
                    _consensus_rate(repetition_maneuvers)
                    if repetition_maneuvers
                    else None
                ),
            }
        )

    count = len(details)
    return {
        "analysis_version": "hierarchical-action-offline-v1",
        "status": "development_diagnostic_not_confirmation",
        "examples": count,
        "hierarchy": {
            "first_stage": list(MANEUVERS),
            "second_stage_keep_lane": "exact original NADE longitudinal actions",
            "lateral_leaves": {
                CHANGE_LEFT: "LANE_LEFT",
                CHANGE_RIGHT: "LANE_RIGHT",
            },
        },
        "lossless_mapping_rate": mapping_complete / count if count else None,
        "best_maneuver_counts": {
            maneuver: best_maneuvers.get(maneuver, 0) for maneuver in MANEUVERS
        },
        "best_maneuver_rates": {
            maneuver: best_maneuvers.get(maneuver, 0) / count if count else None
            for maneuver in MANEUVERS
        },
        "candidate_space": {
            "flat_actions_mean": mean(flat_counts) if flat_counts else None,
            "first_stage_maneuvers_mean": (
                mean(first_stage_counts) if first_stage_counts else None
            ),
            "first_stage_maneuver_count_distribution": {
                str(key): value
                for key, value in sorted(Counter(first_stage_counts).items())
            },
            "states_with_lateral_choice": sum(count > 1 for count in first_stage_counts),
            "longitudinal_leaf_actions_mean": (
                mean(longitudinal_counts) if longitudinal_counts else None
            ),
            "lateral_leaf_actions_mean": (
                mean(lateral_counts) if lateral_counts else None
            ),
            "flat_to_first_stage_reduction_factor": (
                mean(flat_counts) / mean(first_stage_counts)
                if first_stage_counts and mean(first_stage_counts)
                else None
            ),
        },
        "repeated_rollout_stability": {
            "examples": len(exact_consensus_rates),
            "mean_exact_action_consensus_rate": (
                mean(exact_consensus_rates) if exact_consensus_rates else None
            ),
            "mean_maneuver_consensus_rate": (
                mean(maneuver_consensus_rates) if maneuver_consensus_rates else None
            ),
            "maneuver_consensus_non_decreasing_for_all_examples": all(
                maneuver >= exact
                for exact, maneuver in zip(
                    exact_consensus_rates, maneuver_consensus_rates
                )
            ),
        },
        "details": details,
    }
