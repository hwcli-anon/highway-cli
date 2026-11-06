from __future__ import annotations

import json
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .benchmark_split import example_key
from .branch_rollout import compare_rollout_rankings


def _state_is_identical(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def _collision(item: Mapping[str, Any]) -> bool:
    return bool(item.get("av_crash") and item.get("controlled_bv_crash"))


def _quality_label(stability: Mapping[str, Any]) -> str:
    consensus = float(stability["consensus_top_action_rate"])
    rank_correlation = float(stability["mean_pairwise_spearman"])
    aggregate_matches = bool(stability["aggregate_best_matches_consensus"])
    if consensus >= 0.8 and rank_correlation >= 0.8 and aggregate_matches:
        return "stable"
    if consensus >= 0.8 and rank_correlation >= 0.8:
        return "mixed_top_action"
    return "unstable"


def assess_label_stability(
    reference_records: Iterable[Mapping[str, Any]],
    repeated_records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compare single-rollout labels with repeated-rollout aggregates.

    The quality labels are diagnostic categories, not statistical ground truth:
    ``stable`` requires at least 0.8 top-action consensus, at least 0.8 mean
    pairwise Spearman correlation, and agreement between the aggregate best
    action and the modal best action.
    """

    reference = list(reference_records)
    repeated = list(repeated_records)
    reference_by_key = {example_key(row): row for row in reference}
    repeated_by_key = {example_key(row): row for row in repeated}
    if len(reference_by_key) != len(reference):
        raise ValueError("Duplicate example key in reference records")
    if len(repeated_by_key) != len(repeated):
        raise ValueError("Duplicate example key in repeated records")

    details: List[Dict[str, Any]] = []
    for key in sorted(reference_by_key):
        reference_row = reference_by_key[key]
        repeated_row = repeated_by_key.get(key)
        detail: Dict[str, Any] = {
            "episode_id": key[0],
            "decision_step": key[1],
            "controlled_vehicle_id": key[2],
            "collection_seed": reference_row.get("collection_seed"),
        }
        if repeated_row is None:
            detail["quality_label"] = "not_repeated"
            details.append(detail)
            continue

        stability = repeated_row.get("horizon_stability")
        if not stability:
            raise ValueError(f"Repeated record {key!r} has no horizon_stability")
        ranking = compare_rollout_rankings(
            reference_row["horizon_action_outcomes"],
            repeated_row["horizon_action_outcomes"],
        )
        repetition_outcomes: Sequence[Sequence[Mapping[str, Any]]] = repeated_row.get(
            "horizon_repetition_outcomes", []
        )
        collision_repetitions = [
            any(_collision(item) for item in repetition)
            for repetition in repetition_outcomes
        ]
        aggregate_outcomes = repeated_row["horizon_action_outcomes"]
        collision_rates = [
            float(item.get("controlled_collision_rate", 0.0))
            for item in aggregate_outcomes
        ]
        best = min(aggregate_outcomes, key=lambda item: int(item["criticality_rank"]))
        reference_collision_opportunity = any(
            _collision(item) for item in reference_row["horizon_action_outcomes"]
        )
        collision_opportunity_repetition_rate = (
            mean(collision_repetitions) if collision_repetitions else None
        )
        top_action_consensus_stable = (
            float(stability["consensus_top_action_rate"]) >= 0.8
        )
        full_ranking_stable = float(stability["mean_pairwise_spearman"]) >= 0.8
        aggregate_top_action_consistent = bool(
            stability["aggregate_best_matches_consensus"]
        )
        collision_event_consistent = None
        if collision_opportunity_repetition_rate is not None:
            collision_event_consistent = (
                collision_opportunity_repetition_rate == 1.0
                if reference_collision_opportunity
                else collision_opportunity_repetition_rate == 0.0
            )
        detail.update(
            {
                "state_identical": _state_is_identical(
                    reference_row["state"], repeated_row["state"]
                ),
                "rollout_repetitions": int(stability["rollout_repetitions"]),
                "quality_label": _quality_label(stability),
                "consensus_top_action": stability["consensus_top_action"],
                "consensus_top_action_rate": float(
                    stability["consensus_top_action_rate"]
                ),
                "aggregate_best_action": stability["aggregate_best_action"],
                "aggregate_best_matches_consensus": bool(
                    stability["aggregate_best_matches_consensus"]
                ),
                "mean_pairwise_spearman": float(
                    stability["mean_pairwise_spearman"]
                ),
                "min_pairwise_spearman": float(
                    stability["min_pairwise_spearman"]
                ),
                "reference_vs_repeated": ranking,
                "top_action_consensus_stable": top_action_consensus_stable,
                "full_ranking_stable": full_ranking_stable,
                "aggregate_top_action_consistent": aggregate_top_action_consistent,
                "reference_collision_opportunity": reference_collision_opportunity,
                "collision_opportunity_repetition_rate": collision_opportunity_repetition_rate,
                "collision_event_consistent_with_reference": collision_event_consistent,
                "actions_with_nonzero_collision_rate": sum(
                    rate > 0.0 for rate in collision_rates
                ),
                "actions_with_certain_collision": sum(
                    rate == 1.0 for rate in collision_rates
                ),
                "maximum_action_collision_rate": max(collision_rates, default=0.0),
                "aggregate_best_action_collision_rate": float(
                    best.get("controlled_collision_rate", 0.0)
                ),
            }
        )
        details.append(detail)

    matched = [row for row in details if row["quality_label"] != "not_repeated"]
    quality_counts = Counter(row["quality_label"] for row in details)
    collision_reference = [
        row for row in matched if row["reference_collision_opportunity"]
    ]
    summary = {
        "reference_examples": len(reference),
        "repeated_examples": len(repeated),
        "matched_examples": len(matched),
        "unmatched_reference_examples": len(reference) - len(matched),
        "exact_state_match_rate": (
            mean(bool(row["state_identical"]) for row in matched) if matched else None
        ),
        "rollout_repetitions": sorted(
            {int(row["rollout_repetitions"]) for row in matched}
        ),
        "quality_thresholds": {
            "stable_min_consensus_top_action_rate": 0.8,
            "stable_min_mean_pairwise_spearman": 0.8,
            "stable_requires_aggregate_consensus_match": True,
            "interpretation": "diagnostic categories, not statistical ground truth",
        },
        "quality_counts": dict(sorted(quality_counts.items())),
        "top_action_consensus_stable_examples": sum(
            row["top_action_consensus_stable"] for row in matched
        ),
        "full_ranking_stable_examples": sum(
            row["full_ranking_stable"] for row in matched
        ),
        "aggregate_top_action_consistent_examples": sum(
            row["aggregate_top_action_consistent"] for row in matched
        ),
        "collision_event_consistent_examples": sum(
            row["collision_event_consistent_with_reference"] for row in matched
        ),
        "reference_vs_repeated_top1_match_rate": (
            mean(row["reference_vs_repeated"]["top1_match"] for row in matched)
            if matched
            else None
        ),
        "reference_vs_repeated_mean_spearman": (
            mean(
                row["reference_vs_repeated"]["spearman_rank_correlation"]
                for row in matched
            )
            if matched
            else None
        ),
        "mean_consensus_top_action_rate": (
            mean(row["consensus_top_action_rate"] for row in matched)
            if matched
            else None
        ),
        "mean_pairwise_spearman": (
            mean(row["mean_pairwise_spearman"] for row in matched)
            if matched
            else None
        ),
        "reference_collision_examples": len(collision_reference),
        "reference_collision_opportunity_preserved_all_repetitions": sum(
            row["collision_opportunity_repetition_rate"] == 1.0
            for row in collision_reference
        ),
    }
    return {"summary": summary, "details": details}
