from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .benchmark_split import collection_seed, example_key
from .hierarchical_actions import MANEUVERS, KEEP_LANE, maneuver_for_action


ExampleKey = Tuple[str, int, str]


def _collection_seed_optional(row: Mapping[str, Any]) -> Optional[int]:
    try:
        return collection_seed(row)
    except ValueError:
        return None


def _prediction_key(record: Mapping[str, Any]) -> ExampleKey:
    state = record.get("state") or record.get("model_input") or record
    time = state.get("time", {})
    controlled = state.get("controlled_background_vehicle", {})
    return (
        str(state["episode_id"]),
        int(time.get("decision_step", state.get("decision_step"))),
        str(controlled.get("vehicle_id", "unknown")),
    )


def _best_action(row: Mapping[str, Any]) -> str:
    aggregate = row.get("horizon_stability", {}).get("aggregate_best_action")
    if aggregate is not None:
        return str(aggregate)
    outcomes = row.get("horizon_action_outcomes") or row.get("action_outcomes") or []
    if not outcomes:
        raise ValueError("Benchmark row has no rollout outcomes")
    return str(min(outcomes, key=lambda item: int(item["criticality_rank"]))["action"])


def _soft_maneuver_target(row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    repetitions = row.get("horizon_repetition_outcomes") or []
    top_maneuvers: List[str] = []
    for outcomes in repetitions:
        if outcomes:
            top = min(outcomes, key=lambda item: int(item["criticality_rank"]))
            top_maneuvers.append(maneuver_for_action(str(top["action"])))
    if not top_maneuvers:
        outcomes = row.get("horizon_action_outcomes") or []
        if outcomes and all(item.get("repetition_ranks") for item in outcomes):
            repetitions_count = len(outcomes[0]["repetition_ranks"])
            for repetition_index in range(repetitions_count):
                top = min(
                    outcomes,
                    key=lambda item: int(item["repetition_ranks"][repetition_index]),
                )
                top_maneuvers.append(maneuver_for_action(str(top["action"])))
    if not top_maneuvers:
        return None
    counts = Counter(top_maneuvers)
    return {maneuver: counts[maneuver] / len(top_maneuvers) for maneuver in MANEUVERS}


def _maneuver_probabilities(record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    result = record.get("result", {})
    raw_response = record.get("raw_response", result.get("raw_response", {})) or {}
    if raw_response.get("leaf_probabilities_are_conditional_on_selected_intent"):
        return None
    raw = record.get("probabilities", result.get("probabilities", {})) or {}
    if not raw:
        return None
    converted = {str(action): float(value) for action, value in raw.items()}
    if any(value < 0.0 or value > 1.0 for value in converted.values()):
        return None
    total = sum(converted.values())
    if not 0.999 <= total <= 1.001:
        return None
    maneuver_probs = {maneuver: 0.0 for maneuver in MANEUVERS}
    for action, value in converted.items():
        maneuver_probs[maneuver_for_action(action)] += value
    return maneuver_probs


def _prediction_action(record: Mapping[str, Any]) -> Optional[str]:
    result = record.get("result", {})
    action = record.get("action", result.get("action"))
    return str(action) if action is not None else None


def score_maneuver_records(
    benchmarks: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Score the maneuver stage against repeated three-step rollout labels."""

    benchmark_map = {example_key(row): row for row in benchmarks}
    prediction_map: Dict[ExampleKey, Mapping[str, Any]] = {}
    for prediction in predictions:
        key = _prediction_key(prediction)
        if key in prediction_map:
            raise ValueError(f"Duplicate prediction for {key}")
        prediction_map[key] = prediction

    confusion = {truth: {predicted: 0 for predicted in MANEUVERS} for truth in MANEUVERS}
    target_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    recalls: Dict[str, Optional[float]] = {}
    details: List[Dict[str, Any]] = []
    valid = 0
    correct = 0
    briers: List[float] = []
    total_variations: List[float] = []
    soft_target_count = 0
    probability_count = 0

    for key in sorted(benchmark_map):
        row = benchmark_map[key]
        truth_action = _best_action(row)
        truth = maneuver_for_action(truth_action)
        outcomes = row.get("horizon_action_outcomes") or row.get("action_outcomes") or []
        collision_opportunity = any(
            float(
                item.get(
                    "controlled_collision_rate",
                    bool(item.get("av_crash") and item.get("controlled_bv_crash")),
                )
            )
            > 0.0
            for item in outcomes
        )
        target_counts[truth] += 1
        prediction = prediction_map.get(key)
        action = _prediction_action(prediction) if prediction is not None else None
        predicted = maneuver_for_action(action) if action is not None else None
        is_valid = predicted in MANEUVERS
        is_correct = bool(is_valid and predicted == truth)
        detail: Dict[str, Any] = {
            "episode_id": key[0],
            "decision_step": key[1],
            "controlled_vehicle_id": key[2],
            "collection_seed": _collection_seed_optional(row),
            "target_action": truth_action,
            "target_maneuver": truth,
            "collision_opportunity": collision_opportunity,
            "predicted_action": action,
            "predicted_maneuver": predicted,
            "status": "valid" if is_valid else "missing",
            "correct": is_correct,
        }
        if is_valid:
            valid += 1
            correct += int(is_correct)
            prediction_counts[predicted] += 1
            confusion[truth][predicted] += 1

        target_distribution = _soft_maneuver_target(row)
        probabilities = _maneuver_probabilities(prediction) if prediction is not None else None
        if target_distribution is not None:
            soft_target_count += 1
            detail["target_maneuver_distribution"] = target_distribution
        if probabilities is not None:
            probability_count += 1
            detail["predicted_maneuver_probabilities"] = probabilities
        if target_distribution is not None and probabilities is not None:
            brier = sum(
                (probabilities[maneuver] - target_distribution[maneuver]) ** 2
                for maneuver in MANEUVERS
            )
            total_variation = 0.5 * sum(
                abs(probabilities[maneuver] - target_distribution[maneuver])
                for maneuver in MANEUVERS
            )
            briers.append(brier)
            total_variations.append(total_variation)
            detail["maneuver_brier"] = brier
            detail["maneuver_total_variation"] = total_variation
        details.append(detail)

    for maneuver in MANEUVERS:
        support = target_counts[maneuver]
        recalls[maneuver] = confusion[maneuver][maneuver] / support if support else None
    supported_recalls = [value for value in recalls.values() if value is not None]
    lane_support = target_counts["CHANGE_LEFT"] + target_counts["CHANGE_RIGHT"]
    lane_correct = (
        confusion["CHANGE_LEFT"]["CHANGE_LEFT"]
        + confusion["CHANGE_RIGHT"]["CHANGE_RIGHT"]
    )
    total = len(benchmark_map)
    return {
        "evaluation_target": "repeated-three-step-maneuver-labels",
        "examples": total,
        "predictions": len(prediction_map),
        "coverage": valid / total if total else 0.0,
        "accuracy": correct / valid if valid else None,
        "balanced_accuracy": mean(supported_recalls) if supported_recalls else None,
        "per_maneuver_recall": recalls,
        "lane_change_recall": lane_correct / lane_support if lane_support else None,
        "keep_lane_recall": recalls[KEEP_LANE],
        "target_counts": {name: target_counts[name] for name in MANEUVERS},
        "prediction_counts": {name: prediction_counts[name] for name in MANEUVERS},
        "confusion_matrix": confusion,
        "soft_target_examples": soft_target_count,
        "probability_examples": probability_count,
        "mean_maneuver_brier": mean(briers) if briers else None,
        "mean_maneuver_total_variation": mean(total_variations) if total_variations else None,
        "extra_predictions": len(set(prediction_map) - set(benchmark_map)),
        "details": details,
    }


def _exact_mcnemar_p(old_only: int, new_only: int) -> float:
    discordant = old_only + new_only
    if not discordant:
        return 1.0
    lower = min(old_only, new_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def compare_maneuver_predictions(
    benchmarks: Iterable[Mapping[str, Any]],
    baseline_predictions: Iterable[Mapping[str, Any]],
    candidate_predictions: Iterable[Mapping[str, Any]],
    *,
    comparison_status: str = "development_ab_not_confirmation",
) -> Dict[str, Any]:
    benchmark_rows = list(benchmarks)
    baseline = score_maneuver_records(benchmark_rows, baseline_predictions)
    candidate = score_maneuver_records(benchmark_rows, candidate_predictions)
    baseline_details = {
        (row["episode_id"], row["decision_step"], row["controlled_vehicle_id"]): row
        for row in baseline["details"]
    }
    candidate_details = {
        (row["episode_id"], row["decision_step"], row["controlled_vehicle_id"]): row
        for row in candidate["details"]
    }
    paired = []
    both_correct = baseline_only = candidate_only = both_wrong = 0
    for key in sorted(baseline_details):
        old = baseline_details[key]
        new = candidate_details[key]
        old_correct = bool(old["correct"])
        new_correct = bool(new["correct"])
        both_correct += int(old_correct and new_correct)
        baseline_only += int(old_correct and not new_correct)
        candidate_only += int(new_correct and not old_correct)
        both_wrong += int(not old_correct and not new_correct)
        paired.append(
            {
                "episode_id": key[0],
                "decision_step": key[1],
                "controlled_vehicle_id": key[2],
                "collection_seed": old["collection_seed"],
                "target_maneuver": old["target_maneuver"],
                "collision_opportunity": old["collision_opportunity"],
                "baseline_maneuver": old["predicted_maneuver"],
                "candidate_maneuver": new["predicted_maneuver"],
                "baseline_correct": old_correct,
                "candidate_correct": new_correct,
                "outcome": (
                    "both_correct" if old_correct and new_correct else
                    "baseline_only" if old_correct else
                    "candidate_only" if new_correct else "both_wrong"
                ),
            }
        )

    def summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        old_correct = sum(bool(row["baseline_correct"]) for row in rows)
        new_correct = sum(bool(row["candidate_correct"]) for row in rows)
        old_only = sum(
            bool(row["baseline_correct"] and not row["candidate_correct"])
            for row in rows
        )
        new_only = sum(
            bool(row["candidate_correct"] and not row["baseline_correct"])
            for row in rows
        )
        return {
            "examples": len(rows),
            "baseline_correct": old_correct,
            "candidate_correct": new_correct,
            "net_correct_gain": new_correct - old_correct,
            "baseline_only_correct": old_only,
            "candidate_only_correct": new_only,
        }

    by_seed = {
        str(seed): summarize_group(
            [row for row in paired if row["collection_seed"] == seed]
        )
        for seed in sorted({row["collection_seed"] for row in paired})
    }
    by_target = {
        maneuver: summarize_group(
            [row for row in paired if row["target_maneuver"] == maneuver]
        )
        for maneuver in MANEUVERS
    }
    by_collision = {
        str(value).lower(): summarize_group(
            [row for row in paired if row["collision_opportunity"] is value]
        )
        for value in (False, True)
    }
    baseline_summary = {key: value for key, value in baseline.items() if key != "details"}
    candidate_summary = {key: value for key, value in candidate.items() if key != "details"}
    return {
        "comparison_status": comparison_status,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired": {
            "examples": len(paired),
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
            "net_correct_gain": candidate_only - baseline_only,
            "exact_mcnemar_p_two_sided": _exact_mcnemar_p(baseline_only, candidate_only),
        },
        "stratified_paired": {
            "by_collection_seed": by_seed,
            "by_target_maneuver": by_target,
            "by_collision_opportunity": by_collision,
        },
        "details": paired,
    }
