from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .benchmark_split import example_key


ExampleKey = Tuple[str, int, str]


def _prediction_key(record: Mapping[str, Any]) -> ExampleKey:
    state = record.get("state") or record.get("model_input") or record
    time = state.get("time", {})
    controlled = state.get("controlled_background_vehicle", {})
    return (
        str(state["episode_id"]),
        int(time.get("decision_step", state.get("decision_step"))),
        str(controlled.get("vehicle_id", "unknown")),
    )


def _prediction_payload(
    record: Mapping[str, Any],
) -> Tuple[Optional[str], Mapping[str, Any]]:
    result = record.get("result", {})
    action = record.get("action", result.get("action"))
    raw_response = record.get("raw_response", result.get("raw_response", {})) or {}
    if raw_response.get("leaf_probabilities_are_conditional_on_selected_intent"):
        # Conditional probabilities are valid for execution after the selected
        # intent, but they are not a marginal distribution over all leaves and
        # therefore must not be used for Brier/TV calibration comparisons.
        return (str(action) if action is not None else None), {}
    probabilities = record.get("probabilities", result.get("probabilities", {}))
    return (str(action) if action is not None else None), probabilities or {}


def _validate_probabilities(
    probabilities: Mapping[str, Any], actions: Sequence[str]
) -> Tuple[Optional[Dict[str, float]], str]:
    if not probabilities:
        return None, "missing"
    allowed = set(actions)
    if set(probabilities) - allowed:
        return None, "unknown_action"
    converted = {str(action): float(value) for action, value in probabilities.items()}
    if any(value < 0.0 or value > 1.0 for value in converted.values()):
        return None, "out_of_range"
    total = sum(converted.values())
    if not 0.999 <= total <= 1.001:
        return None, "not_normalized"
    return {action: converted.get(action, 0.0) for action in actions}, "valid"


def _top_action_distribution(
    outcomes: Sequence[Mapping[str, Any]], repetitions: int
) -> Dict[str, float]:
    if repetitions < 1:
        raise ValueError("rollout_repetitions must be positive")
    distribution = {}
    for item in outcomes:
        ranks = item.get("repetition_ranks")
        if ranks is None or len(ranks) != repetitions:
            raise ValueError("Soft labels require one repetition rank per rollout")
        distribution[str(item["action"])] = (
            sum(int(rank) == 1 for rank in ranks) / repetitions
        )
    total = sum(distribution.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"Top-action frequencies sum to {total}, expected 1")
    return distribution


def _collision_rate(item: Mapping[str, Any]) -> float:
    return float(
        item.get(
            "controlled_collision_rate",
            bool(item.get("av_crash") and item.get("controlled_bv_crash")),
        )
    )


def _mean_rank(item: Mapping[str, Any]) -> float:
    if item.get("mean_repetition_rank") is not None:
        return float(item["mean_repetition_rank"])
    ranks = item.get("repetition_ranks")
    if not ranks:
        raise ValueError("Soft labels require repeated rollout ranks")
    return mean(float(rank) for rank in ranks)


def _mean_reciprocal_rank(item: Mapping[str, Any]) -> float:
    ranks = item.get("repetition_ranks")
    if not ranks:
        raise ValueError("Soft labels require repeated rollout ranks")
    return mean(1.0 / float(rank) for rank in ranks)


def _mean_distance(item: Mapping[str, Any]) -> float:
    return float(
        item.get("mean_min_center_distance_m", item["min_center_distance_m"])
    )


def score_soft_records(
    benchmarks: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    selection: Optional[Iterable[Mapping[str, Any]]] = None,
    distance_tolerance_m: float = 0.25,
    collision_rate_tolerance: float = 0.0,
    severe_normalized_rank_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Score decisions against repeated three-step rollout soft labels."""

    if distance_tolerance_m < 0.0:
        raise ValueError("distance_tolerance_m cannot be negative")
    if not 0.0 <= collision_rate_tolerance <= 1.0:
        raise ValueError("collision_rate_tolerance must be between 0 and 1")
    if not 0.0 <= severe_normalized_rank_threshold <= 1.0:
        raise ValueError(
            "severe_normalized_rank_threshold must be between 0 and 1"
        )

    prediction_map: Dict[ExampleKey, Tuple[Optional[str], Mapping[str, Any]]] = {}
    for record in predictions:
        key = _prediction_key(record)
        if key in prediction_map:
            raise ValueError(f"Duplicate prediction for {key}")
        prediction_map[key] = _prediction_payload(record)

    selection_keys = None
    if selection is not None:
        selection_rows = list(selection)
        selection_keys = {example_key(row) for row in selection_rows}
        if len(selection_keys) != len(selection_rows):
            raise ValueError("Duplicate example key in selection")

    benchmark_map: Dict[ExampleKey, Mapping[str, Any]] = {}
    for row in benchmarks:
        key = example_key(row)
        if selection_keys is not None and key not in selection_keys:
            continue
        if key in benchmark_map:
            raise ValueError(f"Duplicate benchmark example {key}")
        benchmark_map[key] = row
    if selection_keys is not None:
        missing = selection_keys - set(benchmark_map)
        if missing:
            raise ValueError(f"Selection contains {len(missing)} unmatched examples")

    details: List[Dict[str, Any]] = []
    chosen_top_frequencies: List[float] = []
    chosen_mean_ranks: List[float] = []
    chosen_normalized_ranks: List[float] = []
    chosen_reciprocal_ranks: List[float] = []
    collision_regrets: List[float] = []
    chosen_collision_rates_on_opportunities: List[float] = []
    distance_regrets_without_collision_opportunity: List[float] = []
    expected_ranks: List[float] = []
    expected_normalized_ranks: List[float] = []
    expected_collision_rates: List[float] = []
    brier_scores: List[float] = []
    total_variations: List[float] = []
    coverage = 0
    valid = 0
    aggregate_top1_matches = 0
    modal_top1_matches = 0
    equivalent_matches = 0
    collision_opportunities = 0
    evaluated_collision_opportunities = 0
    collision_opportunity_misses = 0
    severe_valid_decision_errors = 0
    bottom_tail_selections = 0
    distribution_statuses: Counter[str] = Counter()

    for key in sorted(benchmark_map):
        benchmark = benchmark_map[key]
        outcomes_list = benchmark.get("horizon_action_outcomes")
        stability = benchmark.get("horizon_stability")
        if not outcomes_list or not stability:
            raise ValueError(f"Benchmark {key!r} has no repeated horizon soft labels")
        repetitions = int(stability["rollout_repetitions"])
        outcomes = {str(item["action"]): item for item in outcomes_list}
        actions = list(outcomes)
        top_distribution = _top_action_distribution(outcomes_list, repetitions)
        aggregate_best = min(
            outcomes.values(), key=lambda item: int(item["criticality_rank"])
        )
        modal_best_action = str(stability["consensus_top_action"])
        max_collision_rate = max(_collision_rate(item) for item in outcomes.values())
        collision_opportunity = max_collision_rate > 0.0
        best_distance = min(_mean_distance(item) for item in outcomes.values())
        if collision_opportunity:
            collision_opportunities += 1
            equivalent_actions = {
                action
                for action, item in outcomes.items()
                if _collision_rate(item)
                >= max_collision_rate - collision_rate_tolerance
            }
        else:
            best_distance = min(_mean_distance(item) for item in outcomes.values())
            equivalent_actions = {
                action
                for action, item in outcomes.items()
                if _mean_distance(item) <= best_distance + distance_tolerance_m
            }

        attempted = key in prediction_map
        action, raw_probabilities = prediction_map.get(key, (None, {}))
        detail: Dict[str, Any] = {
            "episode_id": key[0],
            "decision_step": key[1],
            "controlled_vehicle_id": key[2],
            "predicted_action": action,
            "aggregate_best_action": str(aggregate_best["action"]),
            "modal_best_action": modal_best_action,
            "modal_best_action_frequency": top_distribution[modal_best_action],
            "collision_opportunity": collision_opportunity,
            "maximum_action_collision_rate": max_collision_rate,
            "equivalent_actions": sorted(equivalent_actions),
            "status": "missing",
            "probability_status": "missing",
        }
        if attempted and action is None:
            detail["status"] = "prediction_error"
        elif action is not None:
            coverage += 1
            if action not in outcomes:
                detail["status"] = "invalid_action"
            else:
                valid += 1
                selected = outcomes[action]
                selected_mean_rank = _mean_rank(selected)
                normalized_rank = (
                    (selected_mean_rank - 1.0) / (len(outcomes) - 1)
                    if len(outcomes) > 1
                    else 0.0
                )
                selected_collision_rate = _collision_rate(selected)
                collision_regret = max_collision_rate - selected_collision_rate
                selected_distance = _mean_distance(selected)
                distance_regret = max(0.0, selected_distance - best_distance)
                misses_collision_opportunity = (
                    collision_opportunity and selected_collision_rate <= 0.0
                )
                is_bottom_tail = (
                    normalized_rank >= severe_normalized_rank_threshold
                )
                selected_top_frequency = top_distribution[action]
                selected_reciprocal_rank = _mean_reciprocal_rank(selected)
                is_aggregate_best = action == str(aggregate_best["action"])
                is_modal_best = action == modal_best_action
                is_equivalent = action in equivalent_actions
                is_severe_decision_error = (
                    misses_collision_opportunity
                    or (is_bottom_tail and not is_equivalent)
                )
                aggregate_top1_matches += int(is_aggregate_best)
                modal_top1_matches += int(is_modal_best)
                equivalent_matches += int(is_equivalent)
                chosen_top_frequencies.append(selected_top_frequency)
                chosen_mean_ranks.append(selected_mean_rank)
                chosen_normalized_ranks.append(normalized_rank)
                chosen_reciprocal_ranks.append(selected_reciprocal_rank)
                if collision_opportunity:
                    evaluated_collision_opportunities += 1
                    collision_regrets.append(collision_regret)
                    chosen_collision_rates_on_opportunities.append(
                        selected_collision_rate
                    )
                    collision_opportunity_misses += int(
                        misses_collision_opportunity
                    )
                else:
                    distance_regrets_without_collision_opportunity.append(
                        distance_regret
                    )
                bottom_tail_selections += int(is_bottom_tail)
                severe_valid_decision_errors += int(is_severe_decision_error)
                detail.update(
                    {
                        "status": "valid",
                        "is_aggregate_best_action": is_aggregate_best,
                        "is_modal_best_action": is_modal_best,
                        "is_soft_equivalent_action": is_equivalent,
                        "selected_top_action_frequency": selected_top_frequency,
                        "selected_mean_repetition_rank": selected_mean_rank,
                        "selected_normalized_repetition_rank": normalized_rank,
                        "selected_mean_reciprocal_rank": selected_reciprocal_rank,
                        "selected_collision_rate": selected_collision_rate,
                        "collision_probability_regret": collision_regret,
                        "selected_mean_min_center_distance_m": selected_distance,
                        "distance_regret_m": distance_regret,
                        "misses_collision_opportunity": misses_collision_opportunity,
                        "is_bottom_tail_selection": is_bottom_tail,
                        "is_severe_valid_decision_error": is_severe_decision_error,
                    }
                )

                probabilities, probability_status = _validate_probabilities(
                    raw_probabilities, actions
                )
                detail["probability_status"] = probability_status
                distribution_statuses[probability_status] += 1
                if probabilities is not None:
                    expected_rank = sum(
                        probabilities[candidate] * _mean_rank(outcomes[candidate])
                        for candidate in actions
                    )
                    expected_normalized_rank = (
                        (expected_rank - 1.0) / (len(outcomes) - 1)
                        if len(outcomes) > 1
                        else 0.0
                    )
                    expected_collision_rate = sum(
                        probabilities[candidate]
                        * _collision_rate(outcomes[candidate])
                        for candidate in actions
                    )
                    brier = sum(
                        (probabilities[candidate] - top_distribution[candidate])
                        ** 2
                        for candidate in actions
                    )
                    total_variation = 0.5 * sum(
                        abs(probabilities[candidate] - top_distribution[candidate])
                        for candidate in actions
                    )
                    expected_ranks.append(expected_rank)
                    expected_normalized_ranks.append(expected_normalized_rank)
                    expected_collision_rates.append(expected_collision_rate)
                    brier_scores.append(brier)
                    total_variations.append(total_variation)
                    detail.update(
                        {
                            "target_top_action_distribution": top_distribution,
                            "expected_repetition_rank": expected_rank,
                            "expected_normalized_repetition_rank": expected_normalized_rank,
                            "expected_collision_rate": expected_collision_rate,
                            "top_action_distribution_brier": brier,
                            "top_action_distribution_total_variation": total_variation,
                        }
                    )
        details.append(detail)

    total = len(details)
    extras = len(set(prediction_map) - set(benchmark_map))
    distribution_count = sum(distribution_statuses.values())
    return {
        "evaluation_target": "repeated-validation-soft-labels",
        "examples": total,
        "predictions": len(prediction_map),
        "coverage": coverage / total if total else 0.0,
        "valid_action_rate": valid / coverage if coverage else 0.0,
        "aggregate_top1_accuracy": (
            aggregate_top1_matches / valid if valid else None
        ),
        "modal_top1_accuracy": modal_top1_matches / valid if valid else None,
        "soft_equivalent_action_accuracy": (
            equivalent_matches / valid if valid else None
        ),
        "near_optimal_action_rate": (
            equivalent_matches / valid if valid else None
        ),
        "mean_selected_top_action_frequency": (
            mean(chosen_top_frequencies) if chosen_top_frequencies else None
        ),
        "mean_selected_repetition_rank": (
            mean(chosen_mean_ranks) if chosen_mean_ranks else None
        ),
        "mean_selected_normalized_repetition_rank": (
            mean(chosen_normalized_ranks) if chosen_normalized_ranks else None
        ),
        "mean_selected_reciprocal_rank": (
            mean(chosen_reciprocal_ranks) if chosen_reciprocal_ranks else None
        ),
        "collision_opportunities": collision_opportunities,
        "mean_selected_collision_rate_on_opportunities": (
            mean(chosen_collision_rates_on_opportunities)
            if chosen_collision_rates_on_opportunities
            else None
        ),
        "mean_collision_probability_regret": (
            mean(collision_regrets) if collision_regrets else None
        ),
        "collision_opportunity_miss_rate": (
            collision_opportunity_misses / evaluated_collision_opportunities
            if evaluated_collision_opportunities
            else None
        ),
        "evaluated_collision_opportunities": evaluated_collision_opportunities,
        "mean_distance_regret_m_without_collision_opportunity": (
            mean(distance_regrets_without_collision_opportunity)
            if distance_regrets_without_collision_opportunity
            else None
        ),
        "bottom_tail_selection_rate": (
            bottom_tail_selections / valid if valid else None
        ),
        "severe_valid_decision_error_rate": (
            severe_valid_decision_errors / valid if valid else None
        ),
        "severe_normalized_rank_threshold": severe_normalized_rank_threshold,
        "distance_tolerance_m": distance_tolerance_m,
        "collision_rate_tolerance": collision_rate_tolerance,
        "distribution_predictions": len(expected_ranks),
        "distribution_valid_rate": (
            distribution_statuses["valid"] / distribution_count
            if distribution_count
            else None
        ),
        "probability_status_counts": dict(sorted(distribution_statuses.items())),
        "mean_expected_repetition_rank": (
            mean(expected_ranks) if expected_ranks else None
        ),
        "mean_expected_normalized_repetition_rank": (
            mean(expected_normalized_ranks) if expected_normalized_ranks else None
        ),
        "mean_expected_collision_rate": (
            mean(expected_collision_rates) if expected_collision_rates else None
        ),
        "mean_top_action_distribution_brier": (
            mean(brier_scores) if brier_scores else None
        ),
        "mean_top_action_distribution_total_variation": (
            mean(total_variations) if total_variations else None
        ),
        "extra_predictions": extras,
        "details": details,
    }
