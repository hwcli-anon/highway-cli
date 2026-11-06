from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .soft_label_eval import score_soft_records


DetailKey = Tuple[str, int, str]


def _detail_key(row: Mapping[str, Any]) -> DetailKey:
    return (
        str(row["episode_id"]),
        int(row["decision_step"]),
        str(row["controlled_vehicle_id"]),
    )


def _exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if not discordant:
        return 1.0
    lower = min(baseline_only, candidate_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_summary(
    pairs: Sequence[Tuple[float, float]],
    *,
    higher_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    if not pairs:
        return {
            "paired_examples": 0,
            "higher_is_better": higher_is_better,
            "baseline_mean": None,
            "candidate_mean": None,
            "candidate_minus_baseline": None,
            "improvement": None,
            "bootstrap_95_ci_candidate_minus_baseline": None,
        }
    baseline_values = [pair[0] for pair in pairs]
    candidate_values = [pair[1] for pair in pairs]
    differences = [candidate - baseline for baseline, candidate in pairs]
    delta = mean(differences)
    interval = None
    if bootstrap_samples > 0:
        rng = random.Random(bootstrap_seed)
        bootstrap_means: List[float] = []
        for _ in range(bootstrap_samples):
            bootstrap_means.append(
                mean(differences[rng.randrange(len(differences))] for _ in differences)
            )
        interval = [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ]
    return {
        "paired_examples": len(pairs),
        "higher_is_better": higher_is_better,
        "baseline_mean": mean(baseline_values),
        "candidate_mean": mean(candidate_values),
        "candidate_minus_baseline": delta,
        "improvement": delta if higher_is_better else -delta,
        "bootstrap_95_ci_candidate_minus_baseline": interval,
    }


def compare_input_quality_records(
    benchmarks: Iterable[Mapping[str, Any]],
    baseline_predictions: Iterable[Mapping[str, Any]],
    candidate_predictions: Iterable[Mapping[str, Any]],
    *,
    selection: Optional[Iterable[Mapping[str, Any]]] = None,
    distance_tolerance_m: float = 0.25,
    collision_rate_tolerance: float = 0.0,
    severe_normalized_rank_threshold: float = 0.75,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20261013,
    comparison_status: str = "development_ab_not_confirmation",
) -> Dict[str, Any]:
    """Compare two representations with paired repeated-rollout outcomes.

    This deliberately reports a metric vector rather than a weighted composite
    score. Positive ``improvement`` always favors the candidate, while the raw
    ``candidate_minus_baseline`` retains the metric's natural direction.
    """

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    benchmark_rows = list(benchmarks)
    selection_rows = list(selection) if selection is not None else None
    common = {
        "selection": selection_rows,
        "distance_tolerance_m": distance_tolerance_m,
        "collision_rate_tolerance": collision_rate_tolerance,
        "severe_normalized_rank_threshold": severe_normalized_rank_threshold,
    }
    baseline = score_soft_records(
        benchmark_rows, list(baseline_predictions), **common
    )
    candidate = score_soft_records(
        benchmark_rows, list(candidate_predictions), **common
    )
    baseline_details = {_detail_key(row): row for row in baseline["details"]}
    candidate_details = {_detail_key(row): row for row in candidate["details"]}
    if set(baseline_details) != set(candidate_details):
        raise ValueError("Baseline and candidate evaluation keys do not match")

    paired = [
        (baseline_details[key], candidate_details[key])
        for key in sorted(baseline_details)
        if baseline_details[key]["status"] == "valid"
        and candidate_details[key]["status"] == "valid"
    ]

    def values(
        field: str,
        *,
        include=lambda _old, _new: True,
    ) -> List[Tuple[float, float]]:
        return [
            (float(old[field]), float(new[field]))
            for old, new in paired
            if include(old, new)
        ]

    metric_pairs = {
        "near_optimal_action_rate": (
            values("is_soft_equivalent_action"),
            True,
        ),
        "mean_normalized_repetition_rank": (
            values("selected_normalized_repetition_rank"),
            False,
        ),
        "mean_collision_probability_regret": (
            values(
                "collision_probability_regret",
                include=lambda old, new: bool(old["collision_opportunity"])
                and bool(new["collision_opportunity"]),
            ),
            False,
        ),
        "collision_opportunity_miss_rate": (
            values(
                "misses_collision_opportunity",
                include=lambda old, new: bool(old["collision_opportunity"])
                and bool(new["collision_opportunity"]),
            ),
            False,
        ),
        "mean_distance_regret_m_without_collision_opportunity": (
            values(
                "distance_regret_m",
                include=lambda old, new: not bool(old["collision_opportunity"])
                and not bool(new["collision_opportunity"]),
            ),
            False,
        ),
        "bottom_tail_selection_rate": (
            values("is_bottom_tail_selection"),
            False,
        ),
        "severe_valid_decision_error_rate": (
            values("is_severe_valid_decision_error"),
            False,
        ),
    }
    metrics = {
        name: _paired_summary(
            pairs,
            higher_is_better=higher_is_better,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, (name, (pairs, higher_is_better)) in enumerate(
            metric_pairs.items()
        )
    }

    binary_pairs = metric_pairs["near_optimal_action_rate"][0]
    baseline_only = sum(old == 1.0 and new == 0.0 for old, new in binary_pairs)
    candidate_only = sum(old == 0.0 and new == 1.0 for old, new in binary_pairs)
    metrics["near_optimal_action_rate"]["exact_mcnemar_p_two_sided"] = (
        _exact_mcnemar_p(baseline_only, candidate_only)
    )
    metrics["near_optimal_action_rate"]["baseline_only_near_optimal"] = (
        baseline_only
    )
    metrics["near_optimal_action_rate"]["candidate_only_near_optimal"] = (
        candidate_only
    )

    return {
        "comparison_status": comparison_status,
        "evaluation_target": "repeated-three-step-soft-labels",
        "paired_valid_examples": len(paired),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metric_policy": (
            "No composite score. Positive improvement favors the candidate; "
            "raw candidate-minus-baseline retains each metric's direction."
        ),
        "baseline": {key: value for key, value in baseline.items() if key != "details"},
        "candidate": {key: value for key, value in candidate.items() if key != "details"},
        "paired_metrics": metrics,
    }
