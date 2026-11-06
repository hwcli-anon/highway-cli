from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from itertools import combinations
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .branch_rollout import compare_rollout_rankings
from .representations import FULL_STATE_V1, REPRESENTATION_PROFILES
from .schema import ScenarioState


ExampleKey = Tuple[str, int]


def _key_from_state(state: Mapping[str, Any]) -> ExampleKey:
    time = state.get("time", {})
    step = time.get("decision_step", state.get("decision_step"))
    return str(state["episode_id"]), int(step)


def _prediction_fields(record: Mapping[str, Any]) -> Tuple[ExampleKey, Optional[str]]:
    state = record.get("state") or record.get("model_input") or record
    key = _key_from_state(state)
    result = record.get("result", {})
    action = record.get("action", result.get("action"))
    return key, str(action) if action is not None else None


def score_records(
    benchmarks: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    target: str = "one-step",
) -> Dict[str, Any]:
    if target not in {"one-step", "validation"}:
        raise ValueError("target must be 'one-step' or 'validation'")
    prediction_map: Dict[ExampleKey, Optional[str]] = {}
    for record in predictions:
        key, action = _prediction_fields(record)
        if key in prediction_map:
            raise ValueError(f"Duplicate prediction for {key}")
        prediction_map[key] = action

    details: List[Dict[str, Any]] = []
    valid_ranks: List[int] = []
    reciprocal_ranks: List[float] = []
    distance_regrets: List[float] = []
    normalized_ranks: List[float] = []
    best_matches = 0
    top3_matches = 0
    tolerance_matches = 0
    covered = 0
    valid = 0
    collision_opportunities = 0
    collision_successes = 0
    benchmark_keys = set()
    versions = set()
    evaluation_horizons = set()
    for benchmark in benchmarks:
        versions.add(str(benchmark.get("benchmark_version", "unknown")))
        state = benchmark["state"]
        key = _key_from_state(state)
        if key in benchmark_keys:
            raise ValueError(f"Duplicate benchmark example {key}")
        benchmark_keys.add(key)
        outcome_field = (
            "action_outcomes" if target == "one-step" else "horizon_action_outcomes"
        )
        if not benchmark.get(outcome_field):
            raise ValueError(
                f"Benchmark example {key} has no {target} rollout outcomes"
            )
        evaluation_horizons.add(
            1
            if target == "one-step"
            else int(benchmark["validation_horizon_policy_steps"])
        )
        outcomes = {item["action"]: item for item in benchmark[outcome_field]}
        best = min(outcomes.values(), key=lambda item: int(item["criticality_rank"]))
        max_collision_rate = max(
            (
                float(item.get("controlled_collision_rate", 1.0))
                if item["av_crash"] and item["controlled_bv_crash"]
                else float(item.get("controlled_collision_rate", 0.0))
            )
            for item in outcomes.values()
        )
        collision_actions = {
            item["action"]
            for item in outcomes.values()
            if max_collision_rate > 0.0
            and float(item.get("controlled_collision_rate", 1.0))
            == max_collision_rate
            and (item["av_crash"] and item["controlled_bv_crash"])
        }
        if collision_actions:
            collision_opportunities += 1
        attempted = key in prediction_map
        action: Optional[str] = prediction_map.get(key)
        detail: Dict[str, Any] = {
            "episode_id": key[0],
            "decision_step": key[1],
            "predicted_action": action,
            "best_action": best["action"],
            "status": "missing",
        }
        if attempted and action is None:
            detail["status"] = "prediction_error"
        elif action is not None:
            covered += 1
            if action not in outcomes:
                detail["status"] = "invalid_action"
            else:
                valid += 1
                selected = outcomes[action]
                rank = int(selected["criticality_rank"])
                valid_ranks.append(rank)
                reciprocal_ranks.append(1.0 / rank)
                normalized_ranks.append(
                    (rank - 1) / (len(outcomes) - 1) if len(outcomes) > 1 else 0.0
                )
                regret = float(selected["min_center_distance_m"]) - float(
                    best["min_center_distance_m"]
                )
                distance_regrets.append(regret)
                is_best = rank == 1
                is_top3 = rank <= 3
                within_tolerance = (
                    action in collision_actions
                    if collision_actions
                    else regret <= 0.25
                )
                best_matches += int(is_best)
                top3_matches += int(is_top3)
                tolerance_matches += int(within_tolerance)
                if collision_actions and action in collision_actions:
                    collision_successes += 1
                detail.update(
                    {
                        "status": "valid",
                        "criticality_rank": rank,
                        "is_best_action": is_best,
                        "is_top3_action": is_top3,
                        "within_0_25m_of_best": within_tolerance,
                        "distance_regret_m": round(regret, 6),
                        "selected_av_crash": bool(selected["av_crash"]),
                        "selected_controlled_collision_rate": float(
                            selected.get(
                                "controlled_collision_rate",
                                bool(
                                    selected["av_crash"]
                                    and selected["controlled_bv_crash"]
                                ),
                            )
                        ),
                        "collision_opportunity": bool(collision_actions),
                    }
                )
        details.append(detail)

    total = len(details)
    extras = len(set(prediction_map) - benchmark_keys)
    return {
        "evaluation_target": target,
        "evaluation_horizon_policy_steps": (
            1
            if target == "one-step"
            else sorted(evaluation_horizons)
        ),
        "benchmark_versions": sorted(versions),
        "examples": total,
        "predictions": len(prediction_map),
        "coverage": covered / total if total else 0.0,
        "valid_action_rate": valid / covered if covered else 0.0,
        "top1_accuracy": best_matches / valid if valid else None,
        "top3_accuracy": top3_matches / valid if valid else None,
        "within_0_25m_accuracy": tolerance_matches / valid if valid else None,
        "mean_criticality_rank": mean(valid_ranks) if valid_ranks else None,
        "mean_normalized_rank": mean(normalized_ranks) if normalized_ranks else None,
        "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "mean_distance_regret_m": mean(distance_regrets) if distance_regrets else None,
        "collision_opportunities": collision_opportunities,
        "collision_choice_success_rate": (
            collision_successes / collision_opportunities
            if collision_opportunities
            else None
        ),
        "extra_predictions": extras,
        "details": details,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_files(
    benchmark_path: Path,
    prediction_path: Path,
    *,
    target: str = "one-step",
) -> Dict[str, Any]:
    return score_records(
        read_jsonl(benchmark_path), read_jsonl(prediction_path), target=target
    )


def _horizon_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    state = row["state"]
    episode_id, decision_step = _key_from_state(state)
    controlled = state.get("controlled_background_vehicle", {})
    vehicle_id = str(controlled.get("vehicle_id", "unknown"))
    return episode_id, decision_step, vehicle_id


def _horizon_and_outcomes(row: Mapping[str, Any]) -> Tuple[int, List[Mapping[str, Any]]]:
    horizon = row.get("validation_horizon_policy_steps")
    outcomes = row.get("horizon_action_outcomes")
    if horizon is None or outcomes is None:
        return 1, list(row["action_outcomes"])
    return int(horizon), list(outcomes)


def summarize_horizon_ablation(
    datasets: Iterable[Iterable[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Compare matched decision states across independently exported horizons."""

    by_horizon: Dict[int, Dict[Tuple[str, int, str], List[Mapping[str, Any]]]] = {}
    for records in datasets:
        for row in records:
            horizon, outcomes = _horizon_and_outcomes(row)
            key = _horizon_key(row)
            horizon_rows = by_horizon.setdefault(horizon, {})
            if key in horizon_rows:
                raise ValueError(f"Duplicate horizon {horizon} example {key}")
            horizon_rows[key] = outcomes
    if 1 not in by_horizon:
        raise ValueError("Horizon ablation requires a one-step benchmark")

    all_key_sets = [set(rows) for rows in by_horizon.values()]
    common_keys = set.intersection(*all_key_sets) if all_key_sets else set()
    details: List[Dict[str, Any]] = []
    pairwise_all: Dict[str, Dict[str, Any]] = {}
    for reference_horizon, comparison_horizon in combinations(sorted(by_horizon), 2):
        comparisons = []
        for key in sorted(common_keys):
            comparison = compare_rollout_rankings(
                by_horizon[reference_horizon][key],
                by_horizon[comparison_horizon][key],
            )
            comparison.update(
                {
                    "episode_id": key[0],
                    "decision_step": key[1],
                    "controlled_vehicle_id": key[2],
                    "reference_horizon": reference_horizon,
                    "comparison_horizon": comparison_horizon,
                }
            )
            comparisons.append(comparison)
            details.append(comparison)
        pairwise_all[f"{reference_horizon}_vs_{comparison_horizon}"] = {
            "matched_examples": len(comparisons),
            "top1_match_rate": (
                mean(item["top1_match"] for item in comparisons)
                if comparisons
                else None
            ),
            "mean_spearman": (
                mean(item["spearman_rank_correlation"] for item in comparisons)
                if comparisons
                else None
            ),
            "mean_reference_best_comparison_rank": (
                mean(item["short_best_long_horizon_rank"] for item in comparisons)
                if comparisons
                else None
            ),
            "mean_top3_jaccard": (
                mean(item["top3_jaccard"] for item in comparisons)
                if comparisons
                else None
            ),
        }

    horizons = sorted(by_horizon)
    pairwise_against_one_step = {
        key: value for key, value in pairwise_all.items() if key.startswith("1_vs_")
    }
    return {
        "horizons": horizons,
        "examples_by_horizon": {
            str(horizon): len(by_horizon[horizon]) for horizon in horizons
        },
        "matched_examples": len(common_keys),
        "all_files_fully_matched": all(
            len(rows) == len(common_keys) for rows in by_horizon.values()
        ),
        "pairwise_against_one_step": pairwise_against_one_step,
        "pairwise_all": pairwise_all,
        "details": details,
    }


def summarize_representation_sizes(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Measure deterministic serialized input size for every representation."""

    rows = list(records)
    sizes: Dict[str, List[int]] = {profile: [] for profile in REPRESENTATION_PROFILES}
    for row in rows:
        state = ScenarioState.from_dict(row["state"])
        for profile in REPRESENTATION_PROFILES:
            payload = state.canonical_representation(profile)
            sizes[profile].append(len(payload.encode("utf-8")))
    full_mean = mean(sizes[FULL_STATE_V1]) if sizes[FULL_STATE_V1] else None
    profiles = {}
    for profile in REPRESENTATION_PROFILES:
        values = sizes[profile]
        profile_mean = mean(values) if values else None
        profiles[profile] = {
            "examples": len(values),
            "utf8_bytes_min": min(values) if values else None,
            "utf8_bytes_mean": profile_mean,
            "utf8_bytes_max": max(values) if values else None,
            "relative_to_full_state": (
                profile_mean / full_mean
                if profile_mean is not None and full_mean
                else None
            ),
        }
    return {"examples": len(rows), "profiles": profiles}


def summarize_strata(
    records: Iterable[Mapping[str, Any]],
    *,
    low_ttc_threshold_s: float = 5.0,
    near_tie_threshold_m: float = 0.25,
) -> Dict[str, Any]:
    """Describe predeclared safety-relevant strata without selecting examples."""

    if low_ttc_threshold_s <= 0.0 or near_tie_threshold_m < 0.0:
        raise ValueError("Stratum thresholds must be non-negative and TTC must be positive")
    manifest = []
    for row in records:
        state = row["state"]
        episode_id, decision_step = _key_from_state(state)
        outcomes = row.get("horizon_action_outcomes") or row["action_outcomes"]
        ranked = sorted(outcomes, key=lambda item: int(item["criticality_rank"]))
        collision_opportunity = any(
            bool(item["av_crash"] and item["controlled_bv_crash"])
            or float(item.get("controlled_collision_rate", 0.0)) > 0.0
            for item in ranked
        )
        current_ttc = state.get("risk", {}).get("ttc_s")
        low_ttc = current_ttc is not None and float(current_ttc) <= low_ttc_threshold_s
        lane_conflict = bool(state.get("risk", {}).get("lane_conflict", False))
        reversal = bool(
            row.get("horizon_validation")
            and not row["horizon_validation"]["top1_match"]
        )
        near_tie = False
        if len(ranked) > 1 and not ranked[0]["av_crash"] and not ranked[1]["av_crash"]:
            separation = float(ranked[1]["min_center_distance_m"]) - float(
                ranked[0]["min_center_distance_m"]
            )
            near_tie = separation <= near_tie_threshold_m
        action_count = len(ranked)
        action_count_bin = (
            "1-10" if action_count <= 10 else "11-20" if action_count <= 20 else "21+"
        )
        manifest.append(
            {
                "episode_id": episode_id,
                "decision_step": decision_step,
                "controlled_vehicle_id": state.get(
                    "controlled_background_vehicle", {}
                ).get("vehicle_id"),
                "collision_opportunity": collision_opportunity,
                "low_ttc": low_ttc,
                "lane_conflict": lane_conflict,
                "one_step_three_step_reversal": reversal,
                "near_tie": near_tie,
                "action_count": action_count,
                "action_count_bin": action_count_bin,
            }
        )

    stratum_names = (
        "collision_opportunity",
        "low_ttc",
        "lane_conflict",
        "one_step_three_step_reversal",
        "near_tie",
    )
    total = len(manifest)
    all_clusters = {
        (item["episode_id"], item["controlled_vehicle_id"]) for item in manifest
    }
    collision_clusters = {
        (item["episode_id"], item["controlled_vehicle_id"])
        for item in manifest
        if item["collision_opportunity"]
    }
    return {
        "stratification_version": "highway-benchmark-strata-v1",
        "examples": total,
        "thresholds": {
            "low_ttc_s": low_ttc_threshold_s,
            "near_tie_distance_m": near_tie_threshold_m,
        },
        "counts": {
            name: sum(bool(item[name]) for item in manifest) for name in stratum_names
        },
        "rates": {
            name: (
                sum(bool(item[name]) for item in manifest) / total if total else None
            )
            for name in stratum_names
        },
        "action_count_bins": dict(
            sorted(Counter(item["action_count_bin"] for item in manifest).items())
        ),
        "interaction_clusters": {
            "episode_vehicle_clusters": len(all_clusters),
            "collision_opportunity_clusters": len(collision_clusters),
            "collision_opportunity_cluster_rate": (
                len(collision_clusters) / len(all_clusters) if all_clusters else None
            ),
        },
        "manifest": manifest,
    }


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    candidate_counts = []
    collision_examples = 0
    validation_collision_examples = 0
    best_actions: Counter = Counter()
    best_kinds: Counter = Counter()
    separation_m = []
    random_top1 = []
    random_top3 = []
    random_mrr = []
    stability_consensus = []
    stability_spearman = []
    stability_matches = []
    horizon_top1_matches = []
    horizon_spearman = []
    horizon_short_best_ranks = []
    horizon_top3_jaccard = []
    horizon_stability_consensus = []
    horizon_stability_spearman = []
    horizon_stability_matches = []
    versions = set()
    for row in rows:
        versions.add(str(row.get("benchmark_version", "unknown")))
        outcomes = sorted(
            row["action_outcomes"], key=lambda item: int(item["criticality_rank"])
        )
        candidate_counts.append(len(outcomes))
        if outcomes:
            count = len(outcomes)
            random_top1.append(1.0 / count)
            random_top3.append(min(3, count) / count)
            random_mrr.append(sum(1.0 / rank for rank in range(1, count + 1)) / count)
        collision_examples += int(
            any(
                float(item.get("controlled_collision_rate", 0.0)) > 0.0
                or (item["av_crash"] and item["controlled_bv_crash"])
                for item in outcomes
            )
        )
        horizon_outcomes = row.get("horizon_action_outcomes") or []
        validation_collision_examples += int(
            any(
                float(item.get("controlled_collision_rate", 0.0)) > 0.0
                or (item["av_crash"] and item["controlled_bv_crash"])
                for item in horizon_outcomes
            )
        )
        stability = row.get("stability")
        if stability:
            stability_consensus.append(float(stability["consensus_top_action_rate"]))
            stability_spearman.append(float(stability["mean_pairwise_spearman"]))
            stability_matches.append(bool(stability["aggregate_best_matches_consensus"]))
        horizon_validation = row.get("horizon_validation")
        if horizon_validation:
            horizon_top1_matches.append(bool(horizon_validation["top1_match"]))
            horizon_spearman.append(
                float(horizon_validation["spearman_rank_correlation"])
            )
            horizon_short_best_ranks.append(
                int(horizon_validation["short_best_long_horizon_rank"])
            )
            horizon_top3_jaccard.append(float(horizon_validation["top3_jaccard"]))
        horizon_stability = row.get("horizon_stability")
        if horizon_stability:
            horizon_stability_consensus.append(
                float(horizon_stability["consensus_top_action_rate"])
            )
            horizon_stability_spearman.append(
                float(horizon_stability["mean_pairwise_spearman"])
            )
            horizon_stability_matches.append(
                bool(horizon_stability["aggregate_best_matches_consensus"])
            )
        if outcomes:
            best_actions[outcomes[0]["action"]] += 1
            best_kinds[outcomes[0]["kind"]] += 1
        if len(outcomes) > 1 and not outcomes[0]["av_crash"]:
            separation_m.append(
                float(outcomes[1]["min_center_distance_m"])
                - float(outcomes[0]["min_center_distance_m"])
            )
    return {
        "benchmark_versions": sorted(versions),
        "examples": len(rows),
        "candidate_actions_min": min(candidate_counts) if candidate_counts else None,
        "candidate_actions_mean": mean(candidate_counts) if candidate_counts else None,
        "candidate_actions_max": max(candidate_counts) if candidate_counts else None,
        "collision_opportunity_rate": (
            collision_examples / len(rows) if rows else None
        ),
        "one_step_collision_opportunity_rate": (
            collision_examples / len(rows) if rows else None
        ),
        "validation_collision_opportunity_rate": (
            validation_collision_examples / len(rows) if rows else None
        ),
        "mean_best_second_distance_separation_m": (
            mean(separation_m) if separation_m else None
        ),
        "best_second_within_0_10m_rate": (
            sum(value <= 0.10 for value in separation_m) / len(separation_m)
            if separation_m
            else None
        ),
        "best_second_within_0_25m_rate": (
            sum(value <= 0.25 for value in separation_m) / len(separation_m)
            if separation_m
            else None
        ),
        "uniform_random_top1_expectation": mean(random_top1) if random_top1 else None,
        "uniform_random_top3_expectation": mean(random_top3) if random_top3 else None,
        "uniform_random_mrr_expectation": mean(random_mrr) if random_mrr else None,
        "stability_examples": len(stability_consensus),
        "mean_consensus_top_action_rate": (
            mean(stability_consensus) if stability_consensus else None
        ),
        "mean_pairwise_spearman": (
            mean(stability_spearman) if stability_spearman else None
        ),
        "aggregate_best_consensus_match_rate": (
            mean(stability_matches) if stability_matches else None
        ),
        "horizon_validation_examples": len(horizon_top1_matches),
        "one_step_long_horizon_top1_match_rate": (
            mean(horizon_top1_matches) if horizon_top1_matches else None
        ),
        "mean_one_step_long_horizon_spearman": (
            mean(horizon_spearman) if horizon_spearman else None
        ),
        "mean_one_step_best_long_horizon_rank": (
            mean(horizon_short_best_ranks) if horizon_short_best_ranks else None
        ),
        "mean_one_step_long_horizon_top3_jaccard": (
            mean(horizon_top3_jaccard) if horizon_top3_jaccard else None
        ),
        "horizon_stability_examples": len(horizon_stability_consensus),
        "mean_horizon_consensus_top_action_rate": (
            mean(horizon_stability_consensus)
            if horizon_stability_consensus
            else None
        ),
        "mean_horizon_pairwise_spearman": (
            mean(horizon_stability_spearman)
            if horizon_stability_spearman
            else None
        ),
        "horizon_aggregate_best_consensus_match_rate": (
            mean(horizon_stability_matches) if horizon_stability_matches else None
        ),
        "best_action_counts": dict(sorted(best_actions.items())),
        "best_action_kind_counts": dict(sorted(best_kinds.items())),
    }
