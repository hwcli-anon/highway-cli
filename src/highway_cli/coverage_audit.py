from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


LONGITUDINAL_RELATIONS = ("ahead", "alongside", "behind")
LATERAL_RELATIONS = ("left", "same_lane", "right")


def _bucket(value: float, edges: Tuple[float, ...], labels: Tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def coverage_features(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return state-only coverage features without rollout-label information."""

    av = state["autonomous_vehicle"]
    bv = state["controlled_background_vehicle"]
    lane_delta = int(bv["lane_index"]) - int(av["lane_index"])
    lateral_relation = "left" if lane_delta < 0 else "right" if lane_delta > 0 else "same_lane"
    gap = float(bv["relative_longitudinal_m"])
    if gap > 0.5:
        longitudinal_relation = "ahead"
    elif gap < -0.5:
        longitudinal_relation = "behind"
    else:
        longitudinal_relation = "alongside"

    relative_speed = float(bv["relative_speed_mps"])
    absolute_gap_rate = (
        relative_speed if gap > 0.5 else -relative_speed if gap < -0.5 else 0.0
    )
    if absolute_gap_rate < -0.5:
        gap_trend = "closing"
    elif absolute_gap_rate > 0.5:
        gap_trend = "opening"
    else:
        gap_trend = "stable"

    time_to_overlap = None
    if abs(gap) > 0.5 and absolute_gap_rate < -1e-9:
        time_to_overlap = abs(gap) / abs(absolute_gap_rate)
    if time_to_overlap is None:
        overlap_bin = "not_closing"
    elif time_to_overlap <= 3.0:
        overlap_bin = "0-3s"
    elif time_to_overlap <= 5.0:
        overlap_bin = "3-5s"
    elif time_to_overlap <= 10.0:
        overlap_bin = "5-10s"
    else:
        overlap_bin = "10s+"

    actions = [str(item["value"]) for item in state.get("available_actions", [])]
    left = "LANE_LEFT" in actions
    right = "LANE_RIGHT" in actions
    maneuver_support = (
        "keep+left+right"
        if left and right
        else "keep+left"
        if left
        else "keep+right"
        if right
        else "keep_only"
    )
    history = state.get("recent_history", {})
    history_available = bool(history.get("history_available", False))
    history_lane_changed = bool(history.get("lane_changed", False)) if history_available else False
    nearby_count = len(state.get("nearby_vehicles", []))
    return {
        "lane_pair": f"AV{int(av['lane_index'])}-BV{int(bv['lane_index'])}",
        "lane_delta": lane_delta,
        "lateral_relation": lateral_relation,
        "longitudinal_relation": longitudinal_relation,
        "relation_cell": f"{lateral_relation}:{longitudinal_relation}",
        "gap_trend": gap_trend,
        "time_to_longitudinal_overlap_bin": overlap_bin,
        "absolute_gap_bin": _bucket(
            abs(gap),
            (5.0, 15.0, 30.0, 60.0),
            ("0-5m", "5-15m", "15-30m", "30-60m", "60m+"),
        ),
        "relative_speed_bin": _bucket(
            relative_speed,
            (-5.0, -1.0, 1.0, 5.0),
            ("<-5", "-5--1", "-1-1", "1-5", ">5"),
        ),
        "maneuver_support": maneuver_support,
        "nearby_density_bin": (
            "0-4" if nearby_count <= 4 else "5-7" if nearby_count <= 7 else "8-10"
        ),
        "history_available": history_available,
        "history_lane_changed": history_lane_changed,
        "candidate_index": history.get("candidate_index"),
        "action_count": len(actions),
        "available_actions": actions,
    }


def _state_key(state: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (
        str(state["episode_id"]),
        int(state["time"]["decision_step"]),
        str(state["controlled_background_vehicle"]["vehicle_id"]),
    )


def _cluster_key(state: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(state["episode_id"]),
        str(state["controlled_background_vehicle"]["vehicle_id"]),
    )


def audit_coverage(
    records: Iterable[Mapping[str, Any]],
    *,
    minimum_clusters_per_relation_cell: int = 5,
    minimum_clusters_per_maneuver_support: int = 10,
) -> Dict[str, Any]:
    rows = list(records)
    source_types = sorted(
        {
            "decision_census"
            if row.get("census_version")
            else "labeled_benchmark"
            if row.get("benchmark_version")
            else "unknown"
            for row in rows
        }
    )
    feature_rows: List[Dict[str, Any]] = []
    state_keys = set()
    cluster_keys = set()
    clusters_by_relation: Dict[str, set] = {}
    clusters_by_maneuver: Dict[str, set] = {}
    action_available = Counter()
    action_best = Counter()
    states_per_cluster = Counter()

    count_fields = (
        "lane_pair",
        "relation_cell",
        "gap_trend",
        "time_to_longitudinal_overlap_bin",
        "absolute_gap_bin",
        "relative_speed_bin",
        "maneuver_support",
        "nearby_density_bin",
        "history_available",
        "history_lane_changed",
        "candidate_index",
    )
    counts: Dict[str, Counter] = {field: Counter() for field in count_fields}

    for row in rows:
        state = row["state"]
        features = coverage_features(state)
        state_key = _state_key(state)
        cluster_key = _cluster_key(state)
        state_keys.add(state_key)
        cluster_keys.add(cluster_key)
        states_per_cluster[cluster_key] += 1
        for field in count_fields:
            counts[field][str(features[field])] += 1
        clusters_by_relation.setdefault(features["relation_cell"], set()).add(cluster_key)
        clusters_by_maneuver.setdefault(features["maneuver_support"], set()).add(cluster_key)
        action_available.update(features["available_actions"])

        outcomes = row.get("horizon_action_outcomes") or row.get("action_outcomes") or []
        best_action: Optional[str] = None
        if row.get("horizon_stability", {}).get("aggregate_best_action") is not None:
            best_action = str(row["horizon_stability"]["aggregate_best_action"])
        elif outcomes:
            best_action = str(
                min(outcomes, key=lambda item: int(item["criticality_rank"]))["action"]
            )
        if best_action is not None:
            action_best[best_action] += 1
        feature_rows.append(
            {
                "episode_id": state_key[0],
                "decision_step": state_key[1],
                "controlled_vehicle_id": state_key[2],
                **{key: value for key, value in features.items() if key != "available_actions"},
                "best_action_if_labeled": best_action,
            }
        )

    expected_relation_cells = [
        f"{lateral}:{longitudinal}"
        for lateral in LATERAL_RELATIONS
        for longitudinal in LONGITUDINAL_RELATIONS
    ]
    relation_cluster_counts = {
        cell: len(clusters_by_relation.get(cell, set())) for cell in expected_relation_cells
    }
    maneuver_cells = ("keep_only", "keep+left", "keep+right", "keep+left+right")
    maneuver_cluster_counts = {
        cell: len(clusters_by_maneuver.get(cell, set())) for cell in maneuver_cells
    }
    per_cluster_values = list(states_per_cluster.values())
    candidate_zero = counts["candidate_index"].get("0", 0)
    has_labeled_benchmark = any(row.get("benchmark_version") for row in rows)
    collection_description = (
        "The source includes labeled benchmark records: only NADE-selected critical "
        "decision points are eligible, and benchmark collection may stop at a global "
        "state cap. These are design properties, not statistical tests of bias."
        if has_labeled_benchmark
        else "The census records every NADE-selected critical decision point in the "
        "requested episodes without a benchmark state cap. It does not cover candidate "
        "background vehicles that NADE did not select."
    )
    return {
        "audit_version": "highway-development-coverage-v1",
        "scope": "observed collected states; not proof of simulator reachability",
        "source_types": source_types,
        "examples": len(rows),
        "unique_state_keys": len(state_keys),
        "interaction_clusters": len(cluster_keys),
        "states_per_cluster": {
            "min": min(per_cluster_values) if per_cluster_values else None,
            "mean": mean(per_cluster_values) if per_cluster_values else None,
            "max": max(per_cluster_values) if per_cluster_values else None,
        },
        "feature_counts": {
            field: dict(sorted(counter.items())) for field, counter in counts.items()
        },
        "relation_cell_cluster_counts": relation_cluster_counts,
        "maneuver_support_cluster_counts": maneuver_cluster_counts,
        "coverage_gaps": {
            "unobserved_relation_cells": [
                cell for cell, count in relation_cluster_counts.items() if count == 0
            ],
            "undercovered_relation_cells": [
                cell
                for cell, count in relation_cluster_counts.items()
                if count < minimum_clusters_per_relation_cell
            ],
            "undercovered_maneuver_support": [
                cell
                for cell, count in maneuver_cluster_counts.items()
                if count < minimum_clusters_per_maneuver_support
            ],
        },
        "action_coverage": {
            "unique_available_actions": len(action_available),
            "availability_counts": dict(sorted(action_available.items())),
            "unique_best_actions": len(action_best),
            "best_action_counts": dict(sorted(action_best.items())),
        },
        "sampling_bias_indicators": {
            "candidate_index_zero_rate": (
                candidate_zero / len(rows) if rows else None
            ),
            "first_hit_global_cap_present_in_source": has_labeled_benchmark,
            "benchmark_requires_nade_selected_action": True,
            "interpretation": collection_description,
        },
        "thresholds": {
            "minimum_clusters_per_relation_cell": minimum_clusters_per_relation_cell,
            "minimum_clusters_per_maneuver_support": minimum_clusters_per_maneuver_support,
            "role": "development coverage targets, not inferential sample-size claims",
        },
        "details": feature_rows,
    }


def coverage_progression(
    record_groups: Iterable[Tuple[str, Iterable[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    """Summarize cumulative coverage as independently collected groups are added."""

    cumulative: List[Mapping[str, Any]] = []
    stages: List[Dict[str, Any]] = []
    previous_relations = set()
    previous_maneuvers = set()
    for label, records in record_groups:
        cumulative.extend(records)
        report = audit_coverage(cumulative)
        observed_relations = {
            key
            for key, count in report["relation_cell_cluster_counts"].items()
            if count > 0
        }
        observed_maneuvers = {
            key
            for key, count in report["maneuver_support_cluster_counts"].items()
            if count > 0
        }
        stages.append(
            {
                "through_group": label,
                "examples": report["examples"],
                "interaction_clusters": report["interaction_clusters"],
                "observed_relation_cells": sorted(observed_relations),
                "new_relation_cells": sorted(observed_relations - previous_relations),
                "relation_cell_cluster_counts": report[
                    "relation_cell_cluster_counts"
                ],
                "observed_maneuver_support": sorted(observed_maneuvers),
                "new_maneuver_support": sorted(
                    observed_maneuvers - previous_maneuvers
                ),
                "maneuver_support_cluster_counts": report[
                    "maneuver_support_cluster_counts"
                ],
            }
        )
        previous_relations = observed_relations
        previous_maneuvers = observed_maneuvers
    return {
        "progression_version": "highway-development-coverage-progression-v1",
        "group_order_is_meaningful": True,
        "stages": stages,
    }


def audit_candidate_selection(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Locate where relation types disappear in NADE's candidate-selection pipeline."""

    rows = list(records)
    stage_counts = {
        "candidate": Counter(),
        "nade_interaction_case": Counter(),
        "positive_criticality": Counter(),
        "selected": Counter(),
    }
    cases_by_relation: Dict[str, Counter] = {}
    trends_by_relation: Dict[str, Counter] = {}
    gaps_by_relation: Dict[str, List[float]] = {}
    relative_speeds_by_relation: Dict[str, List[float]] = {}
    relevant_surrogate_probs_by_relation: Dict[str, List[float]] = {}
    av_lane_by_step: Dict[Tuple[Any, str, int], int] = {}
    steps = set()
    for row in rows:
        state = row["state"]
        relation = coverage_features(state)["relation_cell"]
        features = coverage_features(state)
        stage_counts["candidate"][relation] += 1
        interaction_case = row.get("nade_interaction_case", "unknown")
        cases_by_relation.setdefault(relation, Counter())[interaction_case] += 1
        if interaction_case not in ("av_not_relevant_neighbor", "unknown"):
            stage_counts["nade_interaction_case"][relation] += 1
        trends_by_relation.setdefault(relation, Counter())[features["gap_trend"]] += 1
        bv = state["controlled_background_vehicle"]
        gaps_by_relation.setdefault(relation, []).append(
            abs(float(bv["relative_longitudinal_m"]))
        )
        relative_speeds_by_relation.setdefault(relation, []).append(
            float(bv["relative_speed_mps"])
        )
        lateral = features["lateral_relation"]
        probability_key = (
            "left" if lateral == "left" else "right" if lateral == "right" else "keep_lane"
        )
        probabilities = row.get("surrogate_av_action_probabilities", {})
        if probability_key in probabilities:
            relevant_surrogate_probs_by_relation.setdefault(relation, []).append(
                float(probabilities[probability_key])
            )
        step_key = (
            row.get("collection_seed"),
            str(state["episode_id"]),
            int(state["time"]["decision_step"]),
        )
        av_lane_by_step[step_key] = int(state["autonomous_vehicle"]["lane_index"])
        if row.get("positive_criticality"):
            stage_counts["positive_criticality"][relation] += 1
        if row.get("nade_selected"):
            stage_counts["selected"][relation] += 1
        steps.add((row.get("collection_seed"), state["episode_id"], state["time"]["decision_step"]))

    expected = [
        f"{lateral}:{longitudinal}"
        for lateral in LATERAL_RELATIONS
        for longitudinal in LONGITUDINAL_RELATIONS
    ]
    cells = {}
    for relation in expected:
        candidate = stage_counts["candidate"][relation]
        matched = stage_counts["nade_interaction_case"][relation]
        positive = stage_counts["positive_criticality"][relation]
        selected = stage_counts["selected"][relation]
        if candidate == 0:
            disappearance_stage = "absent_among_nearest_candidates"
        elif matched == 0:
            disappearance_stage = "av_not_relevant_neighbor_for_nade_case"
        elif positive == 0:
            disappearance_stage = "matched_nade_case_but_zero_criticality"
        elif selected == 0:
            disappearance_stage = "loses_highest_criticality_selection"
        else:
            disappearance_stage = "survives_selection"
        cells[relation] = {
            "candidate_states": candidate,
            "nade_case_matched_states": matched,
            "positive_criticality_states": positive,
            "selected_states": selected,
            "positive_criticality_rate": positive / candidate if candidate else None,
            "selection_rate_given_candidate": selected / candidate if candidate else None,
            "selection_rate_given_positive": selected / positive if positive else None,
            "nade_interaction_case_counts": dict(
                sorted(cases_by_relation.get(relation, Counter()).items())
            ),
            "gap_trend_counts": dict(
                sorted(trends_by_relation.get(relation, Counter()).items())
            ),
            "mean_absolute_gap_m": (
                mean(gaps_by_relation[relation]) if gaps_by_relation.get(relation) else None
            ),
            "mean_relative_speed_mps": (
                mean(relative_speeds_by_relation[relation])
                if relative_speeds_by_relation.get(relation)
                else None
            ),
            "relevant_surrogate_action_probability_positive_states": sum(
                value > 0.0
                for value in relevant_surrogate_probs_by_relation.get(relation, [])
            ),
            "mean_relevant_surrogate_action_probability": (
                mean(relevant_surrogate_probs_by_relation[relation])
                if relevant_surrogate_probs_by_relation.get(relation)
                else None
            ),
            "disappearance_stage": disappearance_stage,
        }
    av_lane_counts = Counter(av_lane_by_step.values())
    return {
        "audit_version": "highway-nade-candidate-selection-audit-v1",
        "candidate_states": len(rows),
        "decision_steps": len(steps),
        "mean_candidates_per_step": len(rows) / len(steps) if steps else None,
        "av_lane_decision_step_counts": {
            str(key): value for key, value in sorted(av_lane_counts.items())
        },
        "relation_cells": cells,
        "interpretation": (
            "Counts diagnose the public NADE pipeline; they do not imply that relation "
            "cells should be balanced in development or validation data."
        ),
    }
