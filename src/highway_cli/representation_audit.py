from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from .representations import REPRESENTATION_PROFILES
from .schema import ScenarioState


# Rollout outcomes are benchmark labels and must never be visible to a model.
FORBIDDEN_LABEL_NAMES = {
    "action_outcomes",
    "horizon_action_outcomes",
    "criticality_rank",
    "is_best_action",
    "av_crash",
    "controlled_bv_crash",
    "min_center_distance_m",
    "min_abs_longitudinal_gap_m",
    "collision_rate",
    "mean_criticality_rank",
}

# These are available inside NADE, but are not portable observable scene state.
SIMULATOR_INTERNAL_NAMES = {
    "nade_criticality",
    "candidate_index",
    "nade_native_observation",
}

IDENTIFIER_NAMES = {
    "episode_id",
    "decision_step",
    "simulation_time_s",
}


def _walk(value: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, (*path, "[]"))


def _matching_paths(payload: Dict[str, Any], names: set[str]) -> List[str]:
    return sorted({".".join(path) for path, _ in _walk(payload) if path[-1] in names})


def audit_state(state: ScenarioState) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for profile in REPRESENTATION_PROFILES:
        payload = state.to_representation(profile)
        forbidden = _matching_paths(payload, FORBIDDEN_LABEL_NAMES)
        internal = _matching_paths(payload, SIMULATOR_INTERNAL_NAMES)
        identifiers = _matching_paths(payload, IDENTIFIER_NAMES)
        profiles[profile] = {
            "forbidden_label_paths": forbidden,
            "simulator_internal_paths": internal,
            "identifier_paths": identifiers,
            "passes_no_label_leakage_check": not forbidden,
            "portable_primary_candidate": not forbidden and not internal and not identifiers,
        }
    return profiles


def audit_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    counters = {
        profile: Counter(
            examples=0,
            forbidden_label_examples=0,
            simulator_internal_examples=0,
            identifier_examples=0,
            portable_primary_examples=0,
        )
        for profile in REPRESENTATION_PROFILES
    }
    observed_paths = {
        profile: {"forbidden": set(), "internal": set(), "identifiers": set()}
        for profile in REPRESENTATION_PROFILES
    }

    for record in records:
        per_profile = audit_state(ScenarioState.from_dict(record["state"]))
        for profile, result in per_profile.items():
            counters[profile]["examples"] += 1
            if result["forbidden_label_paths"]:
                counters[profile]["forbidden_label_examples"] += 1
            if result["simulator_internal_paths"]:
                counters[profile]["simulator_internal_examples"] += 1
            if result["identifier_paths"]:
                counters[profile]["identifier_examples"] += 1
            if result["portable_primary_candidate"]:
                counters[profile]["portable_primary_examples"] += 1
            observed_paths[profile]["forbidden"].update(result["forbidden_label_paths"])
            observed_paths[profile]["internal"].update(result["simulator_internal_paths"])
            observed_paths[profile]["identifiers"].update(result["identifier_paths"])

    profile_reports = {}
    for profile in REPRESENTATION_PROFILES:
        counts = dict(counters[profile])
        profile_reports[profile] = {
            **counts,
            "observed_forbidden_label_paths": sorted(observed_paths[profile]["forbidden"]),
            "observed_simulator_internal_paths": sorted(observed_paths[profile]["internal"]),
            "observed_identifier_paths": sorted(observed_paths[profile]["identifiers"]),
            "recommended_role": (
                "primary_portable_representation"
                if counts["portable_primary_examples"] == counts["examples"]
                else "auxiliary_or_privileged_control"
            ),
        }

    return {
        "audit_version": "highway-representation-audit-v1",
        "examples": len(records),
        "direct_label_leakage_detected": any(
            report["forbidden_label_examples"] for report in profile_reports.values()
        ),
        "profile_reports": profile_reports,
        "interpretation": {
            "forbidden_label": "Simulator rollout outcomes or ranks visible to the model.",
            "simulator_internal": "NADE-specific internal information; not future-label leakage, but not portable scene state.",
            "identifier": "Trace-matching metadata without driving semantics; retained only in auxiliary profiles.",
        },
    }
