from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .benchmark_eval import read_jsonl, summarize_strata
from .benchmark_split import collection_seed, example_key


STRATA = (
    "collision_opportunity",
    "low_ttc",
    "lane_conflict",
    "one_step_three_step_reversal",
    "near_tie",
    "ordinary_control",
)


def _canonical_line(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_frozen_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen pilot dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (_canonical_line(row) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def _flags(row: Mapping[str, Any]) -> Dict[str, bool]:
    item = summarize_strata([row])["manifest"][0]
    flags = {name: bool(item[name]) for name in STRATA[:-1]}
    flags["ordinary_control"] = not any(
        flags[name]
        for name in (
            "collision_opportunity",
            "low_ttc",
            "lane_conflict",
            "one_step_three_step_reversal",
        )
    )
    return flags


def select_pilot(
    benchmark_paths: Sequence[Path],
    *,
    output_path: Path,
    manifest_path: Path,
    max_states: int = 12,
    min_per_stratum: int = 2,
) -> Dict[str, Any]:
    if max_states < 1 or min_per_stratum < 0:
        raise ValueError("max_states must be positive and min_per_stratum non-negative")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite pilot manifest: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for path in benchmark_paths:
        loaded = read_jsonl(path)
        rows.extend(loaded)
        source_counts[str(path)] = len(loaded)

    seen_keys = set()
    candidates_by_cluster: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = example_key(row)
        if key in seen_keys:
            raise ValueError(f"Duplicate benchmark example: {key}")
        seen_keys.add(key)
        seed = collection_seed(row)
        flags = _flags(row)
        cluster = (key[0], key[2])
        candidate = {"row": row, "key": key, "seed": seed, "flags": flags}
        incumbent = candidates_by_cluster.get(cluster)
        if incumbent is None:
            candidates_by_cluster[cluster] = candidate
        else:
            candidate_score = sum(flags.values())
            incumbent_score = sum(incumbent["flags"].values())
            if (-candidate_score, key) < (-incumbent_score, incumbent["key"]):
                candidates_by_cluster[cluster] = candidate

    candidates = list(candidates_by_cluster.values())
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    seed_counts: Counter = Counter()
    stratum_counts: Counter = Counter()

    def add(candidate: Dict[str, Any]) -> None:
        selected.append(candidate)
        selected_keys.add(candidate["key"])
        seed_counts[candidate["seed"]] += 1
        for name, present in candidate["flags"].items():
            stratum_counts[name] += int(present)

    for stratum in STRATA:
        while stratum_counts[stratum] < min_per_stratum and len(selected) < max_states:
            eligible = [
                candidate
                for candidate in candidates
                if candidate["key"] not in selected_keys and candidate["flags"][stratum]
            ]
            if not eligible:
                break
            deficient_coverage = {
                candidate["key"]: sum(
                    candidate["flags"][name]
                    and stratum_counts[name] < min_per_stratum
                    for name in STRATA
                )
                for candidate in eligible
            }
            chosen = min(
                eligible,
                key=lambda candidate: (
                    seed_counts[candidate["seed"]],
                    -deficient_coverage[candidate["key"]],
                    candidate["seed"],
                    candidate["key"],
                ),
            )
            add(chosen)

    while len(selected) < min(max_states, len(candidates)):
        remaining = [
            candidate for candidate in candidates if candidate["key"] not in selected_keys
        ]
        chosen = min(
            remaining,
            key=lambda candidate: (
                seed_counts[candidate["seed"]],
                sum(candidate["flags"].values()),
                candidate["seed"],
                candidate["key"],
            ),
        )
        add(chosen)

    selected_rows = [candidate["row"] for candidate in selected]
    digest = _write_frozen_jsonl(output_path, selected_rows)
    manifest = {
        "pilot_version": "highway-api-pilot-v1",
        "purpose": "API schema validation and prompt/parameter freezing only",
        "selection_bias_warning": (
            "This set deliberately balances rare safety strata and must not be used "
            "to estimate population prevalence or as a confirmatory test set."
        ),
        "source_files": source_counts,
        "source_examples": len(rows),
        "source_interaction_clusters": len(candidates),
        "selected_examples": len(selected),
        "selected_interaction_clusters": len(selected),
        "max_states": max_states,
        "min_per_stratum_requested": min_per_stratum,
        "stratum_counts": {name: stratum_counts[name] for name in STRATA},
        "seed_counts": {
            str(seed): count for seed, count in sorted(seed_counts.items())
        },
        "output": str(output_path),
        "sha256": digest,
        "selection": [
            {
                "episode_id": candidate["key"][0],
                "decision_step": candidate["key"][1],
                "controlled_vehicle_id": candidate["key"][2],
                "collection_seed": candidate["seed"],
                "strata": [
                    name for name, present in candidate["flags"].items() if present
                ],
            }
            for candidate in selected
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
