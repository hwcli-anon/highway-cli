from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .benchmark_eval import read_jsonl
from .benchmark_split import collection_seed, example_key
from .coverage_audit import coverage_features


def _canonical_line(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selection_score(key: Tuple[str, int, str], selection_seed: int) -> str:
    payload = f"{selection_seed}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_development_states(
    census_paths: Sequence[Path],
    *,
    output_path: Path,
    manifest_path: Path,
    selection_seed: int = 20261013,
) -> Dict[str, Any]:
    """Freeze one label-blind representative per episode/BV interaction cluster."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen development selection")
    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for path in census_paths:
        loaded = read_jsonl(path)
        rows.extend(loaded)
        source_counts[str(path)] = len(loaded)

    by_cluster: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    seen = set()
    for row in rows:
        key = example_key(row)
        if key in seen:
            raise ValueError(f"Duplicate census state: {key}")
        seen.add(key)
        by_cluster.setdefault((key[0], key[2]), []).append(row)

    selected = []
    for cluster in sorted(by_cluster):
        chosen = min(
            by_cluster[cluster],
            key=lambda row: (_selection_score(example_key(row), selection_seed), example_key(row)),
        )
        selected.append(chosen)
    selected.sort(key=lambda row: (collection_seed(row), example_key(row)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            encoded = (_canonical_line(row) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)

    seed_counts = Counter(collection_seed(row) for row in selected)
    relation_counts = Counter()
    maneuver_counts = Counter()
    for row in selected:
        features = coverage_features(row["state"])
        relation_counts[features["relation_cell"]] += 1
        maneuver_counts[features["maneuver_support"]] += 1
    manifest = {
        "selection_version": "highway-development-selection-v1",
        "selection_rule": (
            "One representative per (episode_id, controlled_vehicle_id) interaction "
            "cluster, chosen by the minimum SHA-256 score of the example key under a "
            "fixed selection seed. No rollout label or provider output is used."
        ),
        "selection_seed": selection_seed,
        "source_files": source_counts,
        "source_states": len(rows),
        "source_interaction_clusters": len(by_cluster),
        "selected_states": len(selected),
        "selected_interaction_clusters": len(selected),
        "seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "relation_counts": dict(sorted(relation_counts.items())),
        "maneuver_support_counts": dict(sorted(maneuver_counts.items())),
        "output": str(output_path),
        "sha256": digest.hexdigest(),
        "selected_keys": [
            {
                "episode_id": example_key(row)[0],
                "decision_step": example_key(row)[1],
                "controlled_vehicle_id": example_key(row)[2],
                "collection_seed": collection_seed(row),
            }
            for row in selected
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def freeze_development_benchmark(
    benchmark_paths: Sequence[Path],
    *,
    output_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Merge already labeled development shards into one immutable benchmark."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen development benchmark")
    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    seen = set()
    for path in benchmark_paths:
        loaded = read_jsonl(path)
        source_counts[str(path)] = len(loaded)
        for row in loaded:
            key = example_key(row)
            if key in seen:
                raise ValueError(f"Duplicate labeled development state: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (collection_seed(row), example_key(row)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (_canonical_line(row) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)

    seed_counts = Counter(collection_seed(row) for row in rows)
    clusters = {
        (example_key(row)[0], example_key(row)[2])
        for row in rows
    }
    manifest = {
        "benchmark_version": "highway-frozen-development-benchmark-v1",
        "role": "representation and decision-method development only; not confirmation",
        "source_files": source_counts,
        "states": len(rows),
        "unique_state_keys": len(seen),
        "interaction_clusters": len(clusters),
        "seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "output": str(output_path),
        "sha256": digest.hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
