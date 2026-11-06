from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .benchmark_eval import read_jsonl
from .benchmark_split import collection_seed, example_key
from .hierarchical_actions import MANEUVERS, hierarchical_action_structure, maneuver_for_action
from .schema import ActionOption


def _best_horizon_action(row: Mapping[str, Any]) -> str:
    aggregate = row.get("horizon_stability", {}).get("aggregate_best_action")
    if aggregate is not None:
        return str(aggregate)
    outcomes = row.get("horizon_action_outcomes") or row.get("action_outcomes") or []
    return str(min(outcomes, key=lambda item: int(item["criticality_rank"]))["action"])


def select_maneuver_ab(
    benchmark_path: Path,
    *,
    output_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Freeze every development state with more than one feasible maneuver."""

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen maneuver A/B selection")
    source = read_jsonl(benchmark_path)
    selected = []
    for row in source:
        actions = [ActionOption(**item) for item in row["state"]["available_actions"]]
        hierarchy = hierarchical_action_structure(actions)
        if len(hierarchy["first_stage_maneuvers"]) > 1:
            selected.append(row)
    selected.sort(key=lambda row: (collection_seed(row), example_key(row)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            encoded = (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)

    best_maneuvers = Counter(maneuver_for_action(_best_horizon_action(row)) for row in selected)
    seed_counts = Counter(collection_seed(row) for row in selected)
    manifest = {
        "selection_version": "highway-maneuver-ab-selection-v1",
        "rule": "Include every frozen development state with at least two feasible first-stage maneuvers.",
        "label_used_for_inclusion": False,
        "source_states": len(source),
        "selected_states": len(selected),
        "best_maneuver_counts_for_analysis_only": {
            maneuver: best_maneuvers.get(maneuver, 0) for maneuver in MANEUVERS
        },
        "seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "output": str(output_path),
        "sha256": digest.hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
