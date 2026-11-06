from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .benchmark_eval import read_jsonl
from .benchmark_split import example_key
from .representations import NADE_NATIVE_V1
from .schema import ScenarioState


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _state_without_native_observation(state: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(state)
    result.pop("nade_native_observation", None)
    return result


def enrich_benchmark_with_native_observations(
    benchmark_path: Path,
    census_paths: Iterable[Path],
    *,
    output_path: Path,
) -> Dict[str, Any]:
    """Attach replayed exact NADE six-slot observations to fixed labels.

    Every non-native state field must match byte-for-byte after canonical JSON
    serialization. This prevents accidentally transferring observations across
    a different stochastic trajectory while preserving the original labels.
    """

    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite enriched benchmark: {output_path}")

    census_by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    source_paths: List[str] = []
    for path in census_paths:
        path = Path(path)
        source_paths.append(str(path))
        for row in read_jsonl(path):
            key = example_key(row)
            if key in census_by_key:
                raise ValueError(f"Duplicate census key: {key}")
            census_by_key[key] = row

    benchmark_rows = list(read_jsonl(benchmark_path))
    enriched: List[Dict[str, Any]] = []
    missing = []
    mismatched = []
    for row in benchmark_rows:
        key = example_key(row)
        census = census_by_key.get(key)
        if census is None:
            missing.append(key)
            continue
        original_state = row["state"]
        replayed_state = census["state"]
        if _canonical(_state_without_native_observation(original_state)) != _canonical(
            _state_without_native_observation(replayed_state)
        ):
            mismatched.append(key)
            continue
        native_observation = replayed_state.get("nade_native_observation")
        if native_observation is None:
            raise ValueError(f"Census state lacks native observation: {key}")
        updated_state = dict(original_state)
        updated_state["nade_native_observation"] = native_observation
        updated = dict(row)
        updated["state"] = updated_state
        representations = dict(updated.get("representations", {}))
        representations[NADE_NATIVE_V1] = ScenarioState.from_dict(
            updated_state
        ).to_representation(NADE_NATIVE_V1)
        updated["representations"] = representations
        enriched.append(updated)

    if missing or mismatched:
        raise ValueError(
            "Cannot enrich benchmark: "
            f"missing={len(missing)}, mismatched={len(mismatched)}, "
            f"first_missing={missing[:1]}, first_mismatched={mismatched[:1]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(_canonical(row) + "\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "enrichment_version": "highway-nade-native-enrichment-v1",
        "benchmark": str(benchmark_path),
        "census_sources": source_paths,
        "output": str(output_path),
        "states": len(enriched),
        "exact_non_native_state_matches": len(enriched),
        "missing_states": 0,
        "mismatched_states": 0,
        "sha256": digest,
        "labels_recomputed": False,
        "claim_boundary": (
            "Only exact live NADE six-slot observations were attached; "
            "all original rollout labels and non-native state fields were preserved."
        ),
    }
