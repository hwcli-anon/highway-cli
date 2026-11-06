from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .benchmark_eval import read_jsonl


_EPISODE_PATTERN = re.compile(r"^nade-(?P<seed>-?\d+)-(?P<episode>\d+)$")


def collection_seed(row: Mapping[str, Any]) -> int:
    explicit = row.get("collection_seed")
    if explicit is not None:
        return int(explicit)
    episode_id = str(row["state"]["episode_id"])
    match = _EPISODE_PATTERN.fullmatch(episode_id)
    if not match:
        raise ValueError(
            f"Cannot infer collection seed from episode_id {episode_id!r}; "
            "regenerate the benchmark with collection_seed metadata"
        )
    return int(match.group("seed"))


def example_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    state = row["state"]
    time = state.get("time", {})
    return (
        str(state["episode_id"]),
        int(time.get("decision_step", state.get("decision_step"))),
        str(state["controlled_background_vehicle"]["vehicle_id"]),
    )


def _canonical_line(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = (_canonical_line(row) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def prepare_seed_splits(
    benchmark_paths: Sequence[Path],
    *,
    development_seeds: Sequence[int],
    confirmation_seeds: Sequence[int],
    development_output: Path,
    confirmation_output: Path,
) -> Dict[str, Any]:
    development_seed_set = set(development_seeds)
    confirmation_seed_set = set(confirmation_seeds)
    if not development_seed_set or not confirmation_seed_set:
        raise ValueError("Both development and confirmation seed sets are required")
    overlap = development_seed_set & confirmation_seed_set
    if overlap:
        raise ValueError(f"Development and confirmation seeds overlap: {sorted(overlap)}")

    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for path in benchmark_paths:
        loaded = read_jsonl(path)
        rows.extend(loaded)
        source_counts[str(path)] = len(loaded)

    seen = set()
    duplicate_keys = []
    development = []
    confirmation = []
    ignored_seed_counts: Dict[int, int] = {}
    observed_seed_counts: Dict[int, int] = {}
    for row in rows:
        key = example_key(row)
        if key in seen:
            duplicate_keys.append(key)
            continue
        seen.add(key)
        seed = collection_seed(row)
        observed_seed_counts[seed] = observed_seed_counts.get(seed, 0) + 1
        if seed in development_seed_set:
            development.append(row)
        elif seed in confirmation_seed_set:
            confirmation.append(row)
        else:
            ignored_seed_counts[seed] = ignored_seed_counts.get(seed, 0) + 1
    if duplicate_keys:
        raise ValueError(f"Duplicate benchmark examples found: {duplicate_keys[:5]}")

    missing_development = development_seed_set - set(observed_seed_counts)
    missing_confirmation = confirmation_seed_set - set(observed_seed_counts)
    if missing_development or missing_confirmation:
        raise ValueError(
            "Requested seeds are missing: "
            f"development={sorted(missing_development)}, "
            f"confirmation={sorted(missing_confirmation)}"
        )

    development_hash = _write_jsonl(Path(development_output), development)
    confirmation_hash = _write_jsonl(Path(confirmation_output), confirmation)
    return {
        "split_version": "highway-seed-split-v1",
        "source_files": source_counts,
        "observed_seed_counts": {
            str(seed): count for seed, count in sorted(observed_seed_counts.items())
        },
        "development": {
            "seeds": sorted(development_seed_set),
            "examples": len(development),
            "output": str(development_output),
            "sha256": development_hash,
        },
        "confirmation": {
            "seeds": sorted(confirmation_seed_set),
            "examples": len(confirmation),
            "output": str(confirmation_output),
            "sha256": confirmation_hash,
        },
        "ignored_seed_counts": {
            str(seed): count for seed, count in sorted(ignored_seed_counts.items())
        },
        "seed_disjoint": True,
        "duplicate_examples": 0,
    }
