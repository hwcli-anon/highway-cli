from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _rows(paths: Iterable[Path]):
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize_model_reliability(paths: Iterable[Path]) -> Dict[str, Any]:
    decisions = 0
    first_pass_valid = 0
    eventual_valid = 0
    fallbacks = 0
    api_calls = 0
    format_retry_calls = 0
    errors: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    stage_first_pass: Counter[str] = Counter()

    for row in _rows(paths):
        decisions += 1
        result: Mapping[str, Any] = row.get("result", {})
        raw: Mapping[str, Any] = result.get("raw_response", {}) or {}
        reliability: Mapping[str, Any] = raw.get("reliability", {}) or {}
        stages = reliability.get("stages", []) or []
        if reliability.get("first_pass_valid") is not None:
            row_first_pass = bool(reliability.get("first_pass_valid"))
        else:
            row_first_pass = bool(stages) and all(
                bool(stage.get("first_pass_valid")) for stage in stages
            )
        if reliability.get("eventual_valid") is not None:
            row_eventual = bool(reliability.get("eventual_valid"))
        else:
            row_eventual = bool(stages) and all(
                bool(stage.get("eventual_valid")) for stage in stages
            )
        first_pass_valid += int(row_first_pass)
        eventual_valid += int(row_eventual and not result.get("fallback_used"))
        fallbacks += int(bool(result.get("fallback_used")))
        usage = result.get("usage", {}) or {}
        api_calls += int(usage.get("api_calls", 0) or 0)
        for stage in stages:
            name = str(stage.get("stage", "unknown"))
            stage_counts[name] += 1
            stage_first_pass[name] += int(bool(stage.get("first_pass_valid")))
            format_retry_calls += max(0, int(stage.get("attempt_count", 0) or 0) - 1)
        if result.get("error"):
            errors[str(result["error"]).split(":", 1)[0]] += 1

    def rate(count: int) -> Any:
        return count / decisions if decisions else None

    return {
        "trace_version": "model-reliability-summary-v1",
        "decisions": decisions,
        "first_pass_valid": first_pass_valid,
        "first_pass_valid_rate": rate(first_pass_valid),
        "eventual_valid_without_fallback": eventual_valid,
        "eventual_valid_without_fallback_rate": rate(eventual_valid),
        "fallbacks": fallbacks,
        "fallback_rate": rate(fallbacks),
        "observed_api_calls": api_calls,
        "format_retry_calls": format_retry_calls,
        "stage_validity": {
            stage: {
                "count": count,
                "first_pass_valid": stage_first_pass[stage],
                "first_pass_valid_rate": stage_first_pass[stage] / count,
            }
            for stage, count in sorted(stage_counts.items())
        },
        "error_types": dict(sorted(errors.items())),
    }
