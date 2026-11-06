from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

from .benchmark_eval import read_jsonl
from .demo import adversarial_specification
from .providers.base import DecisionProvider
from .representations import FULL_STATE_V1, validate_profile
from .schema import ScenarioState


def predict_benchmark(
    benchmark_path: Path,
    output_path: Path,
    provider: DecisionProvider,
    *,
    representation_profile: str = FULL_STATE_V1,
    max_examples: Optional[int] = None,
    request_delay_s: float = 0.0,
    reuse_successes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    representation_profile = validate_profile(representation_profile)
    records = read_jsonl(benchmark_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        records = records[:max_examples]
    if request_delay_s < 0.0:
        raise ValueError("request_delay_s cannot be negative")

    reusable = {}
    if reuse_successes_path is not None:
        for previous in read_jsonl(Path(reuse_successes_path)):
            result = previous.get("result", {})
            if (
                previous.get("representation_profile") == representation_profile
                and result.get("action") is not None
                and not result.get("error")
            ):
                reusable[previous["state_json"]] = previous

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    attempt_latencies = []
    models = set()
    total_cost_usd = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    successes = 0
    errors = 0
    reused_successes = 0
    api_attempted_this_run = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, benchmark in enumerate(records):
            state = ScenarioState.from_dict(benchmark["state"])
            specification = adversarial_specification(state)
            previous = reusable.get(state.canonical_json())
            if previous is not None:
                result_data = previous["result"]
                latency = float(result_data.get("latency_ms", 0.0) or 0.0)
                latencies.append(latency)
                models.add(str(result_data.get("model", "unknown")))
                usage = result_data.get("usage", {}) or {}
                total_cost_usd += float(usage.get("cost", 0.0) or 0.0)
                total_input_tokens += int(
                    usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
                )
                total_output_tokens += int(
                    usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
                )
                successes += 1
                reused_successes += 1
                handle.write(
                    json.dumps(
                        previous,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue
            if hasattr(provider, "representation_profile"):
                provider.representation_profile = representation_profile
            attempt_started = time.perf_counter()
            api_attempted_this_run += 1
            try:
                result = provider.decide(state, specification)
                result.validate(specification)
                attempt_latency_ms = (time.perf_counter() - attempt_started) * 1000
                attempt_latencies.append(attempt_latency_ms)
                latencies.append(float(result.latency_ms))
                models.add(str(result.model))
                usage = result.usage or {}
                total_cost_usd += float(usage.get("cost", 0.0) or 0.0)
                total_input_tokens += int(
                    usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
                )
                total_output_tokens += int(
                    usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
                )
                successes += 1
                result_data: Dict[str, Any] = asdict(result)
            except Exception as error:
                attempt_latency_ms = (time.perf_counter() - attempt_started) * 1000
                attempt_latencies.append(attempt_latency_ms)
                errors += 1
                result_data = {
                    "action": None,
                    "probabilities": {},
                    "confidence": None,
                    "provider": provider.name,
                    "model": getattr(provider, "model", "unknown"),
                    "latency_ms": attempt_latency_ms,
                    "usage": {},
                    "raw_response": {},
                    "fallback_used": False,
                    "error": f"{type(error).__name__}: {str(error)[:500]}",
                }
            handle.write(
                json.dumps(
                    {
                        "benchmark_version": benchmark.get("benchmark_version"),
                        "state_json": state.canonical_json(),
                        "state": state.to_dict(),
                        "representation_profile": representation_profile,
                        # Re-render from the canonical state so newly added
                        # representation profiles can be evaluated on older,
                        # already-fixed benchmark files.
                        "model_input": state.to_representation(
                            representation_profile
                        ),
                        "specification": {
                            "name": specification.name,
                            "instructions": specification.instructions,
                            "actions": [
                                action.to_dict() for action in specification.actions
                            ],
                        },
                        "result": result_data,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if request_delay_s and index + 1 < len(records):
                time.sleep(request_delay_s)

    ordered = sorted(latencies)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1) if ordered else 0
    ordered_attempts = sorted(attempt_latencies)
    attempt_p95_index = (
        max(0, int(0.95 * len(ordered_attempts) + 0.999999) - 1)
        if ordered_attempts
        else 0
    )
    return {
        "provider": provider.name,
        "models": sorted(models),
        "representation_profile": representation_profile,
        "attempted": len(records),
        "successes": successes,
        "errors": errors,
        "success_rate": successes / len(records) if records else 0.0,
        "reused_successes": reused_successes,
        "api_attempted_this_run": api_attempted_this_run,
        "total_cost_usd": total_cost_usd,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_latency_ms": mean(latencies) if latencies else None,
        "p95_latency_ms": ordered[p95_index] if ordered else None,
        "mean_attempt_latency_ms": (
            mean(attempt_latencies) if attempt_latencies else None
        ),
        "p95_attempt_latency_ms": (
            ordered_attempts[attempt_p95_index] if ordered_attempts else None
        ),
        "output": str(output_path),
    }
