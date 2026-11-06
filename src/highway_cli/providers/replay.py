from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import DecisionProvider
from ..schema import DecisionResult, DecisionSpecification, ScenarioState


class ReplayProvider(DecisionProvider):
    """Replay previously recorded provider results by canonical state hash."""

    name = "replay"

    def __init__(self, trace_path: Path):
        self.records: Dict[str, Dict[str, Any]] = {}
        with Path(trace_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                self.records[record["state_json"]] = record["result"]

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        key = state.canonical_json()
        if key not in self.records:
            raise KeyError("No replay result exists for this canonical state")
        record = self.records[key]
        result = DecisionResult(**record)
        original_provider = result.provider
        original_model = result.model
        original_latency_ms = result.latency_ms
        original_usage = dict(result.usage or {})
        result.raw_response = dict(result.raw_response)
        result.raw_response["replay_source"] = {
            "provider": original_provider,
            "model": original_model,
            "latency_ms": original_latency_ms,
            "usage": original_usage,
        }
        result.provider = self.name
        result.latency_ms = 0.0
        result.usage = {
            "api_calls": 0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result.validate(specification)
        return result

