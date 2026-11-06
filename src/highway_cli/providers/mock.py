from __future__ import annotations

import time
from typing import Dict

from .base import DecisionProvider
from ..schema import DecisionResult, DecisionSpecification, ScenarioState


class MockProvider(DecisionProvider):
    """Deterministic offline provider for testing the full framework."""

    name = "mock"

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        started = time.perf_counter()
        values = [action.value for action in specification.actions]
        if not values:
            raise ValueError("Decision specification has no actions")

        # Prefer hard braking for short TTC, otherwise zero acceleration when
        # available. This is a test fixture, not a scientific baseline.
        if state.risk.ttc_s is not None and state.risk.ttc_s < 3.0:
            numeric = []
            for value in values:
                try:
                    numeric.append((float(value), value))
                except ValueError:
                    continue
            chosen = min(numeric)[1] if numeric else values[0]
        elif "0.0" in values:
            chosen = "0.0"
        else:
            chosen = values[0]

        probability = 1.0 / len(values)
        probabilities: Dict[str, float] = {value: probability for value in values}
        probabilities[chosen] += 0.5
        normalizer = sum(probabilities.values())
        probabilities = {key: value / normalizer for key, value in probabilities.items()}
        result = DecisionResult(
            action=chosen,
            probabilities=probabilities,
            confidence=max(probabilities.values()),
            provider=self.name,
            model="deterministic-fixture-v1",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response={"fixture": True},
        )
        result.validate(specification)
        return result

