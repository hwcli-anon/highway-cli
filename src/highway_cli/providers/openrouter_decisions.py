from __future__ import annotations

import os
import time
from typing import Optional

from .base import DecisionProvider
from .http import post_json
from ..representations import RISK_AWARE_V1, validate_profile
from ..schema import DecisionResult, DecisionSpecification, ScenarioState


OPENROUTER_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


class OpenRouterDecisionsProvider(DecisionProvider):
    """Provider for models served through OpenRouter's Decisions API."""

    name = "openrouter-decisions"

    def __init__(
        self,
        model: str,
        *,
        provider_name: Optional[str] = None,
        api_key: str = "",
        endpoint: str = OPENROUTER_DECISIONS_ENDPOINT,
        timeout_s: float = 30.0,
        representation_profile: str = RISK_AWARE_V1,
    ):
        self.model = model
        self.name = provider_name or self.name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.representation_profile = validate_profile(representation_profile)

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        payload = {
            "model": self.model,
            # A deterministic JSON string gives both decision models the same
            # textual input while preserving the selected representation.
            "state": state.canonical_representation(self.representation_profile),
            "questions": {
                specification.name: {
                    "type": "choice",
                    "instructions": specification.instructions,
                    "criteria": {
                        action.value: action.description
                        for action in specification.actions
                    },
                }
            },
        }
        started = time.perf_counter()
        response = post_json(self.endpoint, self.api_key, payload, self.timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000

        answers = response.get("answers")
        if not isinstance(answers, dict) or specification.name not in answers:
            raise ValueError(
                f"OpenRouter response is missing answer {specification.name!r}"
            )
        answer = answers[specification.name]
        if answer.get("type") == "refusal":
            raise RuntimeError(
                f"OpenRouter Decisions refused question {specification.name}"
            )

        raw_probabilities = {
            key: float(value)
            for key, value in answer.get("probabilities", {}).items()
        }
        # Decisions responses may round displayed probabilities (for example,
        # to a total of 0.99). Preserve their relative mass while enforcing the
        # probability contract used by every other provider adapter.
        probability_total = sum(raw_probabilities.values())
        probabilities = (
            {
                key: value / probability_total
                for key, value in raw_probabilities.items()
            }
            if probability_total > 0.0
            else raw_probabilities
        )
        result = DecisionResult(
            action=answer["choice"],
            probabilities=probabilities,
            confidence=(
                float(answer["confidence"])
                if answer.get("confidence") is not None
                else None
            ),
            provider=self.name,
            model=response.get("model", self.model),
            latency_ms=latency_ms,
            usage={**(response.get("usage", {}) or {}), "api_calls": 1},
            raw_response=response,
        )
        result.validate(specification)
        return result
