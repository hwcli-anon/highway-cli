from __future__ import annotations

import os
import time

from .base import DecisionProvider
from .http import post_json
from ..schema import DecisionResult, DecisionSpecification, ScenarioState
from ..representations import RISK_AWARE_V1, validate_profile


class OpenAIDecisionsProvider(DecisionProvider):
    name = "openai-decisions"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        api_key: str = "",
        endpoint: str = "https://api.openai.com/v1/decisions",
        timeout_s: float = 5.0,
        representation_profile: str = RISK_AWARE_V1,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.representation_profile = validate_profile(representation_profile)

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        payload = {
            "model": self.model,
            "input": state.canonical_representation(self.representation_profile),
            "questions": [
                {
                    "type": "choice",
                    "name": specification.name,
                    "instructions": specification.instructions,
                    "choices": [
                        {"value": action.value, "description": action.description}
                        for action in specification.actions
                    ],
                }
            ],
        }
        started = time.perf_counter()
        response = post_json(self.endpoint, self.api_key, payload, self.timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000
        answer = next(
            item for item in response["answers"] if item.get("name") == specification.name
        )
        if answer.get("type") == "refusal":
            raise RuntimeError(f"OpenAI Decisions refused question {specification.name}")
        result = DecisionResult(
            action=answer["choice"],
            probabilities={key: float(value) for key, value in answer["probabilities"].items()},
            confidence=float(answer["confidence"]),
            provider=self.name,
            model=response.get("model", self.model),
            latency_ms=latency_ms,
            usage={**(response.get("usage", {}) or {}), "api_calls": 1},
            raw_response=response,
        )
        result.validate(specification)
        return result

