from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, Mapping, Optional

from .base import DecisionProvider
from .http import post_json
from .openrouter_decisions import OPENROUTER_DECISIONS_ENDPOINT
from ..interaction_graph import INTERACTION_GRAPH_INTENT_V1
from ..post_maneuver_conflict import (
    POST_MANEUVER_CONFLICT_V1,
    POST_MANEUVER_CONFLICT_V2,
    STAGE_ROUTED_CONFLICT_V1,
)
from ..schema import DecisionResult, DecisionSpecification, ScenarioState


class IntentHierarchicalOpenRouterDecisionsProvider(DecisionProvider):
    """Two-stage intent-then-action decision over unchanged NADE leaves.

    The first Decisions request chooses one observable interaction intent.  The
    second request, when needed, chooses only among original NADE actions that
    are compatible with the selected intent.  Returned leaf probabilities are
    conditional on that selected intent; the raw response retains both stages.
    """

    name = "openrouter-decisions-intent-hierarchical"

    def __init__(
        self,
        model: str,
        *,
        provider_name: Optional[str] = None,
        api_key: str = "",
        endpoint: str = OPENROUTER_DECISIONS_ENDPOINT,
        timeout_s: float = 30.0,
        representation_profile: str = INTERACTION_GRAPH_INTENT_V1,
    ):
        if representation_profile not in {
            INTERACTION_GRAPH_INTENT_V1,
            POST_MANEUVER_CONFLICT_V1,
            POST_MANEUVER_CONFLICT_V2,
            STAGE_ROUTED_CONFLICT_V1,
        }:
            raise ValueError(
                "Intent-hierarchical mode requires an intent-interface representation"
            )
        self.model = model
        self.name = provider_name or self.name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.representation_profile = representation_profile

    @staticmethod
    def _answer(response: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        answers = response.get("answers")
        if not isinstance(answers, dict) or name not in answers:
            raise ValueError(f"OpenRouter response is missing answer {name!r}")
        answer = answers[name]
        if answer.get("type") == "refusal":
            raise RuntimeError(f"OpenRouter Decisions refused question {name}")
        if "choice" not in answer:
            raise ValueError(f"OpenRouter answer {name!r} has no choice")
        return answer

    @staticmethod
    def _usage(*responses: Mapping[str, Any]) -> Dict[str, Any]:
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        for response in responses:
            usage = response.get("usage", {}) or {}
            cost += float(usage.get("cost", 0.0) or 0.0)
            input_tokens += int(
                usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
            )
            output_tokens += int(
                usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
            )
        return {
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "api_calls": len(responses),
        }

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        representation = copy.deepcopy(
            state.to_representation(self.representation_profile)
        )
        allowed_by_specification = {
            action.value: action for action in specification.actions
        }
        for item in representation["intent_interface"]["intent_candidates"]:
            compatible = [
                str(value)
                for value in item["compatible_leaf_actions"]
                if str(value) in allowed_by_specification
            ]
            item["compatible_leaf_actions"] = compatible
            item["available"] = bool(item["available"] and compatible)
        intents = {
            str(item["intent_id"]): item
            for item in representation["intent_interface"]["intent_candidates"]
            if item["available"]
        }
        if not intents:
            raise ValueError("No interaction intent is available")
        if self.representation_profile in {
            POST_MANEUVER_CONFLICT_V2,
            STAGE_ROUTED_CONFLICT_V1,
        }:
            post_maneuver_guidance = (
                " Compare the post-maneuver cards with AV as the adversarial safety "
                "target. Prioritize justified AV interaction; use ordinary target-lane "
                "traffic only as feasibility and plausibility context, not as a separate "
                "attack target."
            )
        elif "post_maneuver_conflict_interface" in representation:
            post_maneuver_guidance = (
                " Compare the post-maneuver conflict cards, including target-lane "
                "traffic, AV relation, and follow-up longitudinal pressure, before "
                "choosing the intent."
            )
        else:
            post_maneuver_guidance = ""
        intent_payload = {
            "model": self.model,
            "state": json.dumps(
                representation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "questions": {
                "interaction_intent": {
                    "type": "choice",
                    "instructions": (
                        "Choose the observable interaction intent most likely to expose "
                        "a safety weakness while remaining physically plausible. Use the "
                        "scene graph and applicability fields; they are current-state "
                        "evidence, not future rollout labels."
                        + post_maneuver_guidance
                    ),
                    "criteria": {
                        intent_id: str(item["purpose"])
                        for intent_id, item in intents.items()
                    },
                }
            },
        }
        started = time.perf_counter()
        intent_response = post_json(
            self.endpoint, self.api_key, intent_payload, self.timeout_s
        )
        intent_answer = self._answer(intent_response, "interaction_intent")
        selected_intent = str(intent_answer["choice"])
        if selected_intent not in intents:
            raise ValueError(f"Provider selected unavailable intent: {selected_intent}")

        compatible = [
            str(value)
            for value in intents[selected_intent]["compatible_leaf_actions"]
            if str(value) in allowed_by_specification
        ]
        if not compatible:
            raise ValueError(
                f"Selected intent {selected_intent!r} has no executable NADE action"
            )

        action_response: Optional[Mapping[str, Any]] = None
        if len(compatible) == 1:
            selected_action = compatible[0]
            conditional_probabilities = {selected_action: 1.0}
        else:
            if self.representation_profile == STAGE_ROUTED_CONFLICT_V1:
                conditioned_state = copy.deepcopy(
                    state.to_representation(INTERACTION_GRAPH_INTENT_V1)
                )
                leaf_instruction = (
                    "Use the original interaction graph and current-state longitudinal "
                    "relations for this fine-grained leaf choice."
                )
            else:
                conditioned_state = dict(representation)
                leaf_instruction = (
                    "Use any post-maneuver conflict card as current-state "
                    "counterfactual evidence, not as a rollout label."
                )
            conditioned_state["selected_interaction_intent"] = {
                "intent_id": selected_intent,
                "compatible_leaf_actions": compatible,
                "observable_applicability": intents[selected_intent][
                    "observable_applicability"
                ],
            }
            action_payload = {
                "model": self.model,
                "state": json.dumps(
                    conditioned_state,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "questions": {
                    "action_given_intent": {
                        "type": "choice",
                        "instructions": (
                            f"Given the selected intent {selected_intent}, choose the "
                            "compatible original NADE action that best realizes it. "
                            + leaf_instruction
                        ),
                        "criteria": {
                            value: allowed_by_specification[value].description
                            for value in compatible
                        },
                    }
                },
            }
            action_response = post_json(
                self.endpoint, self.api_key, action_payload, self.timeout_s
            )
            action_answer = self._answer(action_response, "action_given_intent")
            selected_action = str(action_answer["choice"])
            if selected_action not in compatible:
                raise ValueError(
                    f"Provider selected action incompatible with intent: {selected_action}"
                )
            raw_probabilities = {
                str(key): float(value)
                for key, value in action_answer.get("probabilities", {}).items()
                if str(key) in compatible
            }
            total = sum(raw_probabilities.values())
            conditional_probabilities = (
                {
                    value: raw_probabilities.get(value, 0.0) / total
                    for value in compatible
                }
                if total > 0.0
                else {value: float(value == selected_action) for value in compatible}
            )

        probabilities = {
            action.value: conditional_probabilities.get(action.value, 0.0)
            for action in specification.actions
        }
        responses = (
            (intent_response, action_response)
            if action_response is not None
            else (intent_response,)
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = DecisionResult(
            action=selected_action,
            probabilities=probabilities,
            confidence=probabilities.get(selected_action),
            provider=self.name,
            model=str(responses[-1].get("model", self.model)),
            latency_ms=latency_ms,
            usage=self._usage(*responses),
            raw_response={
                "decision_mode": "intent-hierarchical-v1",
                "selected_intent": selected_intent,
                "intent_probabilities": intent_answer.get("probabilities", {}),
                "leaf_probabilities_are_conditional_on_selected_intent": True,
                "intent_stage_representation": self.representation_profile,
                "leaf_stage_representation": (
                    INTERACTION_GRAPH_INTENT_V1
                    if self.representation_profile == STAGE_ROUTED_CONFLICT_V1
                    else self.representation_profile
                ),
                "intent_response": intent_response,
                "action_response": action_response,
            },
        )
        result.validate(specification)
        return result
