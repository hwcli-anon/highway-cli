from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from .base import DecisionProvider
from .http import post_json
from .openrouter_decisions import OPENROUTER_DECISIONS_ENDPOINT
from ..hierarchical_actions import (
    CHANGE_LEFT,
    CHANGE_RIGHT,
    KEEP_LANE,
    hierarchical_action_structure,
)
from ..representations import HIERARCHICAL_RELATIONAL_V1, validate_profile
from ..schema import DecisionResult, DecisionSpecification, ScenarioState


class HierarchicalOpenRouterDecisionsProvider(DecisionProvider):
    """One-request maneuver-first decision over original NADE action leaves.

    The request contains a maneuver question and, when longitudinal actions are
    available, a conditional acceleration question. Their probabilities are
    composed into a normalized distribution over the unchanged NADE leaves.
    """

    name = "openrouter-decisions-hierarchical"

    def __init__(
        self,
        model: str,
        *,
        provider_name: Optional[str] = None,
        api_key: str = "",
        endpoint: str = OPENROUTER_DECISIONS_ENDPOINT,
        timeout_s: float = 30.0,
        representation_profile: str = HIERARCHICAL_RELATIONAL_V1,
    ):
        self.model = model
        self.name = provider_name or self.name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.representation_profile = validate_profile(representation_profile)

    @staticmethod
    def _answer(response: Dict[str, Any], name: str) -> Dict[str, Any]:
        answers = response.get("answers")
        if not isinstance(answers, dict) or name not in answers:
            raise ValueError(f"OpenRouter response is missing answer {name!r}")
        answer = answers[name]
        if answer.get("type") == "refusal":
            raise RuntimeError(f"OpenRouter Decisions refused question {name}")
        if "choice" not in answer:
            raise ValueError(f"OpenRouter answer {name!r} has no choice")
        return answer

    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        hierarchy = hierarchical_action_structure(specification.actions)
        maneuver_items = hierarchy["first_stage_maneuvers"]
        maneuver_criteria = {
            item["maneuver"]: item["description"] for item in maneuver_items
        }
        longitudinal = hierarchy["second_stage_actions"].get(KEEP_LANE, [])
        questions: Dict[str, Any] = {}
        if len(maneuver_criteria) > 1:
            questions["maneuver"] = {
                "type": "choice",
                "instructions": (
                    "Choose the physically feasible maneuver most likely to expose a "
                    "safety weakness of the autonomous vehicle while remaining plausible. "
                    "Judge the interaction first; do not favor a maneuver because it has "
                    "more leaf actions."
                ),
                "criteria": maneuver_criteria,
            }
        if len(longitudinal) > 1:
            questions["longitudinal_action_given_keep_lane"] = {
                "type": "choice",
                "instructions": (
                    "Conditional on KEEP_LANE, choose the listed original NADE "
                    "longitudinal action that best serves the same safety-testing goal."
                ),
                "criteria": {
                    item["value"]: item["description"] for item in longitudinal
                },
            }

        payload = {
            "model": self.model,
            "state": state.canonical_representation(self.representation_profile),
            "questions": questions,
        }
        started = time.perf_counter()
        response = post_json(self.endpoint, self.api_key, payload, self.timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000

        if len(maneuver_criteria) == 1:
            selected_maneuver = next(iter(maneuver_criteria))
            maneuver_probabilities = {selected_maneuver: 1.0}
        else:
            maneuver_answer = self._answer(response, "maneuver")
            selected_maneuver = str(maneuver_answer["choice"])
            if selected_maneuver not in maneuver_criteria:
                raise ValueError(
                    f"Provider selected unavailable maneuver: {selected_maneuver}"
                )
            maneuver_probabilities = {
                key: float(value)
                for key, value in maneuver_answer.get("probabilities", {}).items()
            }

        longitudinal_answer = None
        longitudinal_probabilities: Dict[str, float] = {}
        if len(longitudinal) == 1:
            only_value = str(longitudinal[0]["value"])
            longitudinal_probabilities = {only_value: 1.0}
        elif longitudinal:
            longitudinal_answer = self._answer(
                response, "longitudinal_action_given_keep_lane"
            )
            longitudinal_probabilities = {
                key: float(value)
                for key, value in longitudinal_answer.get("probabilities", {}).items()
            }

        if selected_maneuver == CHANGE_LEFT:
            selected_action = "LANE_LEFT"
        elif selected_maneuver == CHANGE_RIGHT:
            selected_action = "LANE_RIGHT"
        elif selected_maneuver == KEEP_LANE and len(longitudinal) == 1:
            selected_action = str(longitudinal[0]["value"])
        elif selected_maneuver == KEEP_LANE and longitudinal_answer is not None:
            selected_action = str(longitudinal_answer["choice"])
        else:
            raise ValueError(f"Cannot map selected maneuver {selected_maneuver!r}")

        probabilities: Dict[str, float] = {}
        left_probability = maneuver_probabilities.get(CHANGE_LEFT, 0.0)
        right_probability = maneuver_probabilities.get(CHANGE_RIGHT, 0.0)
        keep_probability = maneuver_probabilities.get(KEEP_LANE, 0.0)
        allowed = {action.value for action in specification.actions}
        if "LANE_LEFT" in allowed:
            probabilities["LANE_LEFT"] = left_probability
        if "LANE_RIGHT" in allowed:
            probabilities["LANE_RIGHT"] = right_probability
        for item in longitudinal:
            value = str(item["value"])
            probabilities[value] = keep_probability * longitudinal_probabilities.get(
                value, 0.0
            )

        # Some providers omit probabilities. Preserve a valid deterministic
        # distribution so the execution contract remains explicit.
        total = sum(probabilities.values())
        if total <= 0.0:
            probabilities = {
                action.value: (1.0 if action.value == selected_action else 0.0)
                for action in specification.actions
            }
        else:
            probabilities = {
                key: value / total for key, value in probabilities.items()
            }

        result = DecisionResult(
            action=selected_action,
            probabilities=probabilities,
            confidence=probabilities.get(selected_action),
            provider=self.name,
            model=response.get("model", self.model),
            latency_ms=latency_ms,
            usage={**(response.get("usage", {}) or {}), "api_calls": 1},
            raw_response={
                "decision_mode": "hierarchical-v1",
                "selected_maneuver": selected_maneuver,
                "response": response,
            },
        )
        result.validate(specification)
        return result
