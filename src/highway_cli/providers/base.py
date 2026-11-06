from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import DecisionResult, DecisionSpecification, ScenarioState


class DecisionProvider(ABC):
    name = "base"

    @abstractmethod
    def decide(
        self, state: ScenarioState, specification: DecisionSpecification
    ) -> DecisionResult:
        raise NotImplementedError

