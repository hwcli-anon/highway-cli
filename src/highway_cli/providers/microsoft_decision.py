from __future__ import annotations

from .openrouter_decisions import OpenRouterDecisionsProvider
from ..representations import RISK_AWARE_V1


class MicrosoftDecisionProvider(OpenRouterDecisionsProvider):
    name = "decision-1"

    def __init__(
        self,
        model: str = "microsoft/microsoft-decision-1",
        api_key: str = "",
        endpoint: str = "https://openrouter.ai/api/alpha/decisions",
        timeout_s: float = 30.0,
        representation_profile: str = RISK_AWARE_V1,
    ):
        super().__init__(
            model=model,
            provider_name=self.name,
            api_key=api_key,
            endpoint=endpoint,
            timeout_s=timeout_s,
            representation_profile=representation_profile,
        )
