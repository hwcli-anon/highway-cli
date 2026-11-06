from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .actions import ActionOption


def mixed_proposal(
    model_probabilities: Dict[str, float],
    ndd_probabilities: Sequence[float],
    actions: Sequence[ActionOption],
    *,
    ndd_mix: float = 0.1,
) -> List[float]:
    """Mix a provider proposal with NDD to preserve importance-sampling support."""

    if len(ndd_probabilities) != len(actions):
        raise ValueError("NDD probability vector and action space have different sizes")
    if not 0.0 < ndd_mix <= 1.0:
        raise ValueError("ndd_mix must be in (0, 1]")
    ndd = [max(0.0, float(value)) for value in ndd_probabilities]
    ndd_total = sum(ndd)
    if ndd_total <= 0:
        raise ValueError("NDD distribution has no probability mass")
    ndd = [value / ndd_total for value in ndd]

    model = [
        max(0.0, float(model_probabilities.get(action.value, 0.0)))
        if ndd[index] > 0
        else 0.0
        for index, action in enumerate(actions)
    ]
    model_total = sum(model)
    if model_total <= 0:
        model = list(ndd)
    else:
        model = [value / model_total for value in model]
    proposal = [
        (1.0 - ndd_mix) * model_value + ndd_mix * ndd_value
        for model_value, ndd_value in zip(model, ndd)
    ]
    total = sum(proposal)
    return [value / total for value in proposal]


def importance_ratio(
    action_index: int,
    ndd_probabilities: Sequence[float],
    proposal_probabilities: Sequence[float],
) -> float:
    proposal = float(proposal_probabilities[action_index])
    if proposal <= 0:
        raise ValueError("Executed action has zero proposal probability")
    return float(ndd_probabilities[action_index]) / proposal
