# Architecture

Highway-CLI separates the simulator, scene representation, model provider, and
execution policy so that model comparisons do not silently change the driving
environment.

1. NADE advances naturalistic traffic and selects a critical background vehicle.
2. The encoder converts the current simulator state into a versioned model input.
3. A provider adapter requests an intent and a compatible executable action.
4. The parser validates the structured response and uses a supported fallback
   after an exhausted format retry.
5. Support-preserving mixing combines the model proposal with the original
   naturalistic distribution.
6. The simulator executes the sampled action and observes the next state.
7. The trace records the encoded input, proposal, executed action, probabilities,
   validity, latency, cost metadata, and outcome.

The configured model does not replace the simulator, vehicle dynamics, critical-
vehicle selector, action catalog, or collision detector. This boundary is part
of the experimental protocol.

## Stable extension points

- `representations.py`: deterministic state-to-input profiles.
- `interaction_graph.py`: agent-centric graph and risk relations.
- `model_config.py`: secret-free provider configuration.
- `nade_runner.py`: provider adapters, validation, and closed-loop integration.
- `proposal.py`: probability normalization and support-preserving mixing.
- `cav_policies.py`: interchangeable automated-vehicle policies.
- `formal_runner.py`: paired, resumable model experiments.
- `baseline_runner.py`: paired NDE and NADE experiments.

New extensions should preserve the canonical state schema and make experimental
changes explicit in the manifest rather than modifying defaults in place.
