# Offline mock demo

From the repository root:

```bash
python -m highway_cli demo --representation interaction-graph-intent-v1
```

This example uses a deterministic local provider, does not require NADE, and
does not contact any network service. It verifies state construction,
representation rendering, candidate-action handling, and result serialization.
