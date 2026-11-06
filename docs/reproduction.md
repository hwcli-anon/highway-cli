# Reproduction boundary

This candidate contains the framework and tests required to verify the software
contract. It intentionally excludes private raw traces and third-party NADE
assets.

## Offline verification

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
highway demo --representation interaction-graph-intent-v1
```

Expected result: 88 tests pass and the demo returns a deterministic mock decision.

## Closed-loop smoke test

After installing NADE separately:

```bash
highway doctor --nade-repo ../nade-public
highway experiment --nade-repo ../nade-public --provider mock \
  --decision-mode intent-hierarchical \
  --representation interaction-graph-intent-v1 \
  --episodes 1 --seed 0 --ndd-mix 0.1 \
  --decision-trace results/mock/decisions.jsonl \
  --report-output results/mock/report.json
```

## Formal experiments

Formal results use immutable manifests, paired seeds, fixed model identifiers,
one worker when resource constrained, and resumable per-task outputs. Preview a
manifest before adding `--execute`. The private raw experiment archive should be
released only after a separate privacy, licensing, and size review.

Changing a representation, CAV policy, selected seeds, action mixture, collision
detector, fallback, or provider identifier creates a new protocol condition and
must not overwrite the frozen one.
