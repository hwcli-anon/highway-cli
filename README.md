<div align="center">

# Highway-CLI

**A provider-neutral testbed for decision models and language models in closed-loop, safety-critical scenario generation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![Tests](https://img.shields.io/badge/tests-88%20passing-2F855A?style=flat-square)](#quick-start) [![Action support](https://img.shields.io/badge/action%20support-33%20actions-6B46C1?style=flat-square)](#what-the-interface-promises) [![License](https://img.shields.io/badge/license-MIT-2F855A?style=flat-square)](LICENSE)

[Quick start](#quick-start) · [Framework](#framework) · [NADE integration](#running-against-nade) · [Models](#connecting-a-decision-model) · [Docs](#documentation) · [License](#license-and-third-party)

</div>

<p align="center">
  <a href="assets/framework.pdf"><b>Figure 1 — Highway-CLI closed-loop framework (PDF)</b></a>
  <br>
  <sub><b>Figure 1.</b> (a) Closed loop: the interaction-graph interface builds <i>G</i><sub>t</sub> deterministically, the configured model chooses an intent and then an action, and the proposal is mixed with the naturalistic prior before execution. (b) Where the graph's node and edge features come from. (c) The intent-structured action interface, which keeps the original 33-action support intact.</sub>
</p>

Highway-CLI turns a frozen driving environment into a controllable experiment. The simulator, the critical background-vehicle selector, the executable action support, and the collision detector stay fixed; the model that decides **what the selected background vehicle should do** sits behind one structured interface and can be replaced without touching anything else.

It ships with adapters for two decision models, an OpenAI-compatible adapter for any chat-completions endpoint, and offline `mock`/`replay` providers, so the whole pipeline can be verified without a key or a simulator.

*This repository is an anonymous release candidate prepared for double-blind review.*

## Why this exists

Comparing two decision models, or two LLM backbones, on driving safety is easy to get wrong. A prompt tweak quietly changes the scene encoder. A retry policy quietly changes how many collisions are counted. A "small" fix to the action list quietly changes the support the model is scored against. The comparison then measures the harness, not the model.

Highway-CLI keeps that from happening silently:

- every model input is a **versioned representation profile**, not an ad-hoc prompt;
- every response is **validated against a schema** before it can reach the simulator;
- every run is a **resumable manifest** with fixed seeds and the provider settings recorded next to the results;
- every decision is written to a **trace** that can be replayed offline.

## Framework

Figure 1 is the whole method, in five moves.

1. **One state, two facts, built in parallel.** The interface is a pair, I_t = (x_t, C_t): a canonical scene and its relational context. Canonicalization, the relations β, and serialization are deterministic and read the same snapshot, so the edge features in panel (b) cannot drift out of sync with its node features.
2. **Two queries, one model.** The configured model answers π_θ(z | G_t) for the interaction intent and then π_θ(a | z_t, A_t) for the action. Panel (c) is a zoom of that one block — six intents, a selected subset, then a compatible action — not a second model.
3. **Deterministic code owns feasibility.** Intent membership (a ∈ A_z ∩ A_t), action indices, and execution are checked in code. The model proposes; it never invents an action the simulator cannot execute.
4. **Support is preserved.** The executed distribution is q = λ·π_θ + (1 − λ)·p_NDD with λ = 0.9, so supp(q) = supp(p_NDD) and naturalistic background traffic keeps non-zero mass.
5. **The loop closes on the environment.** The executed action advances the same simulator and the next state re-enters the encoder. Nothing in the loop is optimized against the collision detector.

## What the interface promises

| Guarantee | Where it lives |
| --- | --- |
| Provider-neutral state and action schema | `representations.py`, `schema.py` |
| Agent-centric interaction graph with relational risk cues | `interaction_graph.py` |
| Strict response validation, format retries, supported fallback | `nade_runner.py` |
| Support-preserving mixing of model and naturalistic proposals | `proposal.py` |
| Interchangeable automated-vehicle policies | `cav_policies.py` |
| Resumable, seed-paired manifests and auditable traces | `formal_runner.py`, `baseline_runner.py` |

The active action catalog is the original NADE one: `0–1` lane changes, `2–21` deceleration steps, `22` maintain, `23–32` acceleration steps — 33 executable actions in total. Run `highway show-actions` to print it from the source of truth.

## What is fixed, what you can change

| Fixed by the protocol | Changed deliberately, as a declared condition |
| --- | --- |
| NADE traffic dynamics and collision detection | Model family, provider, and model identifier |
| Critical background-vehicle selection | State representation profile |
| The 33-action support and the support-preserving mixture | Automated-vehicle policy (`av2` or `idm-mobil`) |
| Seeds, episode budgets, and manifest contents | Prompting and decoding settings in the model config |

Anything in the right column creates a new condition and must be recorded in the manifest. Changing a default in place is a protocol change, not a bug fix; see [`docs/reproduction.md`](docs/reproduction.md).

## Quick start

No API key, no simulator, no network access. Python 3.8 or newer.

```bash
git clone https://github.com/hwcli-anon/highway-cli.git
cd highway-cli

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e .

highway demo --representation interaction-graph-intent-v1
python -m unittest discover -s tests -v
```

Expected result: `Ran 88 tests ... OK`, and a demo that ends with a deterministic decision:

```json
"result": {
  "action": "-4.0",
  "confidence": 0.46666666666666673,
  "provider": "mock",
  "model": "deterministic-fixture-v1"
}
```

If the console script is not on `PATH` yet, use `python -m highway_cli` in place of `highway`.

<details>
<summary>What the demo actually exercises</summary>

The demo builds one canonical scene, renders it through the requested representation, asks the local deterministic provider for an intent and an action, normalizes the proposal, and prints the whole payload. It is a contract test for the interface rather than a simulation, which makes it the fastest way to tell whether an installation is sound.
</details>

## Running against NADE

NADE is an external dependency and is **not** redistributed here. Clone the public repository next to this one:

```bash
git clone https://github.com/michigan-traffic-lab/Naturalistic-and-Adversarial-Driving-Environment.git ../nade-public
```

The compatibility environment used during development is pinned in `requirements-nade.txt`. PyTorch must be installed separately, with a wheel matching your platform.

```bash
highway doctor --nade-repo ../nade-public
highway baseline --nade-repo ../nade-public --episodes 1 --seed 0

highway experiment --nade-repo ../nade-public \
  --provider mock \
  --decision-mode intent-hierarchical \
  --representation interaction-graph-intent-v1 \
  --episodes 1 --seed 0 --ndd-mix 0.1 \
  --decision-trace results/mock/decisions.jsonl \
  --report-output results/mock/report.json
```

`doctor` reports the checkout, the importable dependencies, and whether credentials are present. `baseline` runs the unmodified NADE configuration; `experiment` is the only path in which a provider decides for the selected vehicle.

NADE keeps its own PolyForm Noncommercial license. Read [`THIRD_PARTY.md`](THIRD_PARTY.md) before using or redistributing any NADE component.

## Connecting a decision model

A model configuration contains only metadata and the **name** of an environment variable. Keys never appear in YAML, on a command line, in a trace, or in a commit.

```yaml
model:
  adapter: openai-compatible
  model: deepseek/deepseek-v4.1-flash-1017
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  timeout_s: 30
  temperature: 0
  max_output_tokens: 128
  reasoning_effort: none
  format_retries: 1
  response_format: json_object
```

```bash
export OPENROUTER_API_KEY="your-key"
highway model-test --config configs/models/deepseek-openrouter.yaml
```

<details>
<summary>Windows PowerShell</summary>

```powershell
$env:OPENROUTER_API_KEY = "your-key"
highway model-test --config configs\models\deepseek-openrouter.yaml
```
</details>

For a local OpenAI-compatible server, copy a config and set `api_key_required: false`. Provider behaviour, the structured output contract, and retry semantics are described in [`docs/providers.md`](docs/providers.md).

## Command map

| Command | Purpose |
| --- | --- |
| `highway doctor` | Check the NADE checkout, dependencies, and credentials |
| `highway show-actions` | Print the 33-action catalog from the source of truth |
| `highway show-representations` | List representation profiles and payload sizes |
| `highway demo` | One offline, deterministic decision |
| `highway baseline` | Run the public NADE baseline without editing its config |
| `highway prepare-baseline-run`, `run-baseline`, `baseline-status` | Resumable, seed-paired baseline manifests |
| `highway experiment` | Closed loop with a decision provider on the selected vehicle |
| `highway model-test` | One configured model on a synthetic scene, no simulator |
| `highway prepare-formal`, `run-formal`, `formal-status` | Paired formal experiments with per-task outputs |

`highway --help` lists the full surface, including the audit, ablation, and calibration subcommands used during development.

## Repository layout

```
highway-cli/
  src/highway_cli/
    cli.py                    # command surface
    representations.py        # versioned state-to-input profiles
    interaction_graph.py      # agent-centric graph and risk relations
    nade_runner.py            # providers, validation, closed loop
    proposal.py               # normalization and support-preserving mixing
    cav_policies.py           # av2 and idm-mobil policies
    formal_runner.py          # paired, resumable model experiments
    baseline_runner.py        # paired NDE and NADE baselines
  configs/
    models/                   # secret-free provider configurations
    conditions/               # frozen protocol conditions
  docs/                       # architecture, representations, providers, reproduction
  examples/mock_demo/         # smallest end-to-end example
  tests/                      # 88 standard-library unit tests
  tools/check_public_tree.py  # release hygiene check
  assets/framework.pdf        # Figure 1 (PDF)
```

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | The seven steps of the loop and the stable extension points |
| [`docs/representations.md`](docs/representations.md) | `interaction-graph-intent-v1` and the controlled comparison profile |
| [`docs/providers.md`](docs/providers.md) | Provider contract, validation, retries, and fallback |
| [`docs/reproduction.md`](docs/reproduction.md) | Offline checks, smoke test, and the formal-experiment boundary |

## Citation

<!-- TODO: replace with the camera-ready BibTeX once the manuscript is de-anonymized. -->

```bibtex
@inproceedings{highwaycli2026,
  title     = {Decision-Model-Driven Safety-Critical Interaction Generation},
  author    = {Anonymous},
  booktitle = {IEEE International Conference on Intelligent Vehicles (IV)},
  year      = {2026},
  note      = {Under review}
}
```

## License and third-party

Project-owned code is released under the [MIT License](LICENSE). The grant covers only the code authored for this project: it does not relicense NADE, its data, its pretrained models, or any other third-party component, which retain their own terms. See [`THIRD_PARTY.md`](THIRD_PARTY.md) before redistributing anything that depends on NADE.

## Repository status

- 88 standard-library unit tests pass (verified on Python 3.10; the CI matrix covers 3.10 and 3.11).
- The offline interaction-graph demo runs without network access, a model key, or NADE.
- The scientific protocol is frozen. Work on the public repository should focus on documentation, packaging, clean-environment reproduction, and privacy review; [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) records the release gate.
- The MIT copyright holder is written as "The Highway-CLI Authors" to keep the repository anonymous during review; individual authors and affiliations will be named in the public release.

<div align="center">
<sub>Built for reproducible comparison, not for leaderboard screenshots.</sub>
</div>
