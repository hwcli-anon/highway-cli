# Anonymous GitHub repository handoff

## Assigned scope

Prepare and independently verify the anonymous GitHub repository using this
release-candidate directory as the complete source tree.

You may:

- create the anonymous repository and reproduce this directory structure;
- improve wording, navigation, screenshots, and command presentation;
- run installation, tests, the offline demo, and the public-tree checker;
- report commands that fail or documentation that is unclear;
- add a small GIF or screenshot after confirming that it contains no identity
  or local-path information;
- improve repository topics and the anonymous project description.

Do not change without review:

- scene representations or their defaults;
- critical-background-vehicle selection;
- the 33-action interface or support-preserving mixture;
- model parsing, fallback, retries, or experiment manifests;
- CAV policies, collision detection, seeds, or published result files;
- model versions used by the manuscript.

## Required verification

From a new environment, run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m highway_cli demo --representation interaction-graph-intent-v1
python tools/check_public_tree.py
```

Record the operating system, Python version, the exact failing command if any,
and only the minimum error text needed for diagnosis. Do not send credentials or
private provider responses.

## Acceptance criteria

- 88 tests pass.
- The offline demo runs without a key or NADE.
- A new reader can find installation, mock demo, NADE integration, provider
  configuration, architecture, and license status from the README.
- The repository contains no personal identity, institution, local absolute
  path, credential, private paper, raw API response, milestone directory, cache,
  or large trajectory.
- Core-code changes, if genuinely required, are proposed separately and are not
  merged into the anonymous repository without review.

## Current blockers

The authors must select a project-owned code license before the repository is
described as open source. The current official IEEE IV anonymity rules must also
be checked immediately before making the repository public.
