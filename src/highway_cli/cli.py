from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .actions import nade_action_space
from .demo import adversarial_specification, demo_state
from .providers.mock import MockProvider
from .providers.replay import ReplayProvider
from .representations import (
    FULL_STATE_V1,
    REPRESENTATION_PROFILES,
    profile_summary,
)
from .tracing import append_trace


REQUIRED_NADE_PATHS = (
    "NADE_main.py",
    "NADE_core.py",
    "global_val.py",
    "config/configs.yml",
    "Data/AV2_AGENT/100000_model_CAV_agent.pth",
    "Data/CALCULATE_CHALLENGE",
    "Data/NDD_DATA",
)


def command_doctor(args: argparse.Namespace) -> int:
    # Path.resolve() invokes GetFinalPathNameByHandle on Windows and may fail
    # under a restricted workspace even when the directory is readable.
    repo = Path(os.path.abspath(os.fspath(args.nade_repo)))
    report = {
        "nade_repo": str(repo),
        "repo_exists": repo.is_dir(),
        "required_paths": {
            relative: (repo / relative).exists() for relative in REQUIRED_NADE_PATHS
        },
        "credentials": {
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "legacy_typesafe": bool(os.environ.get("TYPESAFE_API_KEY")),
            "legacy_openai": bool(os.environ.get("OPENAI_API_KEY")),
        },
        "python": sys.version.split()[0],
    }
    report["baseline_ready"] = report["repo_exists"] and all(report["required_paths"].values())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["baseline_ready"] else 1


def command_show_actions(_args: argparse.Namespace) -> int:
    print(json.dumps([action.to_dict() for action in nade_action_space()], indent=2))
    return 0


def command_show_representations(_args: argparse.Namespace) -> int:
    state = demo_state()
    print(
        json.dumps(
            [profile_summary(state, profile) for profile in REPRESENTATION_PROFILES],
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    state = demo_state()
    specification = adversarial_specification(state)
    if args.replay:
        provider = ReplayProvider(Path(args.replay))
    else:
        provider = MockProvider()
    result = provider.decide(state, specification)
    if args.output:
        append_trace(
            Path(args.output),
            state,
            specification,
            result,
            representation_profile=args.representation,
        )
    print(
        json.dumps(
            {
                "state": state.to_representation(args.representation),
                "result": {
                    "action": result.action,
                    "probabilities": result.probabilities,
                    "confidence": result.confidence,
                    "provider": result.provider,
                    "model": result.model,
                    "latency_ms": result.latency_ms,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    from .nade_runner import run_public_nade

    report = run_public_nade(
        Path(args.nade_repo),
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        cav_policy=args.cav_policy,
        background_policy=args.background_policy,
        state_output=Path(args.state_output) if args.state_output else None,
        trajectory_output=(
            Path(args.trajectory_output) if args.trajectory_output else None
        ),
        background_action_replay=(
            Path(args.bv_action_replay) if args.bv_action_replay else None
        ),
        representation_profile=args.representation,
    )
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_prepare_baseline_run(args: argparse.Namespace) -> int:
    from .baseline_runner import build_baseline_manifest, write_manifest

    manifest = build_baseline_manifest(
        nade_repo=Path(args.nade_repo),
        results_dir=Path(args.results_dir),
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        max_steps=args.max_steps,
        protocol_status=args.protocol_status,
        method=args.method,
        cav_policy=args.cav_policy,
    )
    write_manifest(Path(args.output), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def command_baseline_status(args: argparse.Namespace) -> int:
    from .baseline_runner import load_manifest, summarize

    print(
        json.dumps(
            summarize(load_manifest(Path(args.manifest))),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_run_baseline(args: argparse.Namespace) -> int:
    from .baseline_runner import load_manifest, planned_commands, run_manifest, summarize

    manifest = load_manifest(Path(args.manifest))
    if not args.execute:
        print(
            json.dumps(
                {
                    "run_status": "dry_run",
                    "summary": summarize(manifest),
                    "workers": args.workers,
                    "retries": args.retries,
                    "task_timeout_s": args.task_timeout,
                    "max_tasks": args.max_tasks,
                    "first_commands": planned_commands(manifest),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    report = run_manifest(
        manifest,
        project_dir=Path.cwd(),
        workers=args.workers,
        retries=args.retries,
        task_timeout_s=args.task_timeout,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["run_status"] in {"completed", "paused"} else 2


def _build_decision_provider(
    provider_name: str,
    *,
    decision_mode: str,
    representation: str,
    model: str = None,
    timeout: float = None,
    config: str = None,
    base_url: str = None,
    api_key_env: str = None,
    api_key_required: bool = None,
    temperature: float = None,
    max_output_tokens: int = None,
    reasoning_effort: str = None,
    format_retries: int = None,
    response_format: str = None,
):
    from .representations import (
        HIERARCHICAL_RELATIONAL_V1,
        INTERACTION_GRAPH_INTENT_V1,
        POST_MANEUVER_CONFLICT_V1,
        POST_MANEUVER_CONFLICT_V2,
        STAGE_ROUTED_CONFLICT_V1,
        SYMMETRIC_MANEUVER_EVIDENCE_V1,
        SYMMETRIC_MANEUVER_EVIDENCE_V2,
    )

    if decision_mode == "hierarchical" and provider_name not in {
        "jev",
        "decision-1",
    }:
        raise ValueError(
            "The legacy hierarchical mode currently supports jev and decision-1; "
            "use intent-hierarchical for an OpenAI-compatible model"
        )
    if (
        decision_mode == "intent-hierarchical"
        and representation
        not in {
            INTERACTION_GRAPH_INTENT_V1,
            POST_MANEUVER_CONFLICT_V1,
            POST_MANEUVER_CONFLICT_V2,
            STAGE_ROUTED_CONFLICT_V1,
        }
    ):
        raise ValueError(
            "Intent-hierarchical decision mode requires an intent-interface "
            f"representation: {INTERACTION_GRAPH_INTENT_V1} or "
            f"{POST_MANEUVER_CONFLICT_V1}, {POST_MANEUVER_CONFLICT_V2}, or "
            f"{STAGE_ROUTED_CONFLICT_V1}"
        )
    if (
        decision_mode == "hierarchical"
        and representation
        not in {
            HIERARCHICAL_RELATIONAL_V1,
            INTERACTION_GRAPH_INTENT_V1,
            POST_MANEUVER_CONFLICT_V1,
            POST_MANEUVER_CONFLICT_V2,
            STAGE_ROUTED_CONFLICT_V1,
            SYMMETRIC_MANEUVER_EVIDENCE_V1,
            SYMMETRIC_MANEUVER_EVIDENCE_V2,
        }
    ):
        raise ValueError(
            "Hierarchical decision mode requires "
            "a maneuver-first representation: "
            f"{HIERARCHICAL_RELATIONAL_V1}, {SYMMETRIC_MANEUVER_EVIDENCE_V1}, "
            f"{SYMMETRIC_MANEUVER_EVIDENCE_V2}, {INTERACTION_GRAPH_INTENT_V1}, "
            f"{POST_MANEUVER_CONFLICT_V1}, {POST_MANEUVER_CONFLICT_V2}, or "
            f"{STAGE_ROUTED_CONFLICT_V1}"
        )

    if provider_name == "mock":
        return MockProvider()

    options = {
        "timeout_s": timeout if timeout is not None else 15.0,
        "representation_profile": representation,
    }
    if model:
        options["model"] = model

    if provider_name == "jev":
        if decision_mode == "intent-hierarchical":
            from .providers.intent_openrouter import (
                IntentHierarchicalOpenRouterDecisionsProvider,
            )

            options.setdefault("model", "typesafe/jev-1.13")
            return IntentHierarchicalOpenRouterDecisionsProvider(
                provider_name="jev-intent-hierarchical", **options
            )
        if decision_mode == "hierarchical":
            from .providers.hierarchical_openrouter import (
                HierarchicalOpenRouterDecisionsProvider,
            )

            options.setdefault("model", "typesafe/jev-1.13")
            return HierarchicalOpenRouterDecisionsProvider(
                provider_name="jev-hierarchical", **options
            )
        from .providers.jev import JevProvider

        return JevProvider(**options)

    if provider_name == "decision-1":
        if decision_mode == "intent-hierarchical":
            from .providers.intent_openrouter import (
                IntentHierarchicalOpenRouterDecisionsProvider,
            )

            options.setdefault("model", "microsoft/microsoft-decision-1")
            return IntentHierarchicalOpenRouterDecisionsProvider(
                provider_name="decision-1-intent-hierarchical", **options
            )
        if decision_mode == "hierarchical":
            from .providers.hierarchical_openrouter import (
                HierarchicalOpenRouterDecisionsProvider,
            )

            options.setdefault("model", "microsoft/microsoft-decision-1")
            return HierarchicalOpenRouterDecisionsProvider(
                provider_name="decision-1-hierarchical", **options
            )
        from .providers.microsoft_decision import MicrosoftDecisionProvider

        return MicrosoftDecisionProvider(**options)

    if provider_name == "openai-decisions":
        from .providers.openai_decisions import OpenAIDecisionsProvider

        return OpenAIDecisionsProvider(**options)

    if provider_name == "openai-compatible":
        from .model_config import load_model_config, validate_model_config
        from .providers.openai_compatible import OpenAICompatibleProvider

        configured = load_model_config(Path(config)) if config else {}
        merged = {
            **configured,
            **{
                key: value
                for key, value in {
                    "adapter": "openai-compatible",
                    "model": model,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "api_key_required": api_key_required,
                    "timeout_s": timeout,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "reasoning_effort": reasoning_effort,
                    "format_retries": format_retries,
                    "response_format": response_format,
                }.items()
                if value is not None
            },
        }
        configured = validate_model_config(merged)
        configured.pop("adapter", None)
        return OpenAICompatibleProvider(
            provider_name=f"openai-compatible:{configured['model']}",
            representation_profile=representation,
            decision_mode=decision_mode,
            **configured,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")


def command_experiment(args: argparse.Namespace) -> int:
    from .nade_runner import run_public_nade

    if args.provider == "replay":
        if not args.replay:
            raise ValueError("--replay is required for the replay provider")
        provider = ReplayProvider(Path(args.replay))
    else:
        provider = _build_decision_provider(
            args.provider,
            decision_mode=args.decision_mode,
            representation=args.representation,
            model=args.model,
            timeout=args.timeout,
            config=args.config,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            api_key_required=args.api_key_required,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            format_retries=args.format_retries,
            response_format=args.response_format,
        )
    report = run_public_nade(
        Path(args.nade_repo),
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        cav_policy=args.cav_policy,
        provider=provider,
        decision_trace=Path(args.decision_trace) if args.decision_trace else None,
        trajectory_output=(
            Path(args.trajectory_output) if args.trajectory_output else None
        ),
        ndd_mix=args.ndd_mix,
        representation_profile=args.representation,
    )
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_audit_trajectories(args: argparse.Namespace) -> int:
    from .trajectory_audit import summarize_trajectories, write_episode_csv

    report = summarize_trajectories([Path(path) for path in args.trajectory])
    if args.episodes_output:
        write_episode_csv(Path(args.episodes_output), report["episode_details"])
    summary = {key: value for key, value in report.items() if key != "episode_details"}
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_prepare_formal_run(args: argparse.Namespace) -> int:
    from .formal_runner import build_formal_manifest, write_formal_manifest
    from .model_config import load_conditions

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    conditions = load_conditions(Path(args.conditions)) if args.conditions else None
    manifest = build_formal_manifest(
        nade_repo=Path(args.nade_repo),
        results_dir=Path(args.results_dir),
        seeds=seeds,
        ndd_mix=args.ndd_mix,
        max_steps=args.max_steps,
        provider_timeout_s=args.timeout,
        protocol_status=args.protocol_status,
        conditions=conditions,
        cav_policy=args.cav_policy,
    )
    write_formal_manifest(Path(args.output), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def command_formal_status(args: argparse.Namespace) -> int:
    from .formal_runner import load_manifest, summarize_formal_run

    report = summarize_formal_run(load_manifest(Path(args.manifest)))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_run_formal(args: argparse.Namespace) -> int:
    from .formal_runner import (
        load_manifest,
        planned_commands,
        run_formal_manifest,
        summarize_formal_run,
    )

    manifest = load_manifest(Path(args.manifest))
    if not args.execute:
        preview = {
            "run_status": "dry_run_no_api_calls",
            "summary": summarize_formal_run(manifest),
            "workers": args.workers,
            "network_retries": args.network_retries,
            "task_timeout_s": args.task_timeout,
            "max_observed_cost_usd": args.max_observed_cost_usd,
            "max_tasks": args.max_tasks,
            "first_commands": planned_commands(manifest),
            "execute_note": "Add --execute only after the formal protocol is authorized.",
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    report = run_formal_manifest(
        manifest,
        project_dir=Path.cwd(),
        workers=args.workers,
        network_retries=args.network_retries,
        task_timeout_s=args.task_timeout,
        max_observed_cost_usd=args.max_observed_cost_usd,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["run_status"] in {"completed", "paused"} else 2


def command_benchmark(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .benchmark_split import collection_seed, example_key
    from .nade_runner import run_public_nade

    selection_keys = None
    max_states = args.max_states
    if args.selection:
        selection_rows = [
            row
            for row in read_jsonl(Path(args.selection))
            if collection_seed(row) == args.seed
        ]
        selection_keys = {example_key(row) for row in selection_rows}
        if len(selection_keys) != len(selection_rows):
            raise ValueError("Duplicate keys in benchmark selection")
        if not selection_keys:
            raise ValueError(f"Selection has no states for seed {args.seed}")
        max_states = len(selection_keys)

    report = run_public_nade(
        Path(args.nade_repo),
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        benchmark_output=Path(args.output),
        max_benchmark_states=max_states,
        benchmark_selection_keys=selection_keys,
        rollout_repetitions=args.rollout_repetitions,
        validation_horizon_steps=args.validation_horizon_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_select_development(args: argparse.Namespace) -> int:
    from .development_selection import select_development_states

    report = select_development_states(
        [Path(path) for path in args.census],
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_output),
        selection_seed=args.selection_seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_freeze_development(args: argparse.Namespace) -> int:
    from .development_selection import freeze_development_benchmark

    report = freeze_development_benchmark(
        [Path(path) for path in args.benchmarks],
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_output),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_census(args: argparse.Namespace) -> int:
    from .nade_runner import run_public_nade

    report = run_public_nade(
        Path(args.nade_repo),
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        decision_state_output=Path(args.output),
        candidate_state_output=(
            Path(args.candidate_output) if args.candidate_output else None
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_enrich_nade_native(args: argparse.Namespace) -> int:
    from .native_benchmark import enrich_benchmark_with_native_observations

    report = enrich_benchmark_with_native_observations(
        Path(args.benchmark),
        [Path(path) for path in args.census],
        output_path=Path(args.output),
    )
    if args.manifest_output:
        manifest = Path(args.manifest_output)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_score_benchmark(args: argparse.Namespace) -> int:
    from .benchmark_eval import score_files

    report = score_files(
        Path(args.benchmark), Path(args.predictions), target=args.target
    )
    if args.details_output:
        detail_path = Path(args.details_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_summarize_benchmark(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl, summarize_records

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = summarize_records(records)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_compare_horizons(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl, summarize_horizon_ablation

    report = summarize_horizon_ablation(
        [read_jsonl(Path(path)) for path in args.benchmarks]
    )
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {key: value for key, value in report.items() if key != "details"},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.details_output:
        detail_path = Path(args.details_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_summarize_representations(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl, summarize_representation_sizes

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = summarize_representation_sizes(records)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_audit_representations(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .representation_audit import audit_records

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = audit_records(records)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["direct_label_leakage_detected"] else 0


def command_summarize_strata(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl, summarize_strata

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = summarize_strata(
        records,
        low_ttc_threshold_s=args.low_ttc_threshold,
        near_tie_threshold_m=args.near_tie_threshold,
    )
    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in report["manifest"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "manifest"}
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_prepare_splits(args: argparse.Namespace) -> int:
    from .benchmark_split import prepare_seed_splits

    report = prepare_seed_splits(
        [Path(path) for path in args.benchmarks],
        development_seeds=args.development_seeds,
        confirmation_seeds=args.confirmation_seeds,
        development_output=Path(args.development_output),
        confirmation_output=Path(args.confirmation_output),
    )
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite split manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_select_pilot(args: argparse.Namespace) -> int:
    from .pilot_selection import select_pilot

    report = select_pilot(
        [Path(path) for path in args.benchmarks],
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_output),
        max_states=args.max_states,
        min_per_stratum=args.min_per_stratum,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_assess_label_stability(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .label_stability import assess_label_stability

    reference = []
    repeated = []
    for path in args.reference:
        reference.extend(read_jsonl(Path(path)))
    for path in args.repeated:
        repeated.extend(read_jsonl(Path(path)))
    report = assess_label_stability(reference, repeated)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(report["summary"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(
                    json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n"
                )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


def command_score_soft_labels(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .soft_label_eval import score_soft_records

    benchmarks = []
    for path in args.benchmark:
        benchmarks.extend(read_jsonl(Path(path)))
    selection = read_jsonl(Path(args.selection)) if args.selection else None
    report = score_soft_records(
        benchmarks,
        read_jsonl(Path(args.predictions)),
        selection=selection,
        distance_tolerance_m=args.distance_tolerance,
        collision_rate_tolerance=args.collision_rate_tolerance,
        severe_normalized_rank_threshold=args.severe_normalized_rank_threshold,
    )
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(
                    json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n"
                )
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_compare_input_quality(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .input_quality_eval import compare_input_quality_records

    benchmarks = []
    for path in args.benchmark:
        benchmarks.extend(read_jsonl(Path(path)))
    selection = read_jsonl(Path(args.selection)) if args.selection else None
    report = compare_input_quality_records(
        benchmarks,
        read_jsonl(Path(args.baseline_predictions)),
        read_jsonl(Path(args.candidate_predictions)),
        selection=selection,
        distance_tolerance_m=args.distance_tolerance,
        collision_rate_tolerance=args.collision_rate_tolerance,
        severe_normalized_rank_threshold=args.severe_normalized_rank_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        comparison_status=args.comparison_status,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_audit_interaction_graph(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .interaction_graph import audit_interaction_graph_intent

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = audit_interaction_graph_intent(records)
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(
                    json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n"
                )
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_predict_benchmark(args: argparse.Namespace) -> int:
    from .benchmark_predict import predict_benchmark
    provider = _build_decision_provider(
        args.provider,
        decision_mode=args.decision_mode,
        representation=args.representation,
        model=args.model,
        timeout=args.timeout,
        config=args.config,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_key_required=args.api_key_required,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        format_retries=args.format_retries,
        response_format=args.response_format,
    )
    report = predict_benchmark(
        Path(args.benchmark),
        Path(args.output),
        provider,
        representation_profile=args.representation,
        max_examples=args.max_examples,
        request_delay_s=args.request_delay,
        reuse_successes_path=(
            Path(args.reuse_successes_from) if args.reuse_successes_from else None
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_select_maneuver_ab(args: argparse.Namespace) -> int:
    from .maneuver_ab import select_maneuver_ab

    report = select_maneuver_ab(
        Path(args.benchmark),
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_output),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_score_maneuvers(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .maneuver_eval import compare_maneuver_predictions

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = compare_maneuver_predictions(
        records,
        read_jsonl(Path(args.baseline_predictions)),
        read_jsonl(Path(args.candidate_predictions)),
        comparison_status=args.comparison_status,
    )
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_analyze_hierarchical_actions(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .hierarchical_actions import analyze_hierarchical_records

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = analyze_hierarchical_records(records)
    if args.details_output:
        detail_path = Path(args.details_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_audit_coverage(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .coverage_audit import audit_coverage, coverage_progression

    records = []
    record_groups = []
    for path in args.benchmark:
        group_records = read_jsonl(Path(path))
        records.extend(group_records)
        record_groups.append((str(path), group_records))
    report = audit_coverage(
        records,
        minimum_clusters_per_relation_cell=args.min_relation_clusters,
        minimum_clusters_per_maneuver_support=args.min_maneuver_clusters,
    )
    if args.details_output:
        detail_path = Path(args.details_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as handle:
            for detail in report["details"]:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.progression_output:
        progression_path = Path(args.progression_output)
        progression_path.parent.mkdir(parents=True, exist_ok=True)
        progression_path.write_text(
            json.dumps(
                coverage_progression(record_groups), indent=2, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_audit_candidate_selection(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .coverage_audit import audit_candidate_selection

    records = []
    for path in args.candidate_census:
        records.extend(read_jsonl(Path(path)))
    report = audit_candidate_selection(records)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_audit_maneuver_evidence(args: argparse.Namespace) -> int:
    from .benchmark_eval import read_jsonl
    from .maneuver_evidence import audit_maneuver_evidence

    records = []
    for path in args.benchmark:
        records.extend(read_jsonl(Path(path)))
    report = audit_maneuver_evidence(records, profile=args.representation)
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for row in report["details"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "details"}
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_model_test(args: argparse.Namespace) -> int:
    """Exercise a configured provider on a synthetic, non-formal scene."""

    state = demo_state()
    specification = adversarial_specification(state)
    provider = _build_decision_provider(
        args.provider,
        decision_mode=args.decision_mode,
        representation=args.representation,
        model=args.model,
        timeout=args.timeout,
        config=args.config,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_key_required=args.api_key_required,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        format_retries=args.format_retries,
        response_format=args.response_format,
    )
    try:
        result = provider.decide(state, specification)
    except Exception as error:
        from .providers.http import ProviderHTTPError
        from .providers.openai_compatible import ModelOutputError

        if not isinstance(error, (ProviderHTTPError, ModelOutputError, RuntimeError)):
            raise
        report = {
            "data_role": "synthetic provider connectivity test; exclude from experiments",
            "status": "failed",
            "failure_type": type(error).__name__,
            "error": str(error),
        }
        if isinstance(error, ProviderHTTPError):
            report["http_status"] = error.status_code
        if isinstance(error, ModelOutputError):
            report["latency_ms"] = error.latency_ms
            report["usage"] = error.usage
            report["reliability"] = error.audit
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    report = {
        "data_role": "synthetic provider connectivity test; exclude from experiments",
        "status": "passed",
        "provider": result.provider,
        "model": result.model,
        "action": result.action,
        "latency_ms": result.latency_ms,
        "usage": result.usage,
        "reliability": result.raw_response.get("reliability"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_summarize_model_reliability(args: argparse.Namespace) -> int:
    from .reliability import summarize_model_reliability

    report = summarize_model_reliability(Path(path) for path in args.trace)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _add_live_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Secret-free YAML/JSON model configuration")
    parser.add_argument("--model", help="Provider model identifier; overrides config")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key-env", help="Environment variable that contains the API key"
    )
    parser.add_argument(
        "--no-api-key",
        dest="api_key_required",
        action="store_false",
        default=None,
        help="Allow a local OpenAI-compatible endpoint without Authorization",
    )
    parser.add_argument("--timeout", type=float, help="Per-request timeout in seconds")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--format-retries",
        type=int,
        help="Retries after invalid model output; transport retries are separate",
    )
    parser.add_argument(
        "--response-format", choices=("json_object", "none")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="highway")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the public NADE baseline and credentials")
    doctor.add_argument("--nade-repo", default="../nade-public")
    doctor.set_defaults(func=command_doctor)

    actions = subparsers.add_parser("show-actions", help="Print the original NADE BV action space")
    actions.set_defaults(func=command_show_actions)

    representations = subparsers.add_parser(
        "show-representations", help="List state representations and demo payload sizes"
    )
    representations.set_defaults(func=command_show_representations)

    demo = subparsers.add_parser("demo", help="Run one offline provider-neutral decision")
    demo.add_argument("--output", help="Append the decision to a JSONL trace")
    demo.add_argument("--replay", help="Replay a result from an existing JSONL trace")
    demo.add_argument(
        "--representation", choices=REPRESENTATION_PROFILES, default=FULL_STATE_V1
    )
    demo.set_defaults(func=command_demo)

    baseline = subparsers.add_parser(
        "baseline", help="Run the public NADE baseline without editing its config"
    )
    baseline.add_argument("--nade-repo", default="../nade-public")
    baseline.add_argument("--episodes", type=int, default=1)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--max-steps", type=int, default=5000)
    baseline.add_argument(
        "--cav-policy", choices=("av2", "idm-mobil"), default="av2"
    )
    baseline.add_argument(
        "--background-policy", choices=("nade", "nde"), default="nade"
    )
    baseline.add_argument(
        "--state-output", help="Write provider-neutral candidate states as JSONL"
    )
    baseline.add_argument(
        "--trajectory-output", help="Write per-step pre/post vehicle trajectories as JSONL"
    )
    baseline.add_argument(
        "--bv-action-replay",
        help="Replay the selected background-vehicle actions from a trajectory JSONL",
    )
    baseline.add_argument(
        "--representation", choices=REPRESENTATION_PROFILES, default=FULL_STATE_V1
    )
    baseline.add_argument("--report-output", help="Write aggregate run metrics as JSON")
    baseline.set_defaults(func=command_baseline)

    prepare_baseline = subparsers.add_parser(
        "prepare-baseline-run",
        help="Create an immutable same-seed public-NADE baseline manifest",
    )
    prepare_baseline.add_argument("--nade-repo", default="../nade-public")
    prepare_baseline.add_argument("--results-dir", required=True)
    prepare_baseline.add_argument("--seed-start", type=int, required=True)
    prepare_baseline.add_argument("--seed-count", type=int, default=1000)
    prepare_baseline.add_argument("--max-steps", type=int, default=5000)
    prepare_baseline.add_argument(
        "--method", choices=("nade", "nde"), default="nade"
    )
    prepare_baseline.add_argument(
        "--cav-policy", choices=("av2", "idm-mobil"), default="av2"
    )
    prepare_baseline.add_argument(
        "--protocol-status",
        default="draft_not_authorized_to_run",
        choices=("draft_not_authorized_to_run", "frozen_authorized"),
    )
    prepare_baseline.add_argument("--output", required=True)
    prepare_baseline.set_defaults(func=command_prepare_baseline_run)

    baseline_status = subparsers.add_parser(
        "baseline-status", help="Summarize a resumable public-NADE baseline run"
    )
    baseline_status.add_argument("--manifest", required=True)
    baseline_status.set_defaults(func=command_baseline_status)

    run_baseline = subparsers.add_parser(
        "run-baseline", help="Run or preview a resumable public-NADE baseline manifest"
    )
    run_baseline.add_argument("--manifest", required=True)
    run_baseline.add_argument("--workers", type=int, default=2)
    run_baseline.add_argument("--retries", type=int, default=1)
    run_baseline.add_argument("--task-timeout", type=float, default=300.0)
    run_baseline.add_argument(
        "--max-tasks", type=int, default=0, help="0 means all pending tasks"
    )
    run_baseline.add_argument("--execute", action="store_true")
    run_baseline.set_defaults(func=command_run_baseline)

    experiment = subparsers.add_parser(
        "experiment", help="Run NADE with a decision provider controlling selected BVs"
    )
    experiment.add_argument("--nade-repo", default="../nade-public")
    experiment.add_argument(
        "--provider",
        choices=(
            "mock",
            "replay",
            "jev",
            "decision-1",
            "openai-decisions",
            "openai-compatible",
        ),
        default="mock",
    )
    experiment.add_argument("--replay", help="Decision trace used by the replay provider")
    _add_live_model_arguments(experiment)
    experiment.add_argument(
        "--decision-mode",
        choices=("flat", "hierarchical", "intent-hierarchical"),
        default="flat",
        help="Use the original flat action question or maneuver-first factorization",
    )
    experiment.add_argument("--episodes", type=int, default=1)
    experiment.add_argument("--seed", type=int, default=0)
    experiment.add_argument("--max-steps", type=int, default=5000)
    experiment.add_argument(
        "--cav-policy", choices=("av2", "idm-mobil"), default="av2"
    )
    experiment.add_argument("--ndd-mix", type=float, default=0.1)
    experiment.add_argument("--decision-trace", help="Write provider decisions as JSONL")
    experiment.add_argument(
        "--trajectory-output", help="Write per-step pre/post vehicle trajectories as JSONL"
    )
    experiment.add_argument("--report-output", help="Write aggregate run metrics as JSON")
    experiment.add_argument(
        "--representation", choices=REPRESENTATION_PROFILES, default=FULL_STATE_V1
    )
    experiment.set_defaults(func=command_experiment)

    model_test = subparsers.add_parser(
        "model-test",
        help="Test one configured model on a synthetic scene without running NADE",
    )
    model_test.add_argument(
        "--provider", choices=("openai-compatible",), default="openai-compatible"
    )
    _add_live_model_arguments(model_test)
    model_test.add_argument(
        "--decision-mode",
        choices=("flat", "intent-hierarchical"),
        default="intent-hierarchical",
    )
    model_test.add_argument(
        "--representation",
        choices=REPRESENTATION_PROFILES,
        default="interaction-graph-intent-v1",
    )
    model_test.set_defaults(func=command_model_test)

    reliability = subparsers.add_parser(
        "summarize-model-reliability",
        help="Report first-pass validity, retries, and fallbacks from decision traces",
    )
    reliability.add_argument("--trace", nargs="+", required=True)
    reliability.add_argument("--output")
    reliability.set_defaults(func=command_summarize_model_reliability)

    audit_trajectories = subparsers.add_parser(
        "audit-trajectories",
        help="Summarize collision geometry and action behavior from trajectory JSONL files",
    )
    audit_trajectories.add_argument(
        "--trajectory", action="append", required=True, help="Trajectory JSONL input"
    )
    audit_trajectories.add_argument("--summary-output", help="Write aggregate JSON")
    audit_trajectories.add_argument("--episodes-output", help="Write episode-level CSV")
    audit_trajectories.set_defaults(func=command_audit_trajectories)

    prepare_formal = subparsers.add_parser(
        "prepare-formal-run",
        help="Create an immutable paired task manifest for formal closed-loop runs",
    )
    prepare_formal.add_argument("--nade-repo", default="../nade-public")
    prepare_formal.add_argument("--results-dir", required=True)
    prepare_formal.add_argument("--seed-start", type=int, required=True)
    prepare_formal.add_argument("--seed-count", type=int, default=500)
    prepare_formal.add_argument("--max-steps", type=int, default=5000)
    prepare_formal.add_argument(
        "--cav-policy", choices=("av2", "idm-mobil"), default="av2"
    )
    prepare_formal.add_argument("--ndd-mix", type=float, default=0.1)
    prepare_formal.add_argument("--timeout", type=float, default=30.0)
    prepare_formal.add_argument(
        "--conditions",
        help="Optional YAML/JSON condition list for model-specific runs",
    )
    prepare_formal.add_argument(
        "--protocol-status",
        default="draft_not_authorized_to_run",
        choices=("draft_not_authorized_to_run", "frozen_authorized"),
    )
    prepare_formal.add_argument("--output", required=True)
    prepare_formal.set_defaults(func=command_prepare_formal_run)

    formal_status = subparsers.add_parser(
        "formal-status", help="Summarize resumable formal-run task outputs"
    )
    formal_status.add_argument("--manifest", required=True)
    formal_status.set_defaults(func=command_formal_status)

    run_formal = subparsers.add_parser(
        "run-formal", help="Run or preview a resumable formal closed-loop manifest"
    )
    run_formal.add_argument("--manifest", required=True)
    run_formal.add_argument("--workers", type=int, default=2)
    run_formal.add_argument("--network-retries", type=int, default=2)
    run_formal.add_argument("--task-timeout", type=float, default=900.0)
    run_formal.add_argument("--max-observed-cost-usd", type=float, default=2.0)
    run_formal.add_argument(
        "--max-tasks", type=int, default=0, help="0 means all pending tasks"
    )
    run_formal.add_argument(
        "--execute",
        action="store_true",
        help="Actually call providers; without this flag the command is a dry run",
    )
    run_formal.set_defaults(func=command_run_formal)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Export fixed decision states with one-step NADE branch-rollout labels",
    )
    benchmark.add_argument("--nade-repo", default="../nade-public")
    benchmark.add_argument("--episodes", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--max-steps", type=int, default=5000)
    benchmark.add_argument("--max-states", type=int, default=10)
    benchmark.add_argument(
        "--rollout-repetitions",
        type=int,
        default=1,
        help="Independent common-random-number rollout groups per state",
    )
    benchmark.add_argument(
        "--validation-horizon-steps",
        type=int,
        default=1,
        help=(
            "Compare one-step labels with this many closed-loop policy intervals; "
            "values greater than one export a v3 benchmark"
        ),
    )
    benchmark.add_argument("--output", required=True, help="Benchmark JSONL output")
    benchmark.add_argument(
        "--selection",
        help="Optional frozen census JSONL selecting exact states to label",
    )
    benchmark.set_defaults(func=command_benchmark)

    census = subparsers.add_parser(
        "census",
        help="Export all NADE-selected decision states without branch-rollout labels",
    )
    census.add_argument("--nade-repo", default="../nade-public")
    census.add_argument("--episodes", type=int, default=10)
    census.add_argument("--seed", type=int, default=0)
    census.add_argument("--max-steps", type=int, default=5000)
    census.add_argument("--output", required=True, help="Decision-state census JSONL")
    census.add_argument(
        "--candidate-output",
        help="Optional census of every nearby candidate before NADE selects one",
    )
    census.set_defaults(func=command_census)

    native_enrichment = subparsers.add_parser(
        "enrich-nade-native",
        help="Attach exact replayed NADE six-slot observations to a fixed benchmark",
    )
    native_enrichment.add_argument("--benchmark", required=True)
    native_enrichment.add_argument("--census", nargs="+", required=True)
    native_enrichment.add_argument("--output", required=True)
    native_enrichment.add_argument("--manifest-output")
    native_enrichment.set_defaults(func=command_enrich_nade_native)

    development = subparsers.add_parser(
        "select-development",
        help="Freeze one label-blind representative per interaction cluster",
    )
    development.add_argument("--census", nargs="+", required=True)
    development.add_argument("--selection-seed", type=int, default=20261013)
    development.add_argument("--output", required=True)
    development.add_argument("--manifest-output", required=True)
    development.set_defaults(func=command_select_development)

    freeze_development = subparsers.add_parser(
        "freeze-development-benchmark",
        help="Merge labeled shards into one immutable development benchmark",
    )
    freeze_development.add_argument("--benchmarks", nargs="+", required=True)
    freeze_development.add_argument("--output", required=True)
    freeze_development.add_argument("--manifest-output", required=True)
    freeze_development.set_defaults(func=command_freeze_development)

    score = subparsers.add_parser(
        "score-benchmark", help="Score model action predictions against rollout labels"
    )
    score.add_argument("--benchmark", required=True, help="Benchmark JSONL file")
    score.add_argument("--predictions", required=True, help="Prediction or trace JSONL file")
    score.add_argument(
        "--target",
        choices=("one-step", "validation"),
        default="one-step",
        help="Rollout label horizon used for scoring",
    )
    score.add_argument("--details-output", help="Optional per-example score JSONL")
    score.add_argument("--summary-output", help="Optional aggregate score JSON")
    score.set_defaults(func=command_score_benchmark)

    summarize = subparsers.add_parser(
        "summarize-benchmark", help="Summarize rollout-label coverage and diversity"
    )
    summarize.add_argument(
        "--benchmark", nargs="+", required=True, help="Benchmark JSONL file(s)"
    )
    summarize.add_argument("--output", help="Optional aggregate summary JSON")
    summarize.set_defaults(func=command_summarize_benchmark)

    horizons = subparsers.add_parser(
        "compare-horizons",
        help="Compare matched action rankings across rollout horizons",
    )
    horizons.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="One-step and multi-step benchmark JSONL files",
    )
    horizons.add_argument("--details-output", help="Optional per-state comparison JSONL")
    horizons.add_argument("--summary-output", help="Optional aggregate comparison JSON")
    horizons.set_defaults(func=command_compare_horizons)

    representation_sizes = subparsers.add_parser(
        "summarize-representations",
        help="Measure serialized input sizes for all representation profiles",
    )
    representation_sizes.add_argument(
        "--benchmark", nargs="+", required=True, help="Benchmark JSONL file(s)"
    )
    representation_sizes.add_argument("--output", help="Optional summary JSON")
    representation_sizes.set_defaults(func=command_summarize_representations)

    representation_audit = subparsers.add_parser(
        "audit-representations",
        help="Check model inputs for rollout-label leakage and non-portable fields",
    )
    representation_audit.add_argument(
        "--benchmark", nargs="+", required=True, help="Benchmark JSONL file(s)"
    )
    representation_audit.add_argument("--output", help="Optional audit JSON")
    representation_audit.set_defaults(func=command_audit_representations)

    strata = subparsers.add_parser(
        "summarize-strata",
        help="Summarize predeclared safety-relevant benchmark strata",
    )
    strata.add_argument(
        "--benchmark", nargs="+", required=True, help="Benchmark JSONL file(s)"
    )
    strata.add_argument("--low-ttc-threshold", type=float, default=5.0)
    strata.add_argument("--near-tie-threshold", type=float, default=0.25)
    strata.add_argument("--output", help="Optional aggregate summary JSON")
    strata.add_argument("--manifest-output", help="Optional per-example JSONL manifest")
    strata.set_defaults(func=command_summarize_strata)

    splits = subparsers.add_parser(
        "prepare-splits",
        help="Freeze seed-disjoint development and confirmation benchmarks",
    )
    splits.add_argument("--benchmarks", nargs="+", required=True)
    splits.add_argument("--development-seeds", nargs="+", type=int, required=True)
    splits.add_argument("--confirmation-seeds", nargs="+", type=int, required=True)
    splits.add_argument("--development-output", required=True)
    splits.add_argument("--confirmation-output", required=True)
    splits.add_argument("--manifest-output", required=True)
    splits.set_defaults(func=command_prepare_splits)

    pilot = subparsers.add_parser(
        "select-pilot",
        help="Freeze a cluster-deduplicated, stratified development API pilot",
    )
    pilot.add_argument("--benchmarks", nargs="+", required=True)
    pilot.add_argument("--max-states", type=int, default=12)
    pilot.add_argument("--min-per-stratum", type=int, default=2)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--manifest-output", required=True)
    pilot.set_defaults(func=command_select_pilot)

    stability = subparsers.add_parser(
        "assess-label-stability",
        help="Compare single-rollout labels with repeated-rollout aggregates",
    )
    stability.add_argument(
        "--reference", nargs="+", required=True, help="Reference benchmark JSONL file(s)"
    )
    stability.add_argument(
        "--repeated", nargs="+", required=True, help="Repeated-rollout benchmark JSONL file(s)"
    )
    stability.add_argument("--summary-output", help="Optional aggregate summary JSON")
    stability.add_argument("--details-output", help="Optional per-state diagnostic JSONL")
    stability.set_defaults(func=command_assess_label_stability)

    soft_score = subparsers.add_parser(
        "score-soft-labels",
        help="Score decisions against repeated-rollout probabilistic labels",
    )
    soft_score.add_argument(
        "--benchmark", nargs="+", required=True, help="Repeated benchmark JSONL file(s)"
    )
    soft_score.add_argument("--predictions", required=True, help="Prediction JSONL file")
    soft_score.add_argument(
        "--selection",
        help="Optional benchmark JSONL whose example keys select an evaluation subset",
    )
    soft_score.add_argument("--distance-tolerance", type=float, default=0.25)
    soft_score.add_argument("--collision-rate-tolerance", type=float, default=0.0)
    soft_score.add_argument(
        "--severe-normalized-rank-threshold",
        type=float,
        default=0.75,
        help=(
            "Transparent tail-error threshold on normalized repeated-rollout rank; "
            "stored in the score summary"
        ),
    )
    soft_score.add_argument("--summary-output", help="Optional aggregate summary JSON")
    soft_score.add_argument("--details-output", help="Optional per-state score JSONL")
    soft_score.set_defaults(func=command_score_soft_labels)

    input_quality = subparsers.add_parser(
        "compare-input-quality",
        help="Paired A/B comparison of two representations on repeated-rollout labels",
    )
    input_quality.add_argument(
        "--benchmark", nargs="+", required=True, help="Repeated benchmark JSONL file(s)"
    )
    input_quality.add_argument("--baseline-predictions", required=True)
    input_quality.add_argument("--candidate-predictions", required=True)
    input_quality.add_argument(
        "--selection",
        help="Optional benchmark JSONL whose example keys select an evaluation subset",
    )
    input_quality.add_argument("--distance-tolerance", type=float, default=0.25)
    input_quality.add_argument("--collision-rate-tolerance", type=float, default=0.0)
    input_quality.add_argument(
        "--severe-normalized-rank-threshold", type=float, default=0.75
    )
    input_quality.add_argument("--bootstrap-samples", type=int, default=10000)
    input_quality.add_argument("--bootstrap-seed", type=int, default=20261013)
    input_quality.add_argument(
        "--comparison-status",
        choices=("development_ab_not_confirmation", "unseen_confirmation"),
        default="development_ab_not_confirmation",
        help="Declare whether the paired comparison is developmental or unseen confirmation",
    )
    input_quality.add_argument("--output", help="Optional JSON comparison report")
    input_quality.set_defaults(func=command_compare_input_quality)

    interaction_graph = subparsers.add_parser(
        "audit-interaction-graph",
        help="Audit graph references, actor/action coverage and identifiers",
    )
    interaction_graph.add_argument("--benchmark", nargs="+", required=True)
    interaction_graph.add_argument("--summary-output")
    interaction_graph.add_argument("--details-output")
    interaction_graph.set_defaults(func=command_audit_interaction_graph)

    predict = subparsers.add_parser(
        "predict-benchmark", help="Run a decision provider on a fixed offline benchmark"
    )
    predict.add_argument("--benchmark", required=True, help="Benchmark JSONL file")
    predict.add_argument("--output", required=True, help="Prediction JSONL output")
    predict.add_argument(
        "--provider",
        choices=(
            "mock",
            "jev",
            "decision-1",
            "openai-decisions",
            "openai-compatible",
        ),
        default="mock",
    )
    _add_live_model_arguments(predict)
    predict.add_argument(
        "--decision-mode",
        choices=("flat", "hierarchical", "intent-hierarchical"),
        default="flat",
        help="Use the original flat action question or maneuver-first factorization",
    )
    predict.add_argument(
        "--representation", choices=REPRESENTATION_PROFILES, default=FULL_STATE_V1
    )
    predict.add_argument("--max-examples", type=int)
    predict.add_argument("--request-delay", type=float, default=0.0)
    predict.add_argument(
        "--reuse-successes-from",
        help="Reuse valid matching predictions and call the provider only for missing or failed records",
    )
    predict.set_defaults(func=command_predict_benchmark)

    hierarchy = subparsers.add_parser(
        "analyze-hierarchical-actions",
        help="Audit a lossless maneuver-first view of fixed NADE action labels",
    )
    hierarchy.add_argument(
        "--benchmark", nargs="+", required=True, help="Repeated benchmark JSONL file(s)"
    )
    hierarchy.add_argument("--summary-output", help="Optional aggregate summary JSON")
    hierarchy.add_argument("--details-output", help="Optional per-state JSONL")
    hierarchy.set_defaults(func=command_analyze_hierarchical_actions)

    coverage = subparsers.add_parser(
        "audit-coverage",
        help="Audit interaction and action coverage of fixed benchmark states",
    )
    coverage.add_argument(
        "--benchmark", nargs="+", required=True, help="Benchmark JSONL file(s)"
    )
    coverage.add_argument("--min-relation-clusters", type=int, default=5)
    coverage.add_argument("--min-maneuver-clusters", type=int, default=10)
    coverage.add_argument("--summary-output", help="Optional aggregate summary JSON")
    coverage.add_argument("--details-output", help="Optional per-state JSONL")
    coverage.add_argument(
        "--progression-output",
        help="Optional cumulative coverage summary in input-file order",
    )
    coverage.set_defaults(func=command_audit_coverage)

    candidate_audit = subparsers.add_parser(
        "audit-candidate-selection",
        help="Trace relation coverage through NADE candidate and criticality selection",
    )
    candidate_audit.add_argument(
        "--candidate-census",
        nargs="+",
        required=True,
        help="Candidate census JSONL file(s)",
    )
    candidate_audit.add_argument("--output", help="Optional summary JSON")
    candidate_audit.set_defaults(func=command_audit_candidate_selection)

    maneuver_evidence = subparsers.add_parser(
        "audit-maneuver-evidence",
        help="Audit symmetric maneuver evidence cards on a fixed benchmark",
    )
    maneuver_evidence.add_argument("--benchmark", nargs="+", required=True)
    maneuver_evidence.add_argument(
        "--representation",
        choices=("symmetric-maneuver-evidence-v1", "symmetric-maneuver-evidence-v2"),
        default="symmetric-maneuver-evidence-v1",
    )
    maneuver_evidence.add_argument("--summary-output")
    maneuver_evidence.add_argument("--details-output")
    maneuver_evidence.set_defaults(func=command_audit_maneuver_evidence)

    maneuver_ab = subparsers.add_parser(
        "select-maneuver-ab",
        help="Freeze all development states with at least two feasible maneuvers",
    )
    maneuver_ab.add_argument("--benchmark", required=True)
    maneuver_ab.add_argument("--output", required=True)
    maneuver_ab.add_argument("--manifest-output", required=True)
    maneuver_ab.set_defaults(func=command_select_maneuver_ab)

    maneuver_score = subparsers.add_parser(
        "score-maneuvers",
        help="Compare two predictions at the maneuver stage on repeated labels",
    )
    maneuver_score.add_argument("--benchmark", nargs="+", required=True)
    maneuver_score.add_argument("--baseline-predictions", required=True)
    maneuver_score.add_argument("--candidate-predictions", required=True)
    maneuver_score.add_argument(
        "--comparison-status",
        choices=("development_ab_not_confirmation", "unseen_confirmation"),
        default="development_ab_not_confirmation",
        help="Declare whether the maneuver comparison is developmental or unseen confirmation",
    )
    maneuver_score.add_argument("--summary-output")
    maneuver_score.add_argument("--details-output")
    maneuver_score.set_defaults(func=command_score_maneuvers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

