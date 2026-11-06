from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MANIFEST_VERSION = "highway-formal-run-v1"

FORMAL_CONDITIONS: Tuple[Dict[str, str], ...] = (
    {
        "condition_id": "jev-reference",
        "provider": "jev",
        "decision_mode": "hierarchical",
        "representation": "hierarchical-relational-v1",
    },
    {
        "condition_id": "jev-candidate",
        "provider": "jev",
        "decision_mode": "intent-hierarchical",
        "representation": "interaction-graph-intent-v1",
    },
    {
        "condition_id": "decision1-reference",
        "provider": "decision-1",
        "decision_mode": "hierarchical",
        "representation": "hierarchical-relational-v1",
    },
    {
        "condition_id": "decision1-candidate",
        "provider": "decision-1",
        "decision_mode": "intent-hierarchical",
        "representation": "interaction-graph-intent-v1",
    },
)

TRANSPORT_ERROR_MARKERS = (
    "providerhttperror",
    "connectionerror",
    "connection error",
    "connection reset",
    "connection refused",
    "remote disconnected",
    "remotedisconnected",
    "timed out",
    "timeout",
    "temporary failure",
    "name resolution",
    "socket",
    "winerror 10013",
    "errno 2",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    temporary.write_text(content, encoding="utf-8")
    try:
        os.replace(str(temporary), str(path))
    except PermissionError:
        # Some Windows app-container temp directories deny rename/replace even
        # though both files are writable. Preserve functionality with a visible
        # best-effort fallback; experiment attempts remain independently logged.
        path.write_text(content, encoding="utf-8")
        temporary.unlink(missing_ok=True)


def build_formal_manifest(
    *,
    nade_repo: Path,
    results_dir: Path,
    seeds: Sequence[int],
    ndd_mix: float = 0.1,
    max_steps: int = 5000,
    provider_timeout_s: float = 30.0,
    protocol_status: str = "draft_not_authorized_to_run",
    conditions: Optional[Sequence[Dict[str, Any]]] = None,
    cav_policy: str = "av2",
) -> Dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    if not (0.0 < ndd_mix <= 1.0):
        raise ValueError("ndd_mix must be in (0, 1]")
    if cav_policy not in {"av2", "idm-mobil"}:
        raise ValueError("cav_policy must be 'av2' or 'idm-mobil'")

    nade_repo = Path(os.path.abspath(os.fspath(nade_repo)))
    results_dir = Path(os.path.abspath(os.fspath(results_dir)))
    selected_conditions = list(conditions) if conditions is not None else list(FORMAL_CONDITIONS)
    if not selected_conditions:
        raise ValueError("At least one condition is required")
    condition_ids = [condition.get("condition_id") for condition in selected_conditions]
    if None in condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Condition identifiers must be present and unique")
    for condition in selected_conditions:
        for key in ("condition_id", "provider", "decision_mode", "representation"):
            if not condition.get(key):
                raise ValueError(f"Condition is missing {key}")
        if condition["provider"] == "openai-compatible":
            from .model_config import validate_model_config

            validated = validate_model_config(condition)
            condition.update(validated)

    tasks: List[Dict[str, Any]] = []
    # Seed-major order keeps the four paired conditions close in wall-clock time.
    for seed in seeds:
        for condition in selected_conditions:
            tasks.append(
                {
                    "task_id": f"seed-{seed:06d}__{condition['condition_id']}",
                    "seed": seed,
                    "episodes": 1,
                    **condition,
                }
            )

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "protocol_status": protocol_status,
        "data_role": "formal closed-loop experiment only after explicit authorization",
        "nade_repo": str(nade_repo),
        "results_dir": str(results_dir),
        "execution": {
            "cav_policy": cav_policy,
            "ndd_mix": ndd_mix,
            "max_steps": max_steps,
            "provider_timeout_s": provider_timeout_s,
            "episodes_per_task": 1,
            "pairing": "same seed across every listed condition",
            "ordering": "seed-major interleaving",
        },
        "conditions": selected_conditions,
        "seeds": list(seeds),
        "tasks": tasks,
        "task_count": len(tasks),
        "claim_boundary": [
            "Preparing or inspecting this manifest does not authorize API execution.",
            "Calibration and failed-attempt outputs must not be merged into formal results.",
            "A transport-contaminated attempt is retained but excluded and rerun with the same seed.",
            "A parse, invalid-action, or other model-semantic fallback is retained as a model outcome.",
        ],
    }


def write_formal_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    validate_manifest(manifest)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    _write_json_atomic(path, manifest)


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported formal-run manifest version")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must contain tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if None in task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Manifest task_id values must be present and unique")
    if manifest.get("task_count") != len(tasks):
        raise ValueError("Manifest task_count does not match tasks")
    if manifest.get("execution", {}).get("cav_policy", "av2") not in {
        "av2",
        "idm-mobil",
    }:
        raise ValueError("Unsupported CAV policy")
    for task in tasks:
        for key in (
            "seed",
            "episodes",
            "condition_id",
            "provider",
            "decision_mode",
            "representation",
        ):
            if key not in task:
                raise ValueError(f"Task {task.get('task_id')} is missing {key}")
        if task["episodes"] != 1:
            raise ValueError("Formal runner requires one episode per task")


def _read_trace_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_transport_error(message: Optional[str]) -> bool:
    if not message:
        return False
    normalized = message.lower()
    # Invalid request payloads are deterministic provider rejections, not
    # transient transport failures. Retrying them unchanged only hides a schema
    # defect and wastes API calls.
    if "http 400" in normalized:
        return False
    return any(marker in normalized for marker in TRANSPORT_ERROR_MARKERS)


def classify_trace(path: Path) -> Dict[str, Any]:
    rows = _read_trace_rows(path)
    transport = []
    semantic = []
    for index, row in enumerate(rows):
        result = row.get("result", {})
        if not result.get("fallback_used"):
            continue
        item = {"row": index + 1, "error": result.get("error")}
        if is_transport_error(result.get("error")):
            transport.append(item)
        else:
            semantic.append(item)
    return {
        "trace_rows": len(rows),
        "transport_fallbacks": transport,
        "semantic_fallbacks": semantic,
    }


def _task_dir(manifest: Dict[str, Any], task: Dict[str, Any]) -> Path:
    return Path(manifest["results_dir"]) / "tasks" / task["task_id"]


def _existing_attempt_count(task_dir: Path) -> int:
    if not task_dir.is_dir():
        return 0
    return len(
        [
            child
            for child in task_dir.iterdir()
            if child.is_dir() and child.name.startswith("attempt-")
        ]
    )


def _completed_result(task_dir: Path) -> Optional[Dict[str, Any]]:
    result_path = task_dir / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") in {"completed", "completed_with_model_fallback"}:
        return result
    return None


def _build_experiment_command(
    manifest: Dict[str, Any], task: Dict[str, Any], attempt_dir: Path
) -> List[str]:
    execution = manifest["execution"]
    command = [
        sys.executable,
        "-m",
        "highway_cli",
        "experiment",
        "--nade-repo",
        manifest["nade_repo"],
        "--provider",
        task["provider"],
        "--cav-policy",
        execution.get("cav_policy", "av2"),
    ]
    if task.get("model"):
        command.extend(["--model", str(task["model"])])
    option_flags = {
        "base_url": "--base-url",
        "api_key_env": "--api-key-env",
        "temperature": "--temperature",
        "max_output_tokens": "--max-output-tokens",
        "reasoning_effort": "--reasoning-effort",
        "format_retries": "--format-retries",
        "response_format": "--response-format",
    }
    for key, flag in option_flags.items():
        if task.get(key) is not None:
            command.extend([flag, str(task[key])])
    if task.get("api_key_required") is False:
        command.append("--no-api-key")
    command.extend(
        [
        "--decision-mode",
        task["decision_mode"],
        "--representation",
        task["representation"],
        "--episodes",
        "1",
        "--seed",
        str(task["seed"]),
        "--max-steps",
        str(execution["max_steps"]),
        "--ndd-mix",
        str(execution["ndd_mix"]),
        "--timeout",
        str(execution["provider_timeout_s"]),
        "--decision-trace",
        str(attempt_dir / "decisions.jsonl"),
        "--report-output",
        str(attempt_dir / "report.json"),
        ]
    )
    return command


def _run_one_task(
    manifest: Dict[str, Any],
    task: Dict[str, Any],
    *,
    project_dir: Path,
    network_retries: int,
    task_timeout_s: float,
) -> Dict[str, Any]:
    task_dir = _task_dir(manifest, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_result(task_dir)
    if completed is not None:
        return completed

    prior_attempts = _existing_attempt_count(task_dir)
    attempt_records = []
    for offset in range(network_retries + 1):
        attempt_number = prior_attempts + offset + 1
        attempt_dir = task_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=False, exist_ok=False)
        command = _build_experiment_command(manifest, task, attempt_dir)
        started_at = _utc_now()
        try:
            process = subprocess.run(
                command,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=task_timeout_s,
                check=False,
            )
            stdout = process.stdout
            stderr = process.stderr
            returncode = process.returncode
            timed_out = False
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            returncode = None
            timed_out = True

        (attempt_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (attempt_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        report_path = attempt_dir / "report.json"
        trace_path = attempt_dir / "decisions.jsonl"
        report = None
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_classification = classify_trace(trace_path)
        process_text = f"{stdout}\n{stderr}"
        retryable_transport = bool(trace_classification["transport_fallbacks"])
        if report is None and (timed_out or is_transport_error(process_text)):
            retryable_transport = True

        attempt_record = {
            "attempt": attempt_number,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "returncode": returncode,
            "timed_out": timed_out,
            "report": str(report_path),
            "decision_trace": str(trace_path),
            **trace_classification,
            "disposition": "excluded_transport_failure"
            if retryable_transport
            else "accepted",
        }
        _write_json_atomic(attempt_dir / "attempt.json", attempt_record)
        attempt_records.append(attempt_record)

        if retryable_transport:
            continue
        if report is None or returncode != 0:
            result = {
                "task_id": task["task_id"],
                "status": "failed_non_transport",
                "seed": task["seed"],
                "condition_id": task["condition_id"],
                "attempts_this_invocation": attempt_records,
                "finished_at": _utc_now(),
            }
            _write_json_atomic(task_dir / "result.json", result)
            return result

        expected_requests = int(report.get("provider_requests", 0))
        if trace_classification["trace_rows"] != expected_requests:
            result = {
                "task_id": task["task_id"],
                "status": "failed_integrity_check",
                "reason": "trace row count does not match provider_requests",
                "seed": task["seed"],
                "condition_id": task["condition_id"],
                "attempts_this_invocation": attempt_records,
                "finished_at": _utc_now(),
            }
            _write_json_atomic(task_dir / "result.json", result)
            return result

        status = (
            "completed_with_model_fallback"
            if trace_classification["semantic_fallbacks"]
            else "completed"
        )
        result = {
            "task_id": task["task_id"],
            "status": status,
            "seed": task["seed"],
            "condition_id": task["condition_id"],
            "accepted_attempt": attempt_number,
            "accepted_report": str(report_path),
            "accepted_decision_trace": str(trace_path),
            "observed_cost_usd": float(
                report.get("provider_cost_usd_successful", 0.0) or 0.0
            ),
            "attempts_this_invocation": attempt_records,
            "finished_at": _utc_now(),
        }
        _write_json_atomic(task_dir / "result.json", result)
        return result

    result = {
        "task_id": task["task_id"],
        "status": "network_retry_exhausted",
        "seed": task["seed"],
        "condition_id": task["condition_id"],
        "attempts_this_invocation": attempt_records,
        "finished_at": _utc_now(),
    }
    _write_json_atomic(task_dir / "result.json", result)
    return result


def summarize_formal_run(manifest: Dict[str, Any]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    observed_cost = 0.0
    completed_ids = set()
    condition_counts: Dict[str, Dict[str, int]] = {}
    for task in manifest["tasks"]:
        task_dir = _task_dir(manifest, task)
        result_path = task_dir / "result.json"
        status = "pending"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = result.get("status", "unknown")
            observed_cost += float(result.get("observed_cost_usd", 0.0) or 0.0)
        status_counts[status] = status_counts.get(status, 0) + 1
        condition = condition_counts.setdefault(task["condition_id"], {})
        condition[status] = condition.get(status, 0) + 1
        if status in {"completed", "completed_with_model_fallback"}:
            completed_ids.add(task["task_id"])
    return {
        "manifest_version": manifest["manifest_version"],
        "protocol_status": manifest["protocol_status"],
        "cav_policy": manifest["execution"].get("cav_policy", "av2"),
        "task_count": len(manifest["tasks"]),
        "status_counts": status_counts,
        "condition_status_counts": condition_counts,
        "completed_task_count": len(completed_ids),
        "observed_successful_call_cost_usd": observed_cost,
        "cost_scope": "successful API calls in accepted attempts only",
    }


def run_formal_manifest(
    manifest: Dict[str, Any],
    *,
    project_dir: Path,
    workers: int = 2,
    network_retries: int = 2,
    task_timeout_s: float = 900.0,
    max_observed_cost_usd: float = 2.0,
    max_tasks: int = 0,
) -> Dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if network_retries < 0:
        raise ValueError("network_retries must be non-negative")
    if max_observed_cost_usd <= 0:
        raise ValueError("max_observed_cost_usd must be positive")

    initial = summarize_formal_run(manifest)
    if initial["observed_successful_call_cost_usd"] >= max_observed_cost_usd:
        return {**initial, "run_status": "observed_cost_guard_reached"}

    pending = []
    for task in manifest["tasks"]:
        if _completed_result(_task_dir(manifest, task)) is None:
            pending.append(task)
    if max_tasks > 0:
        pending = pending[:max_tasks]

    results = []
    # Schedule one bounded wave at a time so the observed-cost guard can stop
    # new work. In-flight tasks can still cause a small, explicitly bounded overshoot.
    cursor = 0
    while cursor < len(pending):
        current = summarize_formal_run(manifest)
        if current["observed_successful_call_cost_usd"] >= max_observed_cost_usd:
            break
        wave = pending[cursor : cursor + workers]
        cursor += len(wave)
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [
                executor.submit(
                    _run_one_task,
                    manifest,
                    task,
                    project_dir=project_dir,
                    network_retries=network_retries,
                    task_timeout_s=task_timeout_s,
                )
                for task in wave
            ]
            for future in as_completed(futures):
                results.append(future.result())

    final = summarize_formal_run(manifest)
    run_status = "completed" if final["completed_task_count"] == len(manifest["tasks"]) else "paused"
    if final["observed_successful_call_cost_usd"] >= max_observed_cost_usd:
        run_status = "observed_cost_guard_reached"
    return {
        **final,
        "run_status": run_status,
        "tasks_attempted_this_invocation": len(results),
        "workers": workers,
        "network_retries": network_retries,
        "task_timeout_s": task_timeout_s,
        "max_observed_cost_usd": max_observed_cost_usd,
        "guard_note": (
            "The cost guard uses provider-reported successful-call cost. "
            "In-flight work and failed calls without usage can exceed it."
        ),
    }


def planned_commands(manifest: Dict[str, Any], limit: int = 4) -> List[List[str]]:
    commands = []
    project_dir = Path.cwd()
    for task in manifest["tasks"][:limit]:
        preview_dir = _task_dir(manifest, task) / "attempt-01"
        commands.append(_build_experiment_command(manifest, task, preview_dir))
    return commands
