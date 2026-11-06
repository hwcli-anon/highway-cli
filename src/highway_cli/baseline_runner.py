from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


MANIFEST_VERSION = "highway-baseline-run-v1"


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
        path.write_text(content, encoding="utf-8")
        temporary.unlink(missing_ok=True)


def build_baseline_manifest(
    *,
    nade_repo: Path,
    results_dir: Path,
    seeds: Sequence[int],
    max_steps: int = 5000,
    protocol_status: str = "draft_not_authorized_to_run",
    method: str = "nade",
    cav_policy: str = "av2",
) -> Dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if method not in {"nade", "nde"}:
        raise ValueError("method must be 'nade' or 'nde'")
    if cav_policy not in {"av2", "idm-mobil"}:
        raise ValueError("cav_policy must be 'av2' or 'idm-mobil'")

    nade_repo = Path(os.path.abspath(os.fspath(nade_repo)))
    results_dir = Path(os.path.abspath(os.fspath(results_dir)))
    tasks = [
        {
            "task_id": f"seed-{seed:06d}__{method}",
            "condition_id": method,
            "seed": seed,
            "episodes": 1,
        }
        for seed in seeds
    ]
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "protocol_status": protocol_status,
        "data_role": f"formal same-seed public-{method.upper()} baseline",
        "method": method,
        "cav_policy": cav_policy,
        "nade_repo": str(nade_repo),
        "results_dir": str(results_dir),
        "execution": {
            "max_steps": max_steps,
            "episodes_per_task": 1,
            "seed_independence": "one fresh process and one episode per seed",
        },
        "conditions": [
            {"condition_id": method, "method": f"public {method.upper()}"}
        ],
        "seeds": list(seeds),
        "tasks": tasks,
        "task_count": len(tasks),
        "claim_boundary": [
            "The seed range matches the completed decision-model experiment.",
            "This baseline makes no external API calls.",
            (
                "NDE leaves every background vehicle on the public environment's "
                "naturalistic NDD policy without NADE intervention."
                if method == "nde"
                else "NADE selects the controlled background vehicle and action."
            ),
            "Each task is retained separately for resumability and auditability.",
        ],
    }


def validate_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported baseline-run manifest version")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must contain tasks")
    if manifest.get("task_count") != len(tasks):
        raise ValueError("Manifest task_count does not match tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if None in task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Manifest task_id values must be present and unique")
    method = manifest.get("method", "nade")
    if method not in {"nade", "nde"}:
        raise ValueError("Unsupported baseline method")
    if manifest.get("cav_policy", "av2") not in {"av2", "idm-mobil"}:
        raise ValueError("Unsupported CAV policy")
    for task in tasks:
        if task.get("episodes") != 1:
            raise ValueError("Baseline runner requires one episode per task")
        if task.get("condition_id") != method:
            raise ValueError("Task condition does not match baseline method")
        if not isinstance(task.get("seed"), int) or task["seed"] < 0:
            raise ValueError("Every task requires a non-negative integer seed")


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    validate_manifest(manifest)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    _write_json_atomic(path, manifest)


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def _task_dir(manifest: Dict[str, Any], task: Dict[str, Any]) -> Path:
    return Path(manifest["results_dir"]) / "tasks" / task["task_id"]


def _completed_result(task_dir: Path) -> Dict[str, Any] | None:
    path = task_dir / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result if result.get("status") == "completed" else None


def _attempt_count(task_dir: Path) -> int:
    if not task_dir.is_dir():
        return 0
    return sum(
        child.is_dir() and child.name.startswith("attempt-")
        for child in task_dir.iterdir()
    )


def _command(
    manifest: Dict[str, Any], task: Dict[str, Any], attempt_dir: Path
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "highway_cli",
        "baseline",
        "--nade-repo",
        manifest["nade_repo"],
        "--episodes",
        "1",
        "--seed",
        str(task["seed"]),
        "--max-steps",
        str(manifest["execution"]["max_steps"]),
        "--background-policy",
        manifest.get("method", "nade"),
        "--cav-policy",
        manifest.get("cav_policy", "av2"),
        "--report-output",
        str(attempt_dir / "report.json"),
    ]


def _run_one(
    manifest: Dict[str, Any],
    task: Dict[str, Any],
    *,
    project_dir: Path,
    task_timeout_s: float,
    retries: int,
) -> Dict[str, Any]:
    task_dir = _task_dir(manifest, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_result(task_dir)
    if completed is not None:
        return completed

    prior_attempts = _attempt_count(task_dir)
    attempts = []
    for offset in range(retries + 1):
        number = prior_attempts + offset + 1
        attempt_dir = task_dir / f"attempt-{number:02d}"
        attempt_dir.mkdir()
        command = _command(manifest, task, attempt_dir)
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
            stdout, stderr = process.stdout, process.stderr
            returncode, timed_out = process.returncode, False
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or "", error.stderr or ""
            returncode, timed_out = None, True
        (attempt_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (attempt_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        report_path = attempt_dir / "report.json"
        report = None
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = None
        accepted = returncode == 0 and not timed_out and report is not None
        attempt = {
            "attempt": number,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "returncode": returncode,
            "timed_out": timed_out,
            "report": str(report_path),
            "disposition": "accepted" if accepted else "failed",
        }
        _write_json_atomic(attempt_dir / "attempt.json", attempt)
        attempts.append(attempt)
        if accepted:
            result = {
                "task_id": task["task_id"],
                "status": "completed",
                "condition_id": manifest.get("method", "nade"),
                "seed": task["seed"],
                "accepted_attempt": number,
                "accepted_report": str(report_path),
                "attempts_this_invocation": attempts,
                "finished_at": _utc_now(),
            }
            _write_json_atomic(task_dir / "result.json", result)
            return result

    result = {
        "task_id": task["task_id"],
        "status": "failed",
        "condition_id": manifest.get("method", "nade"),
        "seed": task["seed"],
        "attempts_this_invocation": attempts,
        "finished_at": _utc_now(),
    }
    _write_json_atomic(task_dir / "result.json", result)
    return result


def summarize(manifest: Dict[str, Any]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    completed = 0
    crashes = 0
    weight_sum = 0.0
    weight_square_sum = 0.0
    weighted_crash_sum = 0.0
    elapsed_s = 0.0
    for task in manifest["tasks"]:
        result_path = _task_dir(manifest, task) / "result.json"
        status = "pending"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = result.get("status", "unknown")
            if status == "completed":
                report = json.loads(
                    Path(result["accepted_report"]).read_text(encoding="utf-8")
                )
                completed += 1
                crashes += int(report["results"]["AV-Crash"])
                weight_sum += float(report.get("importance_weight_sum", 0.0))
                weight_square_sum += float(
                    report.get("importance_weight_square_sum", 0.0)
                )
                weighted_crash_sum += float(report.get("weighted_crash_sum", 0.0))
                elapsed_s += float(report.get("elapsed_s", 0.0))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "manifest_version": manifest["manifest_version"],
        "protocol_status": manifest["protocol_status"],
        "cav_policy": manifest.get("cav_policy", "av2"),
        "task_count": len(manifest["tasks"]),
        "completed_task_count": completed,
        "status_counts": status_counts,
        "crash_count_completed": crashes,
        "crash_rate_unweighted_completed": crashes / completed if completed else None,
        "crash_rate_importance_sampling_completed": (
            weighted_crash_sum / completed if completed else None
        ),
        "effective_sample_size_completed": (
            weight_sum * weight_sum / weight_square_sum
            if weight_square_sum
            else None
        ),
        "simulator_elapsed_s_completed": elapsed_s,
    }


def run_manifest(
    manifest: Dict[str, Any],
    *,
    project_dir: Path,
    workers: int = 2,
    retries: int = 1,
    task_timeout_s: float = 300.0,
    max_tasks: int = 0,
) -> Dict[str, Any]:
    if workers < 1 or retries < 0 or task_timeout_s <= 0:
        raise ValueError("Invalid baseline-run execution settings")
    pending = [
        task
        for task in manifest["tasks"]
        if _completed_result(_task_dir(manifest, task)) is None
    ]
    if max_tasks > 0:
        pending = pending[:max_tasks]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one,
                manifest,
                task,
                project_dir=project_dir,
                task_timeout_s=task_timeout_s,
                retries=retries,
            )
            for task in pending
        ]
        for future in as_completed(futures):
            results.append(future.result())
    final = summarize(manifest)
    return {
        **final,
        "run_status": (
            "completed"
            if final["completed_task_count"] == len(manifest["tasks"])
            else "paused"
        ),
        "tasks_attempted_this_invocation": len(results),
        "workers": workers,
        "retries": retries,
        "task_timeout_s": task_timeout_s,
    }


def planned_commands(manifest: Dict[str, Any], limit: int = 4) -> List[List[str]]:
    return [
        _command(manifest, task, _task_dir(manifest, task) / "attempt-01")
        for task in manifest["tasks"][:limit]
    ]
