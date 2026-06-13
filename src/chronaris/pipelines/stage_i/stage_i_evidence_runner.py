"""Unified Stage I evidence runner for thesis-facing closure tasks."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_evidence"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_PYTHON_EXECUTABLE = "/home/wangminan/env/anaconda3/envs/chronaris/bin/python"
DEFAULT_E_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json"
)
DEFAULT_F_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json"
)
DEFAULT_EXISTING_OUTPUTS = {
    "rigid_body": {
        "summary_path": "docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/rigid_body_ablation_summary.json",
        "report_path": "docs/artifacts/stage_i/stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md",
        "evidence_layer": "rigid_body_support",
    },
    "semantic": {
        "summary_path": "docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json",
        "report_path": "docs/artifacts/stage_i/stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md",
        "evidence_layer": "semantic_support",
    },
    "runtime": {
        "summary_path": "docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/runtime_inference_summary.json",
        "report_path": "docs/artifacts/stage_i/stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md",
        "evidence_layer": "runtime_replay",
    },
}
SUPPORTED_TASKS = (
    "multitask",
    "rigid_body",
    "semantic",
    "runtime",
    "private_proxy",
    "public_adapter",
    "rotation",
)


@dataclass(frozen=True, slots=True)
class StageIEvidenceRunnerConfig:
    """Configuration for one unified Stage I evidence closure run."""

    run_id: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    python_executable: str = DEFAULT_PYTHON_EXECUTABLE
    only: tuple[str, ...] = ("all",)
    reuse_existing: bool = False
    skip_heavy: bool = False
    test_summary: str = "not_run_by_runner"
    e_run_manifest_path: str = DEFAULT_E_MANIFEST_PATH
    f_run_manifest_path: str = DEFAULT_F_MANIFEST_PATH
    existing_outputs: Mapping[str, Mapping[str, str]] | None = None


@dataclass(frozen=True, slots=True)
class StageIEvidenceRunnerRunResult:
    """Artifacts written by one Stage I evidence closure run."""

    run_id: str
    artifact_root: str
    manifest_path: str
    report_path: str
    manifest: Mapping[str, object]


def run_stage_i_evidence_closure(
    config: StageIEvidenceRunnerConfig,
) -> StageIEvidenceRunnerRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_evidence_runner",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "only": list(_resolve_requested_tasks(config.only)),
            "reuse_existing": config.reuse_existing,
        },
    ) as progress:
        return _run_stage_i_evidence_closure_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_evidence_closure_observed(
    *,
    config: StageIEvidenceRunnerConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIEvidenceRunnerRunResult:
    manifest_path = run_root / "evidence_manifest.json"
    report_path = Path(config.report_root) / f"stage-i-evidence-closure-{config.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": _utc_now(),
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "python_executable": config.python_executable,
        "git_commit": resolve_git_commit(),
        "test_summary": config.test_summary,
        "tasks": {},
    }
    requested_tasks = _resolve_requested_tasks(config.only)
    progress.update("tasks_resolved", task_count=len(requested_tasks))

    for task_name in requested_tasks:
        try:
            task_payload = _run_one_task(
                task_name=task_name,
                config=config,
                run_root=run_root,
            )
            manifest["tasks"][task_name] = task_payload
            progress.update(
                "task_finished",
                task_name=task_name,
                status=task_payload["status"],
            )
        except Exception as exc:
            manifest["tasks"][task_name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "evidence_layer": _default_layer(task_name),
            }
            _write_manifest(manifest_path, manifest)
            progress.update(
                "task_failed",
                task_name=task_name,
                error_type=type(exc).__name__,
            )
            continue
        _write_manifest(manifest_path, manifest)

    status = "completed"
    if any(task["status"] == "failed" for task in manifest["tasks"].values()):
        status = "partial_failed"
    manifest["status"] = status
    _write_manifest(manifest_path, manifest)
    report_path.write_text(
        render_stage_i_evidence_manifest_report(manifest) + "\n",
        encoding="utf-8",
    )
    progress.finish(manifest_path=str(manifest_path), report_path=str(report_path), status=status)
    return StageIEvidenceRunnerRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
        manifest=manifest,
    )


def render_stage_i_evidence_manifest_report(manifest: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Evidence Closure - {manifest['run_id']}",
        "",
        f"- status: `{manifest.get('status', 'unknown')}`",
        f"- git_commit: `{manifest.get('git_commit')}`",
        f"- test_summary: `{manifest.get('test_summary')}`",
        "",
        "| task | status | evidence_layer | reused_existing | outputs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task_name, payload in manifest.get("tasks", {}).items():
        outputs = payload.get("outputs", {})
        output_text = ", ".join(f"{key}={value}" for key, value in outputs.items()) or "-"
        lines.append(
            "| "
            f"`{task_name}` | "
            f"`{payload.get('status')}` | "
            f"`{payload.get('evidence_layer')}` | "
            f"`{payload.get('reused_existing', False)}` | "
            f"`{output_text}` |"
        )
    return "\n".join(lines)


def _run_one_task(
    *,
    task_name: str,
    config: StageIEvidenceRunnerConfig,
    run_root: Path,
) -> dict[str, object]:
    existing_outputs = {
        **DEFAULT_EXISTING_OUTPUTS,
        **dict(config.existing_outputs or {}),
    }
    if task_name in {"rigid_body", "semantic", "runtime"}:
        outputs = dict(existing_outputs[task_name])
        _assert_outputs_exist(outputs)
        return {
            "status": "completed",
            "evidence_layer": outputs["evidence_layer"],
            "reused_existing": True,
            "commands": [],
            "outputs": outputs,
        }
    if task_name == "multitask":
        max_runs = "2" if config.skip_heavy else "4"
        command = [
            config.python_executable,
            "scripts/run_stage_i_multitask_sweep.py",
            "--run-id",
            f"{config.run_id}-multitask",
            "--e-run-manifest",
            config.e_run_manifest_path,
            "--f-run-manifest",
            config.f_run_manifest_path,
            "--max-runs",
            max_runs,
            "--epoch-count",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ]
        if config.skip_heavy:
            command.extend(
                [
                    "--sample-source",
                    "stage_h_window_stats_proxy",
                    "--physiology-point-limit",
                    "500",
                    "--vehicle-point-limit",
                    "500",
                ]
            )
        return _run_script_task(
            command=command,
            evidence_layer="thesis_weak_label",
            cwd=Path.cwd(),
            run_root=run_root / "logs",
            task_name=task_name,
        )
    if task_name == "private_proxy":
        command = [
            config.python_executable,
            "scripts/run_stage_i_private_component_ablation.py",
            "--run-id",
            f"{config.run_id}-private-proxy",
            "--e-run-manifest",
            config.e_run_manifest_path,
            "--f-run-manifest",
            config.f_run_manifest_path,
        ]
        return _run_script_task(
            command=command,
            evidence_layer="private_proxy",
            cwd=Path.cwd(),
            run_root=run_root / "logs",
            task_name=task_name,
        )
    if task_name == "public_adapter":
        calibration_command = [
            config.python_executable,
            "scripts/run_stage_i_public_adapter_calibration.py",
            "--run-id",
            f"{config.run_id}-public-adapter",
        ]
        calibration_payload = _run_script_task(
            command=calibration_command,
            evidence_layer="public_adapter_calibration",
            cwd=Path.cwd(),
            run_root=run_root / "logs",
            task_name="public_adapter_calibration",
        )
        transfer_command = [
            config.python_executable,
            "scripts/build_stage_i_public_transfer_boundary.py",
            "--run-id",
            f"{config.run_id}-transfer-boundary",
            "--calibration-summary-path",
            calibration_payload["outputs"]["summary_path"],
        ]
        transfer_payload = _run_script_task(
            command=transfer_command,
            evidence_layer="transfer_boundary",
            cwd=Path.cwd(),
            run_root=run_root / "logs",
            task_name="public_transfer_boundary",
        )
        return {
            "status": "completed",
            "evidence_layer": "public_adapter_closure",
            "reused_existing": False,
            "commands": calibration_payload["commands"] + transfer_payload["commands"],
            "outputs": {
                "calibration_summary_path": calibration_payload["outputs"]["summary_path"],
                "calibration_report_path": calibration_payload["outputs"]["report_path"],
                "transfer_summary_path": transfer_payload["outputs"]["summary_path"],
                "transfer_report_path": transfer_payload["outputs"]["report_path"],
            },
        }
    if task_name == "rotation":
        command = [
            config.python_executable,
            "scripts/run_stage_i_rigid_body_rotation_audit.py",
            "--run-id",
            f"{config.run_id}-rotation",
        ]
        return _run_script_task(
            command=command,
            evidence_layer="rotation_diagnostics",
            cwd=Path.cwd(),
            run_root=run_root / "logs",
            task_name=task_name,
        )
    raise ValueError(f"unsupported evidence task: {task_name}")


def _run_script_task(
    *,
    command: Sequence[str],
    evidence_layer: str,
    cwd: Path,
    run_root: Path,
    task_name: str,
) -> dict[str, object]:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / f"{task_name}.log"
    payload = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        payload.stdout + ("\n" if payload.stdout and payload.stderr else "") + payload.stderr,
        encoding="utf-8",
    )
    if payload.returncode != 0:
        raise RuntimeError(
            f"{task_name} failed with exit code {payload.returncode}: {payload.stderr.strip()}"
        )
    parsed = _extract_json_payload(payload.stdout)
    return {
        "status": "completed",
        "evidence_layer": evidence_layer,
        "reused_existing": False,
        "commands": [list(command)],
        "outputs": parsed,
        "log_path": str(log_path),
    }


def _extract_json_payload(stdout: str) -> dict[str, object]:
    text = stdout.strip()
    if not text:
        raise ValueError("task stdout is empty; expected JSON payload.")
    return json.loads(text)


def _assert_outputs_exist(outputs: Mapping[str, str]) -> None:
    for key, value in outputs.items():
        if key.endswith("_path") and not Path(value).exists():
            raise FileNotFoundError(f"missing required evidence output: {value}")


def _default_layer(task_name: str) -> str:
    mapping = {
        "multitask": "thesis_weak_label",
        "rigid_body": "rigid_body_support",
        "semantic": "semantic_support",
        "runtime": "runtime_replay",
        "private_proxy": "private_proxy",
        "public_adapter": "public_adapter_closure",
        "rotation": "rotation_diagnostics",
    }
    return mapping.get(task_name, task_name)


def _resolve_requested_tasks(only: Sequence[str]) -> tuple[str, ...]:
    if not only or "all" in only:
        return SUPPORTED_TASKS
    unsupported = [task for task in only if task not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError("unsupported evidence tasks: " + ", ".join(sorted(unsupported)))
    return tuple(dict.fromkeys(only))


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_git_commit(*, cwd: str | Path = ".") -> str | None:
    try:
        payload = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return payload.stdout.strip() or None
