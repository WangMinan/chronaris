#!/usr/bin/env python3
"""CLI orchestration for the Chronaris core-task recovery protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from chronaris.evaluation.application_tasks.core_recovery_protocol import (
    CoreRecoveryProtocol,
    sha256_file,
    write_protocol_locks,
)
from chronaris.evaluation.application_tasks.core_recovery_confirmation import (
    CoreRecoveryConfirmationConfig,
    run_core_recovery_confirmation,
)
from chronaris.evaluation.application_tasks.core_recovery_development import (
    CoreRecoveryDevelopmentConfig,
    run_core_recovery_development,
)
from chronaris.evaluation.application_tasks.core_recovery_lock import (
    CoreRecoveryLockConfig,
    lock_core_recovery_development,
)
from chronaris.evaluation.application_tasks.core_recovery_outer_audit import (
    CoreRecoveryOuterAuditConfig,
    audit_and_invalidate_confirmation,
)
from chronaris.evaluation.application_tasks.core_recovery_task_audit import (
    CoreTaskAuditConfig,
    run_core_task_audit,
)
from chronaris.evaluation.application_tasks.core_recovery_run import (
    CoreRecoverySmokeConfig,
    run_core_recovery_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery"
DEFAULT_DATA_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/runs/2026-07-11_dingxin-context-bindings/source_manifest.json"
)
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits/split_manifest.json"
)
DEFAULT_TASK_DEFINITION = REPO_ROOT / "docs/requirements/downstream-evaluation-spec.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "lock-protocol",
            "audit-tasks",
            "smoke-train",
            "develop",
            "lock-development",
            "confirm",
            "audit-confirmation",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--task-definition", type=Path, default=DEFAULT_TASK_DEFINITION)
    parser.add_argument(
        "--branch-name",
        default="codex/chronaris-core-task-recovery-20260713",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--methods", nargs="+", default=("chronaris",))
    parser.add_argument("--candidate-ids", nargs="+", default=None)
    parser.add_argument("--frozen-steps", type=int, default=60)
    parser.add_argument("--partial-steps", type=int, default=60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "lock-protocol":
        return lock_protocol(args)
    if args.command == "audit-tasks":
        result = run_core_task_audit(CoreTaskAuditConfig())
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "completed" else 2
    if args.command == "smoke-train":
        result = run_core_recovery_smoke(CoreRecoverySmokeConfig())
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "completed" else 2
    if args.command == "develop":
        candidate_ids = args.candidate_ids
        if candidate_ids is None:
            candidate_ids = tuple(
                f"{method_name}-c{index:02d}"
                for method_name in args.methods
                for index in range(1, 5)
            )
        result = run_core_recovery_development(
            CoreRecoveryDevelopmentConfig(
                method_names=tuple(args.methods),
                candidate_ids=tuple(candidate_ids),
                frozen_backbone_steps=args.frozen_steps,
                partial_unfreeze_steps=args.partial_steps,
                device=args.device,
            )
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "completed" else 2
    if args.command == "lock-development":
        result = lock_core_recovery_development(CoreRecoveryLockConfig())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "locked" else 2
    if args.command == "confirm":
        result = run_core_recovery_confirmation(CoreRecoveryConfirmationConfig())
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "completed" else 2
    if args.command == "audit-confirmation":
        result = audit_and_invalidate_confirmation(CoreRecoveryOuterAuditConfig())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 2
    raise AssertionError("unreachable command")


def lock_protocol(args: argparse.Namespace) -> int:
    source_commit = _git("rev-parse", "origin/main")
    current_branch = _git("branch", "--show-current")
    if current_branch != args.branch_name:
        raise RuntimeError(
            f"refusing to lock protocol on {current_branch}; expected {args.branch_name}"
        )
    source_files = {
        "data_manifest": args.data_manifest,
        "outer_split_manifest": args.split_manifest,
        "task_definition": args.task_definition,
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = CoreRecoveryProtocol(
        source_commit=source_commit,
        branch_name=current_branch,
        data_manifest_sha256=sha256_file(args.data_manifest),
        outer_split_manifest_sha256=sha256_file(args.split_manifest),
        task_definition_sha256=sha256_file(args.task_definition),
    )
    paths = write_protocol_locks(
        args.output_root,
        protocol=protocol,
        source_files=source_files,
    )
    manifest = {
        "run_id": args.output_root.name,
        "status": "protocol_locked",
        "source_commit": source_commit,
        "branch_name": current_branch,
        "outer_test_opened": False,
        "confirmed_metrics_changed": False,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    (args.output_root / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
