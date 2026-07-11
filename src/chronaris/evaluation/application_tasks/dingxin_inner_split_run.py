"""Build and audit inner validation plans for fixed Dingxin outer folds."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.dingxin_inner_split_audit import (
    build_inner_split_acceptance_rows,
)
from chronaris.evaluation.application_tasks.dingxin_inner_split_reporting import (
    write_inner_split_outputs,
)
from chronaris.evaluation.application_tasks.dingxin_inner_splits import (
    build_dingxin_inner_split_plans,
    build_inner_task_coverage_rows,
    interval_overlap_count,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_inner_splits")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinInnerSplitConfig:
    run_id: str = "2026-07-11_dingxin-inner-splits"
    output_root: str = "docs/artifacts/runs"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    context_binding_root: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-context-bindings"
    )


@dataclass(frozen=True, slots=True)
class DingxinInnerSplitResult:
    run_id: str
    status: str
    run_root: str
    fold_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_inner_split(
    config: DingxinInnerSplitConfig,
) -> DingxinInnerSplitResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    fixed_root = Path(config.fixed_audit_root)
    binding_root = Path(config.context_binding_root)
    source_paths = {
        "outer_split_manifest": fixed_root / "split_manifest.json",
        "context_catalog": binding_root / "context_catalog.csv",
        "fold_task_binding": binding_root / "fold_task_binding.csv",
    }
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="dingxin_inner_validation_splits",
        logger=LOGGER,
        initial_progress={
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "formal_screen_requires_nested_target_refit": True,
        },
    ) as progress:
        missing = sorted(name for name, path in source_paths.items() if not path.is_file())
        if missing:
            raise FileNotFoundError(f"Dingxin inner split sources missing: {missing}")
        outer = json.loads(
            source_paths["outer_split_manifest"].read_text(encoding="utf-8")
        )
        contexts = pd.read_csv(source_paths["context_catalog"])
        bindings = pd.read_csv(source_paths["fold_task_binding"])
        plans, role_frame = build_dingxin_inner_split_plans(
            context_catalog=contexts,
            outer_split_manifest=outer,
        )
        coverage_rows = build_inner_task_coverage_rows(
            plans=plans,
            task_bindings=bindings,
        )
        rebuilt_plans, rebuilt_roles = build_dingxin_inner_split_plans(
            context_catalog=contexts,
            outer_split_manifest=outer,
        )
        deterministic = (
            _plan_hash(plans, role_frame)
            == _plan_hash(rebuilt_plans, rebuilt_roles)
        )
        overlap_count = interval_overlap_count(role_frame)
        acceptance_rows = build_inner_split_acceptance_rows(
            plans=plans,
            role_rows=role_frame,
            coverage_rows=coverage_rows,
            overlap_count=overlap_count,
            deterministic_rebuild=deterministic,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_inner_split_outputs(
            run_root=run_root,
            run_id=config.run_id,
            status=status,
            source_manifest={
                "format": "chronaris.dingxin_inner_split_sources.v1",
                "source_hashes": {
                    name: sha256_file(path) for name, path in source_paths.items()
                },
                "available_input_context_count": int(
                    contexts["input_fully_observed"].sum()
                ),
                "shared_vehicle_overlap_count": overlap_count,
            },
            plans=plans,
            role_rows=role_frame.to_dict("records"),
            coverage_rows=coverage_rows,
            acceptance_rows=acceptance_rows,
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            fold_count=len(plans),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
        )
        return DingxinInnerSplitResult(
            run_id=config.run_id,
            status=status,
            run_root=str(run_root),
            fold_count=len(plans),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _plan_hash(plans, role_frame):
    payload = {
        "plans": [plan.to_dict() for plan in plans],
        "roles": role_frame.sort_values(["fold_id", "context_id"]).to_dict("records"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
