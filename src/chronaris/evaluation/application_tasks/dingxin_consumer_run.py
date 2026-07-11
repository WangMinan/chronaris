"""Run fixed linear and MiniRocket consumers over five Dingxin folds."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_audit import (
    build_dingxin_consumer_acceptance_rows,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_models import (
    DingxinConsumerConfig,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_reporting import (
    write_dingxin_consumer_outputs,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_runtime import (
    run_dingxin_method_consumers,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    load_dingxin_fold_consumer_targets,
)
from chronaris.evaluation.application_tasks.dingxin_pretraining_aggregate import (
    DEFAULT_FOLD_RUN_IDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.representation import (
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_consumer_smoke")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinConsumerSmokeConfig:
    run_id: str = "2026-07-11_dingxin-consumer-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    binding_root: str = "docs/artifacts/runs/2026-07-11_dingxin-context-bindings"
    fold_run_ids: tuple[str, ...] = DEFAULT_FOLD_RUN_IDS
    seed: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinConsumerSmokeResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_fold_count: int
    metric_count: int
    fusion_gain_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_consumer_smoke(
    config: DingxinConsumerSmokeConfig,
) -> DingxinConsumerSmokeResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    binding_path = Path(config.binding_root) / "fold_task_binding.csv"
    consumer_config = DingxinConsumerConfig(random_state=config.seed)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_frozen_consumer_smoke",
        logger=LOGGER,
        initial_progress={
            "fold_count": len(config.fold_run_ids),
            "consumer_names": ["linear", "minirocket"],
            "threshold_scope": "outer_train_smoke_only",
            "confirmed_metrics_changed": False,
        },
    ) as progress:
        result_rows = []
        resource_rows = []
        metric_rows = []
        target_manifests = []
        source_rows = []
        prediction_hash_match_count = 0
        for fold_run_id in config.fold_run_ids:
            fold_root = Path(config.compact_output_root) / fold_run_id
            split_path = fold_root / "split_manifest.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            fold_id = str(split["fold_id"])
            targets = load_dingxin_fold_consumer_targets(
                fold_id=fold_id,
                split_manifest_path=split_path,
                binding_path=binding_path,
            )
            target_manifests.append(targets.to_manifest())
            outputs, representation_rows, alignment_hashes = _load_fold_outputs(
                fold_root
            )
            source_rows.append(
                {
                    "fold_run_id": fold_run_id,
                    "fold_id": fold_id,
                    "split_sha256": sha256_file(split_path),
                    "representation_alignment_sha256": alignment_hashes,
                    "representation_rows": representation_rows,
                    "target_source_sha256": targets.target_source_sha256,
                }
            )
            for method_name in APPLICATION_METHODS:
                initial = run_dingxin_method_consumers(
                    method_name=method_name,
                    fold_id=fold_id,
                    outputs=outputs[method_name],
                    targets=targets,
                    output_root=heavy_root / "consumers",
                    config=consumer_config,
                    resume=config.resume,
                )
                resumed = run_dingxin_method_consumers(
                    method_name=method_name,
                    fold_id=fold_id,
                    outputs=outputs[method_name],
                    targets=targets,
                    output_root=heavy_root / "consumers",
                    config=consumer_config,
                    resume=True,
                )
                if initial.metric_rows != resumed.metric_rows:
                    raise ValueError(
                        f"Dingxin consumer resume metrics changed: {fold_id}/{method_name}"
                    )
                prediction_match = (
                    initial.manifest["prediction_sha256"]
                    == resumed.manifest["prediction_sha256"]
                )
                prediction_hash_match_count += int(prediction_match)
                metric_rows.extend(dict(row) for row in initial.metric_rows)
                resource_rows.extend(dict(row) for row in initial.resource_rows)
                result_rows.append(
                    {
                        "fold_run_id": fold_run_id,
                        "fold_id": fold_id,
                        "method_name": method_name,
                        "initial_status": initial.status,
                        "initial_component_count": len(initial.component_status),
                        "resume_component_count": sum(
                            value == "resumed"
                            for value in resumed.component_status.values()
                        ),
                        "resume_all_components": all(
                            value == "resumed"
                            for value in resumed.component_status.values()
                        ),
                        "prediction_hash_match": prediction_match,
                        "prediction_sha256": initial.manifest[
                            "prediction_sha256"
                        ],
                        "protocol_sha256": initial.protocol_sha256,
                        "consumer_config_sha256": _mapping_hash(
                            asdict(consumer_config)
                        ),
                        "representation_checkpoint_sha256": json.dumps(
                            initial.manifest[
                                "representation_checkpoint_sha256"
                            ],
                            sort_keys=True,
                        ),
                        "label_used_for_encoder_training": initial.manifest[
                            "label_used_for_encoder_training"
                        ],
                        "threshold_scope": targets.threshold_scope,
                    }
                )
                progress.update(
                    "method_fold_consumers_complete",
                    fold_id=fold_id,
                    method_name=method_name,
                    initial_status=initial.status,
                    resume_status=resumed.status,
                )
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=(
                "naive_time_sync",
                "mult",
                "contiformer",
                "chronaris",
            ),
        )
        acceptance_rows = build_dingxin_consumer_acceptance_rows(
            result_rows=result_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            target_manifests=target_manifests,
            prediction_hash_match_count=prediction_hash_match_count,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = write_dingxin_consumer_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            source_manifest={
                "format": "chronaris.dingxin_consumer_sources.v1",
                "binding_path": str(binding_path),
                "binding_sha256": sha256_file(binding_path),
                "fold_sources": source_rows,
                "representation_method_count": len(APPLICATION_METHODS),
            },
            target_manifests=target_manifests,
            consumer_config=consumer_config,
            result_rows=result_rows,
            resource_rows=resource_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            acceptance_rows=acceptance_rows,
            heavy_run_root=str(heavy_root),
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            metric_count=len(metric_rows),
            fusion_gain_count=len(fusion_gain_rows),
        )
        return DingxinConsumerSmokeResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            method_fold_count=len(result_rows),
            metric_count=len(metric_rows),
            fusion_gain_count=len(fusion_gain_rows),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _load_fold_outputs(fold_root: Path):
    manifest = json.loads(
        (fold_root / "representation_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    outputs = {method: {} for method in APPLICATION_METHODS}
    rows = []
    for item in manifest["exports"]:
        method = str(item["method_name"])
        role = str(item["export_role"])
        if method not in outputs or role not in {"train", "validation", "held_out"}:
            raise ValueError("unexpected Dingxin representation method/role")
        output = load_fusion_stream_batch(item["output_root"])
        outputs[method][role] = output
        rows.append(
            {
                "method_name": method,
                "role": role,
                "sample_count": len(output.sample_ids),
                "representation_sha256": item["representation_sha256"],
                "checkpoint_sha256": output.checkpoint_sha256,
            }
        )
    if any(set(roles) != {"train", "validation", "held_out"} for roles in outputs.values()):
        raise ValueError("Dingxin representation role matrix is incomplete")
    alignment = {
        role: validate_fusion_method_alignment(
            [outputs[method][role] for method in APPLICATION_METHODS]
        )
        for role in ("train", "validation", "held_out")
    }
    return outputs, rows, alignment


def _mapping_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
