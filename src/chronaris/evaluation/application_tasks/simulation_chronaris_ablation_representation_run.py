"""Export clean G1-to-G2 representations for locked Chronaris ablations."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_pretraining_run import (
    CHRONARIS_ABLATION_VARIANTS,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.training import (
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import (
    FusionStreamBatch,
    load_fusion_stream_batch,
    select_observation_batch,
    write_fusion_stream_batch,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger(
    "chronaris.pipelines.task_eval.simulation_chronaris_ablation_representations"
)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationRepresentationConfig:
    run_id: str = "2026-07-12_simulation-chronaris-ablation-representations"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-chronaris-ablation-pretraining"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    variants: tuple[str, ...] = CHRONARIS_ABLATION_VARIANTS
    export_batch_size: int = 32
    device: str = "cpu"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationChronarisAblationRepresentationResult:
    run_id: str
    status: str
    variant_seed_count: int
    export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    compact_run_root: str
    heavy_run_root: str
    report_path: str
    evidence_manifest_path: str


def run_simulation_chronaris_ablation_representations(
    config: SimulationChronarisAblationRepresentationConfig,
) -> SimulationChronarisAblationRepresentationResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_compact = Path(config.compact_output_root) / config.pretraining_run_id
    pretraining_heavy = Path(config.heavy_output_root) / config.pretraining_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(
        (pretraining_compact / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    if evidence.get("status") != "completed":
        raise ValueError("ablation representations require completed pretraining")
    checkpoints = _require_complete_checkpoints(
        pretraining_heavy, seeds=config.seeds, variants=config.variants
    )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("ablation representation requested unavailable CUDA")
    data = load_simulation_locked_context_data(config.simulation_root)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_chronaris_ablation_representations",
        logger=LOGGER,
        initial_progress={
            "checkpoint_count_verified_before_g2_open": len(checkpoints),
            "task_oracle_opened": False,
        },
    ) as progress:
        seed_variant_rows = []
        export_rows = []
        for seed in config.seeds:
            for variant in config.variants:
                checkpoint = checkpoints[(seed, variant)]
                encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(
                    checkpoint, device=config.device
                )
                adapter = TrainedFusionAdapter(
                    encoder=encoder,
                    normalizer=normalizer,
                    fold_id=data.fold.fold_id,
                    checkpoint_sha256=sha256_file(checkpoint),
                )
                method_name = f"chronaris_{variant}"
                local_rows = []
                for role in ("train", "validation", "held_out"):
                    destination = (
                        heavy_root
                        / "representations"
                        / f"seed_{seed}"
                        / method_name
                        / role
                    )
                    output, status = _export_role(
                        adapter=adapter,
                        method_name=method_name,
                        batch=select_observation_batch(
                            data.batch, data.role_sample_ids[role]
                        ),
                        destination=destination,
                        batch_size=config.export_batch_size,
                        resume=config.resume,
                    )
                    row = {
                        "seed": seed,
                        "variant": variant,
                        "method_name": method_name,
                        "role": role,
                        "status": status,
                        "sample_count": len(output.sample_ids),
                        "checkpoint_sha256": output.checkpoint_sha256,
                        "representation_sha256": sha256_file(
                            destination / "fusion_stream.npz"
                        ),
                        "output_root": str(destination),
                    }
                    local_rows.append(row)
                    export_rows.append(row)
                seed_variant_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "checkpoint_variant": payload["encoder_manifest"][
                            "backbone_config"
                        ]["variant"],
                        "export_count": len(local_rows),
                        "task_oracle_opened": False,
                    }
                )
                progress.update(
                    "simulation_chronaris_ablation_representation_complete",
                    seed=seed,
                    variant=variant,
                    export_count=len(local_rows),
                )
                del encoder, adapter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        acceptance = _acceptance_rows(config, seed_variant_rows, export_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            seed_variant_rows=seed_variant_rows,
            export_rows=export_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            variant_seed_count=len(seed_variant_rows),
            export_count=len(export_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationChronarisAblationRepresentationResult(
        run_id=config.run_id,
        status=status,
        variant_seed_count=len(seed_variant_rows),
        export_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def _require_complete_checkpoints(root, *, seeds, variants):
    paths = {}
    for seed in seeds:
        for variant in variants:
            path = (
                root
                / "checkpoints"
                / f"seed_{seed}"
                / variant
                / "chronaris"
                / "best.pt"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_variant = payload["encoder_manifest"]["backbone_config"][
                "variant"
            ]
            if (
                payload.get("training_status") != "completed"
                or int(payload["seed"]) != seed
                or checkpoint_variant != variant
            ):
                raise ValueError("ablation checkpoint is incomplete or mislabelled")
            paths[(seed, variant)] = path
    return paths


def _export_role(*, adapter, method_name, batch, destination, batch_size, resume):
    if resume and (destination / "fusion_stream.npz").is_file():
        output = load_fusion_stream_batch(destination)
        if (
            output.sample_ids == batch.sample_ids
            and output.method_name == method_name
            and output.checkpoint_sha256 == adapter.checkpoint_sha256
        ):
            return output, "resumed"
    output = _rename_output(_encode_in_batches(adapter, batch, batch_size), method_name)
    write_fusion_stream_batch(
        output, root=destination, export_role="application_ablation"
    )
    return load_fusion_stream_batch(destination), "completed"


def _encode_in_batches(adapter, batch, batch_size):
    outputs = []
    for offset in range(0, len(batch.sample_ids), batch_size):
        ids = batch.sample_ids[offset : offset + batch_size]
        outputs.append(adapter(select_observation_batch(batch, ids)))
    first = outputs[0]
    return FusionStreamBatch(
        sample_ids=tuple(value for output in outputs for value in output.sample_ids),
        timestamps_s=torch.cat([output.timestamps_s for output in outputs]),
        sequence_embedding=torch.cat([output.sequence_embedding for output in outputs]),
        valid_mask=torch.cat([output.valid_mask for output in outputs]),
        pooled_embedding=torch.cat([output.pooled_embedding for output in outputs]),
        method_name=first.method_name,
        fold_id=first.fold_id,
        checkpoint_sha256=first.checkpoint_sha256,
        source_sample_hashes=tuple(
            value for output in outputs for value in output.source_sample_hashes
        ),
    )


def _rename_output(output, method_name):
    return FusionStreamBatch(
        sample_ids=output.sample_ids,
        timestamps_s=output.timestamps_s,
        sequence_embedding=output.sequence_embedding,
        valid_mask=output.valid_mask,
        pooled_embedding=output.pooled_embedding,
        method_name=method_name,
        fold_id=output.fold_id,
        checkpoint_sha256=output.checkpoint_sha256,
        source_sample_hashes=output.source_sample_hashes,
    )


def _acceptance_rows(config, rows, exports):
    expected = len(config.seeds) * len(config.variants)
    return (
        _check("all_variant_seed_exports", len(rows) == expected, len(rows), expected),
        _check("three_roles_per_variant_seed", all(row["export_count"] == 3 for row in rows), [row["export_count"] for row in rows], 3),
        _check("checkpoint_variant_matches", all(row["variant"] == row["checkpoint_variant"] for row in rows), True, True),
        _check("all_representation_exports", len(exports) == expected * 3, len(exports), expected * 3),
        _check("task_oracle_closed", all(not row["task_oracle_opened"] for row in rows), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "inventory": root / "seed_variant_inventory.csv",
        "exports": root / "representation_inventory.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["seed_variant_rows"]).to_csv(paths["inventory"], index=False)
    pd.DataFrame(values["export_rows"]).to_csv(paths["exports"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_chronaris_ablation_representations.v1",
        "config": asdict(values["config"]),
        "checkpoint_set_verified_before_g2_open": True,
        "task_oracle_opened": False,
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# Chronaris 机制消融 G1 到 G2 表示导出",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['export_rows'])} 份消融表示；任务真值仍保持关闭。",
        "",
    )), encoding="utf-8")
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_chronaris_ablation_representations.py "
        f"--run-id {values['config'].run_id} --export-batch-size {values['config'].export_batch_size} "
        f"--device {values['config'].device} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_chronaris_ablation_representation_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "variant_seed_count": len(values["seed_variant_rows"]),
        "export_count": len(values["export_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
