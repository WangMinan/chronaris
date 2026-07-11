"""Export clean G1-to-G2 representations from every locked seed checkpoint."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
    export_application_context_representations,
)
from chronaris.evaluation.application_tasks.simulation_locked_context_data import (
    load_simulation_locked_context_data,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    TRAINABLE_FUSION_METHODS,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import validate_fusion_method_alignment
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.simulation_locked_representations")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class SimulationLockedRepresentationConfig:
    run_id: str = "2026-07-12_simulation-locked-representations"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    pretraining_run_id: str = "2026-07-12_simulation-locked-pretraining"
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    selected_candidates_path: str = (
        "docs/artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/"
        "selected_candidates.json"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    export_batch_size: int = 32
    baseline_device: str = "auto"
    chronaris_device: str = "cpu"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class SimulationLockedRepresentationResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    seed_count: int
    export_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_simulation_locked_representations(config: SimulationLockedRepresentationConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    pretraining_root = Path(config.heavy_output_root) / config.pretraining_run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    selected_payload = json.loads(
        Path(config.selected_candidates_path).read_text(encoding="utf-8")
    )
    selected_ids = {
        method: str(selected_payload[method]["candidate_id"])
        for method in TRAINABLE_FUSION_METHODS
    }
    checkpoints = require_complete_locked_checkpoint_set(
        pretraining_root,
        seeds=config.seeds,
        selected_ids=selected_ids,
    )
    pretraining_data = load_simulation_locked_pretraining_data(config.simulation_root)
    data = load_simulation_locked_context_data(config.simulation_root)
    baseline_device = _resolve_device(config.baseline_device)
    chronaris_device = _resolve_device(config.chronaris_device)
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="simulation_locked_representation_export",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "checkpoint_count_verified_before_g2_open": len(checkpoints),
            "task_oracle_opened": False,
        },
    ) as progress:
        export_rows = []
        seed_rows = []
        for seed in config.seeds:
            adapters, checkpoint_rows = load_locked_seed_adapters(
                seed=seed,
                checkpoints=checkpoints,
                selected_ids=selected_ids,
                pretraining_data=pretraining_data,
                heavy_root=heavy_root,
                resume=config.resume,
                baseline_device=baseline_device,
                chronaris_device=chronaris_device,
            )
            outputs, rows, alignment = export_application_context_representations(
                adapters=adapters,
                batch=data.batch,
                role_sample_ids=data.role_sample_ids,
                output_root=heavy_root / "representations" / f"seed_{seed}",
                resume=config.resume,
                batch_size=config.export_batch_size,
            )
            verified_alignment = {
                role: validate_fusion_method_alignment(
                    [outputs[method][role] for method in APPLICATION_METHODS]
                )
                for role in ("train", "validation", "held_out")
            }
            if alignment != verified_alignment:
                raise ValueError("locked representation alignment changed after export")
            export_rows.extend({"seed": seed, **row} for row in rows)
            seed_rows.append(
                {
                    "seed": seed,
                    "checkpoint_count": len(checkpoint_rows),
                    "export_count": len(rows),
                    "train_context_count": len(data.role_sample_ids["train"]),
                    "validation_context_count": len(data.role_sample_ids["validation"]),
                    "g2_context_count": len(data.role_sample_ids["held_out"]),
                    "alignment_role_count": len(alignment),
                    "task_oracle_opened": False,
                }
            )
            progress.update(
                "locked_seed_representation_complete",
                seed=seed,
                export_count=len(rows),
            )
        acceptance = _acceptance_rows(config, seed_rows, export_rows)
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            data=data,
            seed_rows=seed_rows,
            export_rows=export_rows,
            acceptance=acceptance,
            status=status,
            baseline_device=baseline_device,
            chronaris_device=chronaris_device,
        )
        progress.finish(
            status=status,
            export_count=len(export_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return SimulationLockedRepresentationResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        seed_count=len(seed_rows),
        export_count=len(export_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def require_complete_locked_checkpoint_set(root, *, seeds, selected_ids):
    paths = {}
    for seed in seeds:
        for method in TRAINABLE_FUSION_METHODS:
            path = (
                root / "checkpoints" / f"seed_{seed}" / method / "best.pt"
                if method == "chronaris"
                else root / "checkpoints" / f"seed_{seed}" / method / selected_ids[method] / "best.pt"
            )
            if not path.is_file():
                raise FileNotFoundError(f"locked representation requires complete checkpoint set: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("training_status") != "completed" or int(payload["seed"]) != seed:
                raise ValueError("locked representation checkpoint is incomplete or seed-mismatched")
            paths[(seed, method)] = path
    return paths


def load_locked_seed_adapters(
    *, seed, checkpoints, selected_ids, pretraining_data, heavy_root, resume,
    baseline_device="cpu", chronaris_device="cpu"
):
    adapters = {}
    rows = []
    normalizer = None
    normalizer_hashes = set()
    fold_id = f"simulation_g1_to_g2_clean_locked__seed_{seed}"
    for method in TRAINABLE_FUSION_METHODS:
        path = checkpoints[(seed, method)]
        device = chronaris_device if method == "chronaris" else baseline_device
        encoder, _heads, loaded_normalizer, payload = load_common_pretraining_checkpoint(
            path,
            device=_resolve_device(device),
        )
        if payload["candidate_config"]["candidate_id"] != selected_ids[method]:
            raise ValueError("locked representation candidate mismatch")
        normalizer_hashes.add(loaded_normalizer.to_manifest()["transform_sha256"])
        normalizer = loaded_normalizer if normalizer is None else normalizer
        checkpoint_hash = sha256_file(path)
        adapters[method] = TrainedFusionAdapter(
            encoder=encoder,
            normalizer=loaded_normalizer,
            fold_id=fold_id,
            checkpoint_sha256=checkpoint_hash,
        )
        rows.append({"method_name": method, "path": str(path), "sha256": checkpoint_hash})
    if len(normalizer_hashes) != 1 or normalizer is None:
        raise ValueError("locked seed checkpoints do not share one normalizer")
    naive_path = heavy_root / "checkpoints" / f"seed_{seed}" / "naive_time_sync" / "best.pt"
    if not naive_path.is_file() or not resume:
        naive = NaiveTimeSyncEncoder().fit(
            pretraining_data.batch,
            train_sample_ids=pretraining_data.fold.train_sample_ids,
            held_out_sample_ids=(
                pretraining_data.fold.validation_sample_ids
                + pretraining_data.fold.held_out_sample_ids
            ),
            normalizer=normalizer,
        )
        save_naive_time_sync_checkpoint(naive_path, encoder=naive)
    naive_hash = sha256_file(naive_path)
    adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
        encoder=load_naive_time_sync_checkpoint(naive_path),
        fold_id=fold_id,
        checkpoint_sha256=naive_hash,
    )
    rows.append({"method_name": "naive_time_sync", "path": str(naive_path), "sha256": naive_hash})
    return adapters, rows


def _resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("simulation representation device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("simulation representation requested unavailable CUDA")
    return value


def _acceptance_rows(config, seeds, exports):
    expected_exports = len(config.seeds) * 18
    return (
        _check("all_locked_seeds", len(seeds) == len(config.seeds), len(seeds), len(config.seeds)),
        _check("six_checkpoints_per_seed", all(row["checkpoint_count"] == 6 for row in seeds), [row["checkpoint_count"] for row in seeds], 6),
        _check("eighteen_exports_per_seed", all(row["export_count"] == 18 for row in seeds), [row["export_count"] for row in seeds], 18),
        _check("all_exports", len(exports) == expected_exports, len(exports), expected_exports),
        _check("fixed_context_counts", all((row["train_context_count"], row["validation_context_count"], row["g2_context_count"]) == (384, 96, 192) for row in seeds), [(row["train_context_count"], row["validation_context_count"], row["g2_context_count"]) for row in seeds], [384, 96, 192]),
        _check("three_aligned_roles", all(row["alignment_role_count"] == 3 for row in seeds), [row["alignment_role_count"] for row in seeds], 3),
        _check("task_oracle_closed", all(not row["task_oracle_opened"] for row in seeds), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "seeds": root / "seed_inventory.csv",
        "exports": root / "representation_inventory.csv",
        "data": root / "data_manifest.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    pd.DataFrame(values["seed_rows"]).to_csv(paths["seeds"], index=False)
    pd.DataFrame(values["export_rows"]).to_csv(paths["exports"], index=False)
    pd.DataFrame(values["data"].sample_manifest_rows).to_csv(paths["data"], index=False)
    pd.DataFrame(values["acceptance"]).to_csv(paths["acceptance"], index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.simulation_locked_representation_protocol.v1",
        "config": asdict(values["config"]),
        "fold": values["data"].fold.to_dict(),
        "checkpoint_set_verified_before_g2_open": True,
        "task_oracle_opened": False,
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# G1 到 G2 锁定融合表示导出",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"三个随机种子共导出 {len(values['export_rows'])} 份六方法 train/validation/G2 表示。",
        "所有训练 checkpoint 完成后才读取 G2 原始观测；任务真值和指标仍保持关闭。",
        "",
    )), encoding="utf-8")
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_simulation_locked_representations.py "
        f"--run-id {values['config'].run_id} --export-batch-size {values['config'].export_batch_size} "
        f"--baseline-device {values['baseline_device']} --chronaris-device {values['chronaris_device']} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.simulation_locked_representation_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "seed_count": len(values["seed_rows"]),
        "export_count": len(values["export_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "task_oracle_opened": False,
        "baseline_device": values["baseline_device"],
        "chronaris_device": values["chronaris_device"],
        "heavy_run_root": str(values["heavy_root"]),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
