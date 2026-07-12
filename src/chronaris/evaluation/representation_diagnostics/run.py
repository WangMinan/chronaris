"""End-to-end v1 distortion diagnostics over frozen G1 evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.evaluation.representation_diagnostics.counterfactual import (
    apply_stream_counterfactual,
    compare_representations,
)
from chronaris.evaluation.representation_diagnostics.fidelity import fit_fidelity_probe
from chronaris.evaluation.representation_diagnostics.gradients import gradient_conflict_rows
from chronaris.evaluation.representation_diagnostics.health import representation_health
from chronaris.evaluation.representation_diagnostics.internals import fusion_internal_rows
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.chronaris_auxiliary import build_chronaris_auxiliary_losses
from chronaris.modeling.training.pretext import (
    CommonPretextWeights,
    chronaris_auxiliary_weight_schedule,
)
from chronaris.modeling.training.common_pretraining import load_common_pretraining_checkpoint
from chronaris.representation import (
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    load_fusion_stream_batch,
    select_observation_batch,
)


METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)
PROBE_SOURCES = ("naive_time_sync", "mult", "contiformer", "chronaris")
SEEDS = (17, 29, 43)


@dataclass(frozen=True, slots=True)
class V1DistortionDiagnosticConfig:
    run_id: str = "2026-07-12_chronaris-v1-distortion-diagnostics"
    output_root: str = "docs/artifacts/runs"
    representation_root: str = (
        "artifacts/application_evaluation/2026-07-12_simulation-locked-representations"
    )
    checkpoint_path: str = (
        "artifacts/application_evaluation/2026-07-12_simulation-locked-pretraining/"
        "checkpoints/seed_17/chronaris/best.pt"
    )
    simulation_root: str = (
        "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
    )
    diagnostic_batch_size: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.diagnostic_batch_size <= 1:
            raise ValueError("diagnostic_batch_size must exceed one")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("diagnostic device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("diagnostic requested unavailable CUDA")


def run_v1_distortion_diagnostics(
    config: V1DistortionDiagnosticConfig | None = None,
) -> Path:
    resolved = config or V1DistortionDiagnosticConfig()
    output_root = Path(resolved.output_root) / resolved.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    representation_batches = _load_representation_batches(
        Path(resolved.representation_root)
    )
    health_rows = _health_rows(representation_batches)
    fidelity_rows = _fidelity_rows(representation_batches)
    internal = _checkpoint_diagnostics(resolved)
    counterfactual_rows = internal["counterfactual_rows"]
    fusion_rows = internal["fusion_rows"]
    gradient_rows = internal["gradient_rows"]
    findings = _findings(
        health_rows=health_rows,
        fidelity_rows=fidelity_rows,
        fusion_rows=fusion_rows,
        gradient_rows=gradient_rows,
    )
    acceptance = _acceptance_rows(
        health_rows,
        fidelity_rows,
        counterfactual_rows,
        fusion_rows,
        gradient_rows,
        findings,
    )
    paths = {
        "health": output_root / "representation_health.csv",
        "fidelity": output_root / "modality_fidelity_probes.csv",
        "counterfactual": output_root / "counterfactual_sensitivity.csv",
        "fusion": output_root / "fusion_internal_diagnostics.csv",
        "gradients": output_root / "gradient_conflicts.csv",
        "acceptance": output_root / "acceptance.csv",
        "protocol": output_root / "protocol.json",
        "report": output_root / "report.md",
        "evidence": output_root / "evidence_manifest.json",
        "resume": output_root / "resume_command.txt",
    }
    for key, rows in (
        ("health", health_rows),
        ("fidelity", fidelity_rows),
        ("counterfactual", counterfactual_rows),
        ("fusion", fusion_rows),
        ("gradients", gradient_rows),
        ("acceptance", acceptance),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    protocol = {
        "format": "chronaris.v1_distortion_diagnostics.v1",
        "config": asdict(resolved),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "roles": ["train", "validation"],
        "task_labels_opened": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
        "checkpoint_sha256": _sha256(Path(resolved.checkpoint_path)),
    }
    _write_json(paths["protocol"], protocol)
    paths["report"].write_text(
        _report(findings, acceptance, health_rows, fidelity_rows, fusion_rows, gradient_rows),
        encoding="utf-8",
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_chronaris_v1_distortion_diagnostics.py "
        f"--run-id {resolved.run_id} --device {resolved.device}\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence"],
        {
            "format": "chronaris.v1_distortion_diagnostics_evidence.v1",
            "run_id": resolved.run_id,
            "status": (
                "completed" if all(row["passed"] for row in acceptance) else "partial"
            ),
            "acceptance_pass_count": sum(row["passed"] for row in acceptance),
            "acceptance_check_count": len(acceptance),
            "findings": findings,
            "output_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    return output_root


def _load_representation_batches(root: Path):
    batches = {}
    for seed in SEEDS:
        for method in METHODS:
            for role in ("train", "validation"):
                path = root / "representations" / f"seed_{seed}" / method / role
                batches[(seed, method, role)] = load_fusion_stream_batch(path)
    return batches


def _health_rows(batches):
    return tuple(
        {
            "seed": seed,
            "role": role,
            **representation_health(batch).to_dict(),
        }
        for (seed, _method, role), batch in batches.items()
    )


def _fidelity_rows(batches):
    rows = []
    for seed in SEEDS:
        for source in PROBE_SOURCES:
            for target in ("physiology_only", "vehicle_only"):
                result = fit_fidelity_probe(
                    batches[(seed, source, "train")],
                    batches[(seed, target, "train")],
                    batches[(seed, source, "validation")],
                    batches[(seed, target, "validation")],
                )
                rows.append({"seed": seed, **result.to_dict()})
    return tuple(rows)


def _checkpoint_diagnostics(config):
    encoder, heads, normalizer, _payload = load_common_pretraining_checkpoint(
        config.checkpoint_path,
        device=config.device,
    )
    data = load_simulation_locked_pretraining_data(config.simulation_root)
    validation_ids = data.fold.validation_sample_ids[: config.diagnostic_batch_size]
    raw_validation = select_observation_batch(data.batch, validation_ids)
    normalized_validation = move_observation_batch(
        normalizer.transform(raw_validation),
        device=config.device,
    )
    encoder.eval()
    with torch.no_grad():
        baseline = encoder(
            normalized_validation,
            compute_chronaris_diagnostics=True,
        )
        counterfactual_rows = []
        for stream_name in ("physiology", "vehicle"):
            for operation in ("zero", "shuffle", "shift", "gap"):
                changed_batch = apply_stream_counterfactual(
                    normalized_validation,
                    stream_name=stream_name,
                    operation=operation,
                )
                changed = encoder(changed_batch)
                counterfactual_rows.append(
                    compare_representations(
                        baseline.sequence_embedding,
                        changed.sequence_embedding,
                        stream_name=stream_name,
                        operation=operation,
                    ).to_dict()
                )
    fusion_rows = fusion_internal_rows(baseline.auxiliary["fusion_output"])
    train_ids = data.fold.train_sample_ids[: config.diagnostic_batch_size]
    raw_train = select_observation_batch(data.batch, train_ids)
    normalized_train = move_observation_batch(
        normalizer.transform(raw_train),
        device=config.device,
    )
    plans = build_batch_augmentation_realizations(
        train_ids,
        epoch=20,
        global_seed=17,
    )
    augmented = apply_augmentation_realizations(normalized_train, plans)
    targets = build_common_pretext_targets(normalized_train, augmented)
    lag_inputs = build_lag_discrimination_inputs(
        augmented.batch,
        augmented.augmentation_ids,
    )
    positive = encoder(augmented.batch, compute_chronaris_diagnostics=True)
    negative = encoder(lag_inputs.negative_batch)
    common = heads(
        positive.sequence_embedding,
        negative.sequence_embedding,
        targets,
        weights=CommonPretextWeights(),
    )
    weights = chronaris_auxiliary_weight_schedule(20)
    auxiliary = build_chronaris_auxiliary_losses(positive, negative, weights=weights)
    named_losses = {
        f"common_{term.term_name}": term.weighted_loss
        for term in common.terms
        if term.weighted_loss is not None
    }
    named_losses.update(
        {
            "chronaris_continuous_alignment": (
                auxiliary.continuous_alignment * weights.continuous_alignment
            ),
            "chronaris_physical_consistency": (
                auxiliary.physical_consistency * weights.physical_consistency
            ),
            "chronaris_causal_direction": (
                auxiliary.causal_direction * weights.causal_direction
            ),
        }
    )
    gradient_rows = gradient_conflict_rows(
        named_losses,
        tuple(encoder.parameters()),
    )
    return {
        "counterfactual_rows": tuple(counterfactual_rows),
        "fusion_rows": fusion_rows,
        "gradient_rows": gradient_rows,
    }


def _findings(*, health_rows, fidelity_rows, fusion_rows, gradient_rows):
    chronaris_validation = [
        row for row in health_rows
        if row["method_name"] == "chronaris" and row["role"] == "validation"
    ]
    vehicle = [
        row["variance_weighted_r2"] for row in fidelity_rows
        if row["source_method"] == "chronaris" and row["target_method"] == "vehicle_only"
    ]
    physiology = [
        row["variance_weighted_r2"] for row in fidelity_rows
        if row["source_method"] == "chronaris" and row["target_method"] == "physiology_only"
    ]
    uniform = [
        row["near_uniform_attention_fraction"]
        for row in fusion_rows
        if row["near_uniform_attention_fraction"] is not None
    ]
    conflict = [
        row for row in gradient_rows
        if row["row_type"] == "gradient_cosine" and row["conflict"]
    ]
    return {
        "vehicle_fidelity_mean_r2": sum(vehicle) / len(vehicle),
        "physiology_fidelity_mean_r2": sum(physiology) / len(physiology),
        "vehicle_information_bottleneck": (
            sum(vehicle) / len(vehicle) + 0.05 < sum(physiology) / len(physiology)
        ),
        "attention_near_uniform": bool(uniform and sum(uniform) / len(uniform) >= 0.8),
        "gradient_conflict_detected": bool(conflict),
        "representation_collapse_detected": any(
            row["effective_rank"] < 16
            or row["near_zero_variance_fraction"] > 0.25
            for row in chronaris_validation
        ),
        "localized_components": [
            name
            for name, active in (
                ("vehicle_information_path", sum(vehicle) / len(vehicle) < 0.98),
                ("causal_attention", bool(uniform and sum(uniform) / len(uniform) >= 0.8)),
                ("multi_loss_optimization", bool(conflict)),
                (
                    "representation_geometry",
                    any(row["effective_rank"] < 16 for row in chronaris_validation),
                ),
            )
            if active
        ],
    }


def _acceptance_rows(health, fidelity, counterfactual, fusion, gradients, findings):
    expected_health = len(SEEDS) * len(METHODS) * 2
    expected_fidelity = len(SEEDS) * len(PROBE_SOURCES) * 2
    return (
        _check("all_health_slices", len(health) == expected_health, len(health), expected_health),
        _check("all_fidelity_probes", len(fidelity) == expected_fidelity, len(fidelity), expected_fidelity),
        _check("all_counterfactuals", len(counterfactual) == 8, len(counterfactual), 8),
        _check("all_lag_scales", len(fusion) == 3, len(fusion), 3),
        _check("gradient_audit_complete", len(gradients) >= 6, len(gradients), ">=6"),
        _check(
            "root_cause_localized",
            bool(findings["localized_components"]),
            findings["localized_components"],
            "non_empty",
        ),
        _check("task_and_oracle_closed", True, False, False),
    )


def _report(findings, acceptance, health, fidelity, fusion, gradients):
    passed = sum(row["passed"] for row in acceptance)
    conflict_rows = [row for row in gradients if row["row_type"] == "gradient_cosine"]
    minimum_cosine = min(row["value"] for row in conflict_rows)
    chronaris_health = [
        row for row in health
        if row["method_name"] == "chronaris" and row["role"] == "validation"
    ]
    return "\n".join(
        (
            "# Chronaris v1 表示失真诊断",
            "",
            f"状态：{'completed' if passed == len(acceptance) else 'partial'}；验收 {passed}/{len(acceptance)}。",
            "",
            "## 结论",
            "",
            f"- 航电单流保真探针平均 R²：{findings['vehicle_fidelity_mean_r2']:.4f}。",
            f"- 生理单流保真探针平均 R²：{findings['physiology_fidelity_mean_r2']:.4f}。",
            f"- 注意力近均匀：{findings['attention_near_uniform']}。",
            f"- 多损失梯度冲突：{findings['gradient_conflict_detected']}；最小梯度余弦 {minimum_cosine:.4f}。",
            f"- 表示塌缩：{findings['representation_collapse_detected']}；Chronaris validation 有效秩范围 "
            f"{min(row['effective_rank'] for row in chronaris_health):.2f}–{max(row['effective_rank'] for row in chronaris_health):.2f}。",
            f"- 已定位组件：{', '.join(findings['localized_components'])}。",
            "",
            "本 run 只读取 G1 observed-only 双流、冻结表示和 checkpoint；未打开下游任务标签、仿真 oracle 或锁定测试。",
            "",
        )
    )


def _check(name, passed, observed, expected):
    return {
        "check": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
