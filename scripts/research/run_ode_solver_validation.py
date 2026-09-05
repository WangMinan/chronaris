"""Select the frozen Chronaris ODE scheme using simulation validation only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from chronaris.modeling.training import (
    CandidateScreenConfig,
    EncoderCandidateConfig,
    train_pretext_candidate,
)
from chronaris.representation import (
    AugmentationPolicy,
    FoldLineage,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
    load_simulation_observed_context,
)


REPO = Path(__file__).resolve().parents[2]
SIMULATION_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
RUN_ROOT = REPO / "docs/artifacts/runs/2026-09-01_ode-solver-validation"
HEAVY_ROOT = REPO / "artifacts/application_evaluation/2026-09-01_ode-solver-validation"


def main() -> int:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing ODE validation requires available CUDA")
    train_paths = _scenario_paths("train")[: args.train_limit]
    validation_paths = _scenario_paths("validation")[: args.validation_limit]
    if len(train_paths) != args.train_limit or len(validation_paths) != args.validation_limit:
        raise RuntimeError("simulation ODE validation input count is incomplete")
    samples = [
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"ode_train_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(train_paths)
    ]
    samples.extend(
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"ode_validation_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(validation_paths)
    )
    batch = collate_observation_samples(samples)
    train_ids = tuple(sample.sample_id for sample in samples[: len(train_paths)])
    validation_ids = tuple(
        sample.sample_id for sample in samples[len(train_paths) : -1]
    )
    fold = FoldLineage(
        fold_id="simulation_ode_validation",
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        held_out_sample_ids=(samples[-1].sample_id,),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    schema = samples[0].schema
    rows = []
    for label, ode_method, max_step in (
        ("single_step_euler", "euler", None),
        ("euler_max_0p5s", "euler", 0.5),
        ("rk4", "rk4", None),
    ):
        print(f"[ode-validation] {label} starting", flush=True)
        result = train_pretext_candidate(
            "chronaris",
            candidate=EncoderCandidateConfig(candidate_id="C", hidden_dim=32),
            batch=batch,
            fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=tuple((name, name) for name in schema.vehicle_feature_names),
            normalizer=normalizer,
            output_root=HEAVY_ROOT / label,
            config=CandidateScreenConfig(
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.epochs,
                seed=args.seed,
                device=args.device,
                ode_method=ode_method,
                max_ode_step_s=max_step,
            ),
            augmentation_policy=AugmentationPolicy(),
            chronaris_fusion_kind="safe_lag",
            chronaris_mechanism_enabled=True,
            chronaris_lag_aware_weight=0.1,
            include_candidate_subdirectory=False,
            resume=args.resume,
        )
        payload = torch.load(
            result.best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        best_row = next(
            row for row in result.epoch_rows if int(row["epoch"]) == result.best_epoch
        )
        validation_terms = {
            row["term_name"]: row
            for row in best_row["mechanism_validation"]["terms"]
        }
        alignment = validation_terms["chronaris_continuous_alignment"]["raw_loss"]
        gradients = [
            float(row["related_parameter_gradient_norm"])
            for row in payload["training_rows"]
            if row["term_name"] == "chronaris_continuous_alignment"
        ]
        rows.append(
            {
                "label": label,
                "ode_method": ode_method,
                "max_ode_step_s": max_step,
                "validation_alignment_loss": alignment,
                "training_elapsed_s": result.training_elapsed_s,
                "gradients_finite": bool(gradients)
                and all(math.isfinite(value) and value > 0 for value in gradients),
                "best_epoch": result.best_epoch,
                "protocol_sha256": result.protocol_sha256,
                "checkpoint_path": result.best_checkpoint_path,
            }
        )
        print(
            f"[ode-validation] {label} alignment={alignment:.6f} "
            f"elapsed={result.training_elapsed_s:.2f}s",
            flush=True,
        )
    baseline_runtime = next(
        row["training_elapsed_s"] for row in rows if row["label"] == "single_step_euler"
    )
    eligible = [
        row
        for row in rows
        if row["gradients_finite"]
        and row["training_elapsed_s"] <= 2.0 * baseline_runtime
        and row["validation_alignment_loss"] is not None
    ]
    if not eligible:
        raise RuntimeError("no ODE solver satisfies finite-gradient and runtime gates")
    selected = min(eligible, key=lambda row: row["validation_alignment_loss"])
    payload = {
        "seed": args.seed,
        "epochs": args.epochs,
        "device": args.device,
        "train_sample_count": len(train_ids),
        "validation_sample_count": len(validation_ids),
        "held_out_opened": False,
        "runtime_limit_ratio": 2.0,
        "rows": rows,
        "selected": selected,
    }
    _atomic_write_json(RUN_ROOT / "solver_validation_metrics.json", payload)
    _atomic_write_text(RUN_ROOT / "report.md", _render_report(payload))
    return 0


def _scenario_paths(split: str) -> list[Path]:
    return sorted(
        (SIMULATION_ROOT / split).glob(
            "*/*/clean_asynchronous/raw_dual_stream.npz"
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--validation-limit", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _atomic_write_json(path: Path, payload) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _render_report(payload) -> str:
    selected = payload["selected"]
    lines = [
        "# 连续演化数值方案训练内验证选择",
        "",
        "本实验只使用仿真训练集拟合，并以仿真训练内验证（validation）的连续对齐损失选择数值方案；预留样本和应用任务结果均未打开。下表同时检查梯度有限性与相对单步 Euler 不超过 2 倍的运行时间门。",
        "",
        "| 数值方案 | 训练内对齐损失 | 训练用时（秒） | 梯度有限 | 运行时间门 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    baseline = payload["rows"][0]["training_elapsed_s"]
    names = {
        "single_step_euler": "单步 Euler",
        "euler_max_0p5s": "最大 0.5 秒 Euler 子步",
        "rk4": "四阶 Runge-Kutta",
    }
    for row in payload["rows"]:
        lines.append(
            f"| {names[row['label']]} | {row['validation_alignment_loss']:.6f} | "
            f"{row['training_elapsed_s']:.2f} | {'是' if row['gradients_finite'] else '否'} | "
            f"{'通过' if row['training_elapsed_s'] <= 2 * baseline else '未通过'} |"
        )
    lines.extend(
        [
            "",
            f"冻结选择为 **{names[selected['label']]}**。该选择只用于后续训练内候选，不根据公开数据外层折或鼎新确认结果重新选择。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
