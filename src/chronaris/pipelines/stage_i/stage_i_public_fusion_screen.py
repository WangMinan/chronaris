"""GPU-preferred screening helpers for public Chronaris fusion candidates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.pipelines.stage_i.stage_i_deep_baseline import (
    StageIDeepBaselineConfig,
    run_stage_i_deep_baseline,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_reference import (
    load_public_opt_prepared_dataset,
    validate_public_opt_prepared_dataset_contract,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIPublicFusionCandidate:
    candidate_id: str
    hidden_dim: int
    num_heads: int
    layers: int
    dropout: float
    fusion_event_bias_weight: float
    fusion_lag_window_points: int | None
    fusion_normalize_states: bool


@dataclass(frozen=True, slots=True)
class StageIPublicFusionScreenConfig:
    run_id: str
    dataset_prepared_roots: Mapping[str, str]
    artifact_root: str
    report_root: str = "docs/reports"
    epochs: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_folds: int | None = 2
    seed: int = 42
    device: str = "auto"
    require_cuda: bool = False
    train_sampling_policy: str = "none"
    candidates: tuple[StageIPublicFusionCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class StageIPublicFusionScreenRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    leaderboard_csv_path: str
    report_path: str
    summary: Mapping[str, object]


DEFAULT_PUBLIC_FUSION_CANDIDATES = (
    StageIPublicFusionCandidate(
        candidate_id="fusion_h64_l2_hd4_do02_bias025_lag8_norm1",
        hidden_dim=64,
        num_heads=4,
        layers=2,
        dropout=0.2,
        fusion_event_bias_weight=0.25,
        fusion_lag_window_points=8,
        fusion_normalize_states=True,
    ),
    StageIPublicFusionCandidate(
        candidate_id="fusion_h96_l2_hd4_do02_bias050_lag8_norm1",
        hidden_dim=96,
        num_heads=4,
        layers=2,
        dropout=0.2,
        fusion_event_bias_weight=0.5,
        fusion_lag_window_points=8,
        fusion_normalize_states=True,
    ),
    StageIPublicFusionCandidate(
        candidate_id="fusion_h64_l2_hd4_do01_bias025_lag16_norm1",
        hidden_dim=64,
        num_heads=4,
        layers=2,
        dropout=0.1,
        fusion_event_bias_weight=0.25,
        fusion_lag_window_points=16,
        fusion_normalize_states=True,
    ),
)


def run_stage_i_public_fusion_screen(
    config: StageIPublicFusionScreenConfig,
) -> StageIPublicFusionScreenRunResult:
    artifact_root = Path(config.artifact_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=artifact_root,
        run_id=config.run_id,
        stage_name="stage_i_public_fusion_screen",
        logger=LOGGER,
        initial_progress={
            "datasets": sorted(config.dataset_prepared_roots),
            "requested_device": config.device,
            "require_cuda": config.require_cuda,
            "artifact_root": str(artifact_root),
        },
    ) as progress:
        return _run_stage_i_public_fusion_screen_observed(
            config=config,
            artifact_root=artifact_root,
            progress=progress,
        )


def _run_stage_i_public_fusion_screen_observed(
    *,
    config: StageIPublicFusionScreenConfig,
    artifact_root: Path,
    progress: StageIRunProgress,
) -> StageIPublicFusionScreenRunResult:
    runtime_device = resolve_torch_device_name(config.device)
    progress.update("device_resolved", runtime_device=runtime_device)
    LOGGER.info(
        "stage_i_public_fusion_screen start run_id=%s requested_device=%s resolved_device=%s datasets=%s",
        config.run_id,
        config.device,
        runtime_device,
        sorted(config.dataset_prepared_roots),
    )
    if runtime_device != "cuda":
        if config.require_cuda:
            raise RuntimeError(
                "stage_i_public_fusion_screen requires CUDA but resolved runtime_device="
                f"{runtime_device}. Use --device cuda on a CUDA host or pass "
                "--allow-cpu-debug for explicit debugging only."
            )
        LOGGER.warning(
            "stage_i_public_fusion_screen using CPU fallback run_id=%s requested_device=%s",
            config.run_id,
            config.device,
        )

    candidates = config.candidates or DEFAULT_PUBLIC_FUSION_CANDIDATES
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    per_dataset_rankings: dict[str, list[dict[str, object]]] = {}
    candidate_summaries: dict[str, object] = {}

    for dataset_id, prepared_root in config.dataset_prepared_roots.items():
        prepared = load_public_opt_prepared_dataset(prepared_root)
        validate_public_opt_prepared_dataset_contract(prepared, dataset_id=dataset_id)
        progress.update(
            "dataset_start",
            dataset_id=dataset_id,
            prepared_root=prepared_root,
            candidate_count=len(candidates),
        )
        LOGGER.info(
            "stage_i_public_fusion_screen dataset start dataset_id=%s candidate_count=%d",
            dataset_id,
            len(candidates),
        )
        dataset_rows: list[dict[str, object]] = []
        dataset_candidate_summaries: dict[str, object] = {}
        for candidate_index, candidate in enumerate(candidates, start=1):
            LOGGER.info(
                "stage_i_public_fusion_screen dataset=%s candidate %d/%d candidate_id=%s",
                dataset_id,
                candidate_index,
                len(candidates),
                candidate.candidate_id,
            )
            progress.update(
                "candidate_start",
                dataset_id=dataset_id,
                candidate_index=candidate_index,
                candidate_count=len(candidates),
                candidate=candidate.candidate_id,
            )
            candidate_root = artifact_root / dataset_id / candidate.candidate_id
            result = run_stage_i_deep_baseline(
                StageIDeepBaselineConfig(
                    model_name="chronaris_public_fusion",
                    dataset_id=dataset_id,
                    profile="window_v2",
                    prepared_artifact_root=prepared_root,
                    artifact_root=str(candidate_root),
                    epochs=config.epochs,
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                    hidden_dim=candidate.hidden_dim,
                    num_heads=candidate.num_heads,
                    layers=candidate.layers,
                    dropout=candidate.dropout,
                    fusion_event_bias_weight=candidate.fusion_event_bias_weight,
                    fusion_lag_window_points=candidate.fusion_lag_window_points,
                    fusion_normalize_states=candidate.fusion_normalize_states,
                    max_folds=config.max_folds,
                    seed=config.seed,
                    device=config.device,
                    train_sampling_policy=config.train_sampling_policy,
                )
            )
            score_payload = _extract_screen_score(dataset_id, result.summary)
            row = {
                "dataset_id": dataset_id,
                "candidate_id": candidate.candidate_id,
                "screen_metric": score_payload["screen_metric"],
                "selection_score": score_payload["selection_score"],
                "secondary_score": score_payload["secondary_score"],
                "summary_path": result.summary_path,
                "report_path": result.report_path,
                "artifact_root": result.artifact_root,
            }
            row.update(score_payload["metrics"])
            rows.append(row)
            dataset_rows.append(row)
            dataset_candidate_summaries[candidate.candidate_id] = result.summary
            LOGGER.info(
                "stage_i_public_fusion_screen dataset=%s candidate=%s metric=%s selection_score=%.4f",
                dataset_id,
                candidate.candidate_id,
                score_payload["screen_metric"],
                float(score_payload["selection_score"]),
            )
            progress.update(
                "candidate_done",
                dataset_id=dataset_id,
                candidate=candidate.candidate_id,
                screen_metric=score_payload["screen_metric"],
                selection_score=float(score_payload["selection_score"]),
                summary_path=result.summary_path,
                report_path=result.report_path,
            )

        ordered_rows = sorted(
            dataset_rows,
            key=lambda item: (
                item["selection_score"]
                if dataset_id == "nasa_csm"
                else -item["selection_score"],
                item["secondary_score"]
                if dataset_id == "nasa_csm"
                else -item["secondary_score"],
            ),
            reverse=(dataset_id == "nasa_csm"),
        )
        if dataset_id == "uab_workload_dataset":
            ordered_rows = sorted(
                dataset_rows,
                key=lambda item: (item["selection_score"], item["secondary_score"]),
            )
        per_dataset_rankings[dataset_id] = ordered_rows
        candidate_summaries[dataset_id] = dataset_candidate_summaries
        if ordered_rows:
            LOGGER.info(
                "stage_i_public_fusion_screen dataset complete dataset_id=%s best_candidate=%s",
                dataset_id,
                ordered_rows[0]["candidate_id"],
            )
            progress.update(
                "dataset_done",
                dataset_id=dataset_id,
                best_candidate=ordered_rows[0]["candidate_id"],
            )

    leaderboard = pd.DataFrame(rows)
    leaderboard_csv_path = artifact_root / "candidate_leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv_path, index=False)

    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "runtime_device": runtime_device,
        "artifact_root": str(artifact_root),
        "dataset_prepared_roots": dict(config.dataset_prepared_roots),
        "screen_config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_folds": config.max_folds,
            "seed": config.seed,
            "device": config.device,
            "require_cuda": config.require_cuda,
            "train_sampling_policy": config.train_sampling_policy,
        },
        "candidate_order": [candidate.candidate_id for candidate in candidates],
        "leaderboard_csv_path": str(leaderboard_csv_path),
        "run_log_path": str(artifact_root / "run.log"),
        "progress_path": str(artifact_root / "progress.json"),
        "per_dataset_rankings": per_dataset_rankings,
        "candidate_summaries": candidate_summaries,
    }
    summary_path = artifact_root / "fusion_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = report_root / f"stage-i-public-fusion-screen-{config.run_id}.md"
    report_path.write_text(
        _render_public_fusion_screen_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(
        summary_path=str(summary_path),
        report_path=str(report_path),
        leaderboard_csv_path=str(leaderboard_csv_path),
    )
    LOGGER.info(
        "stage_i_public_fusion_screen finished run_id=%s summary_path=%s report_path=%s",
        config.run_id,
        summary_path,
        report_path,
    )
    return StageIPublicFusionScreenRunResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        summary_path=str(summary_path),
        leaderboard_csv_path=str(leaderboard_csv_path),
        report_path=str(report_path),
        summary=summary,
    )


def _extract_screen_score(dataset_id: str, summary: Mapping[str, object]) -> dict[str, object]:
    if dataset_id == "nasa_csm":
        combined = summary["objective"]["groups"]["combined"]
        return {
            "screen_metric": "macro_f1",
            "selection_score": float(combined["macro_f1"]),
            "secondary_score": float(combined["balanced_accuracy"]),
            "metrics": {
                "combined_macro_f1": float(combined["macro_f1"]),
                "combined_balanced_accuracy": float(combined["balanced_accuracy"]),
            },
        }
    subjective_groups = summary["subjective"]["groups"]
    mean_rmse = float(
        (
            float(subjective_groups["n_back"]["rmse"])
            + float(subjective_groups["heat_the_chair"]["rmse"])
        )
        / 2.0
    )
    mean_mae = float(
        (
            float(subjective_groups["n_back"]["mae"])
            + float(subjective_groups["heat_the_chair"]["mae"])
        )
        / 2.0
    )
    return {
        "screen_metric": "mean_rmse",
        "selection_score": mean_rmse,
        "secondary_score": mean_mae,
        "metrics": {
            "n_back_rmse": float(subjective_groups["n_back"]["rmse"]),
            "heat_the_chair_rmse": float(subjective_groups["heat_the_chair"]["rmse"]),
            "mean_rmse": mean_rmse,
            "mean_mae": mean_mae,
        },
    }


def _render_public_fusion_screen_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Public Fusion Screen - {summary['run_id']}",
        "",
        f"- generated_at_utc：`{summary['generated_at_utc']}`",
        f"- runtime_device：`{summary['runtime_device']}`",
        f"- artifact_root：`{summary['artifact_root']}`",
        f"- leaderboard_csv：`{summary['leaderboard_csv_path']}`",
    ]
    for dataset_id, rows in summary["per_dataset_rankings"].items():
        lines.extend(
            [
                "",
                f"## {dataset_id}",
                "",
                "| rank | candidate_id | screen_metric | selection_score | secondary_score |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
        )
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"| {index} | `{row['candidate_id']}` | `{row['screen_metric']}` | "
                f"{float(row['selection_score']):.6f} | {float(row['secondary_score']):.6f} |"
            )
    return "\n".join(lines)
