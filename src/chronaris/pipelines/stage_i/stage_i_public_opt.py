"""Public-opt runners for Stage I UAB and NASA sequence assets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.stage_i_public_opt_data import (
    PUBLIC_OPT_DEFAULT_DATASET_ID,
    PUBLIC_OPT_PROFILE,
    build_stage_i_public_opt_feature_frame,
    get_public_opt_spec,
    normalize_public_opt_dataset_id,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_reference import (
    build_public_opt_reference_comparison,
    evaluate_public_opt_winning_margins,
    load_public_opt_prepared_dataset,
    validate_public_opt_prepared_dataset_contract,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_reporting import (
    render_stage_i_public_opt_report,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_sklearn import (
    resolve_head_feature_columns,
    run_public_opt_backend,
    validate_public_opt_config,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import open_stage_i_run_observer

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIPublicOptConfig:
    run_id: str
    prepared_artifact_root: str
    artifact_root: str = "docs/reports/assets/stage_i_public_opt"
    report_root: str = "docs/reports"
    dataset_id: str = PUBLIC_OPT_DEFAULT_DATASET_ID
    profile: str = PUBLIC_OPT_PROFILE
    seed: int = 42
    feature_profile: str = "full"
    head_catalog: str = "expanded"
    train_balance_policy: str = "class_weight_balanced"
    ensemble_policy: str = "none"
    prediction_aggregation_policy: str = "none"
    winner_margin_policy: str = "paper_gate"
    selected_subsets: tuple[str, ...] = ()
    reference_phase3_closure_summary_path: str | None = None
    reference_deep_comparison_summary_path: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicOptRunResult:
    run_id: str
    dataset_id: str
    profile: str
    artifact_root: str
    feature_frame_path: str
    predictions_path: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_public_opt(
    config: StageIPublicOptConfig,
) -> StageIPublicOptRunResult:
    canonical_dataset_id = normalize_public_opt_dataset_id(config.dataset_id)
    _validate_public_opt_config(config)
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_public_opt_sklearn",
        logger=LOGGER,
        initial_progress={
            "dataset_id": canonical_dataset_id,
            "prepared_artifact_root": str(Path(config.prepared_artifact_root)),
            "artifact_root": str(run_root),
            "head_catalog": config.head_catalog,
            "selected_subsets": list(config.selected_subsets),
        },
    ) as progress:
        LOGGER.info(
            "stage_i_public_opt start run_id=%s dataset_id=%s prepared_root=%s",
            config.run_id,
            canonical_dataset_id,
            config.prepared_artifact_root,
        )
        prepared = load_public_opt_prepared_dataset(config.prepared_artifact_root)
        validate_public_opt_prepared_dataset_contract(
            prepared,
            dataset_id=canonical_dataset_id,
        )
        if prepared["dataset_id"] != canonical_dataset_id:
            raise ValueError(
                "prepared dataset mismatch: expected "
                f"{canonical_dataset_id}, got {prepared['dataset_id']}"
            )
        spec = get_public_opt_spec(canonical_dataset_id)
        expected_profile = str(spec["profile"])
        prepared_profile = str(prepared["summary"]["profile"])
        if prepared_profile != config.profile:
            raise ValueError(
                f"prepared profile mismatch: expected {config.profile}, got {prepared_profile}"
            )
        if config.profile != expected_profile:
            raise ValueError(
                f"public opt only supports profile={expected_profile}, got {config.profile}"
            )

        feature_result = build_stage_i_public_opt_feature_frame(
            prepared["entries"],
            prepared["bundle"],
            dataset_id=canonical_dataset_id,
            profile=config.profile,
        )
        progress.update(
            "feature_frame_ready",
            feature_frame_shape=list(feature_result.feature_frame.shape),
            subsets=sorted(feature_result.feature_frame["subset_id"].astype(str).unique()),
        )
        head_feature_columns = resolve_head_feature_columns(
            feature_result=feature_result,
            feature_profile=config.feature_profile,
            head_catalog=config.head_catalog,
        )
        active_subset_order = config.selected_subsets or feature_result.subset_order
        report_root = Path(config.report_root)
        report_root.mkdir(parents=True, exist_ok=True)
        report_path = report_root / f"stage-i-public-opt-{config.run_id}.md"
        feature_frame_path = run_root / "public_opt_feature_frame.parquet"
        predictions_path = run_root / "public_opt_predictions.csv"
        summary_path = run_root / "public_opt_summary.json"
        progress.update(
            "output_paths_ready",
            feature_frame_path=str(feature_frame_path),
            predictions_path=str(predictions_path),
            summary_path=str(summary_path),
            report_path=str(report_path),
        )

        predictions, subset_results = run_public_opt_backend(
            feature_result=feature_result,
            head_feature_columns=head_feature_columns,
            head_catalog=config.head_catalog,
            train_balance_policy=config.train_balance_policy,
            ensemble_policy=config.ensemble_policy,
            prediction_aggregation_policy=config.prediction_aggregation_policy,
            selected_evaluation_groups=active_subset_order,
            progress=progress,
        )
        reference_comparison = build_public_opt_reference_comparison(
            dataset_id=canonical_dataset_id,
            track=feature_result.track,
            subset_results=subset_results,
            phase3_closure_summary_path=config.reference_phase3_closure_summary_path,
            deep_comparison_summary_path=config.reference_deep_comparison_summary_path,
        )
        summary = {
            "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
            "run_id": config.run_id,
            "dataset_id": canonical_dataset_id,
            "profile": config.profile,
            "feature_profile": config.feature_profile,
            "head_catalog": config.head_catalog,
            "train_balance_policy": config.train_balance_policy,
            "ensemble_policy": config.ensemble_policy,
            "prediction_aggregation_policy": config.prediction_aggregation_policy,
            "winner_margin_policy": config.winner_margin_policy,
            "selected_subsets": list(active_subset_order),
            "track": feature_result.track,
            "task_type": feature_result.task_type,
            "subset_order": list(active_subset_order),
            "evaluation_groups": {
                key: list(value)
                for key, value in feature_result.evaluation_groups.items()
            },
            "heads": list(head_feature_columns),
            "feature_group_sizes": {
                key: len(value) for key, value in feature_result.feature_groups.items()
            },
            "artifact_root": str(run_root),
            "prepared_artifact_root": str(Path(config.prepared_artifact_root)),
            "run_log_path": str(run_root / "run.log"),
            "progress_path": str(run_root / "progress.json"),
            "subset_results": subset_results,
            "reference_comparison": reference_comparison,
        }
        winning_margin_vs_deep, needs_deep_rerun = evaluate_public_opt_winning_margins(
            track=feature_result.track,
            reference_comparison=reference_comparison,
            policy=config.winner_margin_policy,
        )
        summary["winning_margin_vs_deep"] = winning_margin_vs_deep
        summary["needs_deep_rerun"] = needs_deep_rerun

        feature_result.feature_frame.to_parquet(feature_frame_path, index=False)
        predictions.to_csv(predictions_path, index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            render_stage_i_public_opt_report(
                summary=summary,
                feature_frame=feature_result.feature_frame,
            )
            + "\n",
            encoding="utf-8",
        )
        progress.finish(
            summary_path=str(summary_path),
            report_path=str(report_path),
            predictions_path=str(predictions_path),
        )
        LOGGER.info(
            "stage_i_public_opt finished run_id=%s summary_path=%s report_path=%s",
            config.run_id,
            summary_path,
            report_path,
        )
        return StageIPublicOptRunResult(
            run_id=config.run_id,
            dataset_id=canonical_dataset_id,
            profile=config.profile,
            artifact_root=str(run_root),
            feature_frame_path=str(feature_frame_path),
            predictions_path=str(predictions_path),
            summary_path=str(summary_path),
            report_path=str(report_path),
            summary=summary,
        )


def _validate_public_opt_config(config: StageIPublicOptConfig) -> None:
    validate_public_opt_config(
        feature_profile=config.feature_profile,
        head_catalog=config.head_catalog,
        train_balance_policy=config.train_balance_policy,
        ensemble_policy=config.ensemble_policy,
        prediction_aggregation_policy=config.prediction_aggregation_policy,
        winner_margin_policy=config.winner_margin_policy,
    )
