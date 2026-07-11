"""Formal three-seed Dingxin consumers over the locked five-fold representations."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
)
from chronaris.evaluation.application_tasks.application_metrics import (
    compute_fusion_gain_rows,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_models import (
    DingxinConsumerConfig,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_runtime import (
    run_dingxin_method_consumers,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    load_dingxin_nested_fold_consumer_targets,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DEFAULT_FOLDS,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    LOCKED_SEEDS,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.representation import load_fusion_stream_batch


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_locked_consumers")
LOGGER.addHandler(logging.NullHandler())
MAIN_VIEW_FOLDS = DEFAULT_FOLDS[:3]


@dataclass(frozen=True, slots=True)
class DingxinLockedConsumerConfig:
    run_id: str = "2026-07-12_dingxin-locked-consumers"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    representation_run_id: str = "2026-07-12_dingxin-locked-representations"
    nested_target_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
    )
    seeds: tuple[int, ...] = LOCKED_SEEDS
    fold_ids: tuple[str, ...] = DEFAULT_FOLDS
    minirocket_kernels: int = 10_000
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinLockedConsumerResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    method_fold_seed_count: int
    metric_count: int
    fusion_gain_count: int
    main_summary_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_locked_consumers(config: DingxinLockedConsumerConfig):
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    representation_root = Path(config.heavy_output_root) / config.representation_run_id
    representation_compact = (
        Path(config.compact_output_root) / config.representation_run_id
    )
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(
        (representation_compact / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    if evidence.get("status") != "completed":
        raise ValueError("Dingxin locked consumers require completed representations")
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_locked_consumers",
        logger=LOGGER,
        initial_progress={
            "seeds": list(config.seeds),
            "folds": list(config.fold_ids),
            "outer_test_metrics_opened": True,
            "window_level_significance_allowed": False,
        },
    ) as progress:
        result_rows = []
        metric_rows = []
        prediction_rows = []
        resource_rows = []
        for seed in config.seeds:
            consumer_config = DingxinConsumerConfig(
                n_kernels=config.minirocket_kernels,
                random_state=seed,
                tune_on_validation=True,
            )
            for fold_id in config.fold_ids:
                targets = load_dingxin_nested_fold_consumer_targets(
                    fold_id=fold_id,
                    nested_target_path=config.nested_target_path,
                )
                for method in APPLICATION_METHODS:
                    outputs = _load_outputs(
                        representation_root,
                        seed=seed,
                        fold_id=fold_id,
                        method=method,
                    )
                    result = run_dingxin_method_consumers(
                        method_name=method,
                        fold_id=fold_id,
                        outputs=outputs,
                        targets=targets,
                        output_root=heavy_root / "consumers" / f"seed_{seed}",
                        config=consumer_config,
                        evaluation_roles=("validation", "held_out"),
                        smoke_only=False,
                        resume=config.resume,
                    )
                    result_rows.append(
                        {
                            "seed": seed,
                            "fold_id": fold_id,
                            "fold_family": (
                                "leave_one_view_out_main"
                                if fold_id in MAIN_VIEW_FOLDS
                                else "leave_one_sortie_out_auxiliary"
                            ),
                            "method_name": method,
                            "status": result.status,
                            "protocol_sha256": result.protocol_sha256,
                            "hyperparameter_selection_role": "validation",
                            "outer_test_metrics_opened": True,
                        }
                    )
                    metric_rows.extend(dict(row) for row in result.metric_rows)
                    prediction_rows.extend(
                        {"seed": seed, **dict(row)}
                        for row in result.prediction_rows
                    )
                    resource_rows.extend(
                        {"seed": seed, **dict(row)}
                        for row in result.resource_rows
                    )
                    progress.update(
                        "dingxin_locked_method_consumer_complete",
                        seed=seed,
                        fold_id=fold_id,
                        method_name=method,
                        status=result.status,
                    )
        fusion_gain_rows = compute_fusion_gain_rows(
            metric_rows,
            fusion_methods=("naive_time_sync", "mult", "contiformer", "chronaris"),
        )
        main_summary_rows = aggregate_dingxin_main_fold_metrics(metric_rows)
        acceptance = _acceptance_rows(
            config=config,
            results=result_rows,
            metrics=metric_rows,
            summary=main_summary_rows,
        )
        status = "completed" if all(row["passed"] for row in acceptance) else "partial"
        paths = _write_outputs(
            compact_root=compact_root,
            heavy_root=heavy_root,
            config=config,
            result_rows=result_rows,
            metric_rows=metric_rows,
            fusion_gain_rows=fusion_gain_rows,
            main_summary_rows=main_summary_rows,
            resource_rows=resource_rows,
            prediction_rows=prediction_rows,
            acceptance=acceptance,
            status=status,
        )
        progress.finish(
            status=status,
            method_fold_seed_count=len(result_rows),
            metric_count=len(metric_rows),
            acceptance_pass_count=sum(row["passed"] for row in acceptance),
            acceptance_check_count=len(acceptance),
        )
    return DingxinLockedConsumerResult(
        run_id=config.run_id,
        status=status,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        method_fold_seed_count=len(result_rows),
        metric_count=len(metric_rows),
        fusion_gain_count=len(fusion_gain_rows),
        main_summary_count=len(main_summary_rows),
        acceptance_pass_count=sum(row["passed"] for row in acceptance),
        acceptance_check_count=len(acceptance),
        report_path=str(paths["report"]),
        evidence_manifest_path=str(paths["evidence"]),
    )


def aggregate_dingxin_main_fold_metrics(metric_rows):
    frame = pd.DataFrame(metric_rows)
    selected = frame[
        (frame["role"] == "held_out")
        & (frame["fold"].isin(MAIN_VIEW_FOLDS))
    ].copy()
    rows = []
    keys = ["seed", "method", "task", "consumer", "metric", "direction"]
    for key, group in selected.groupby(keys, sort=True, dropna=False):
        values = group["value"].dropna().astype(float).to_numpy()
        if len(values) == 0:
            mean = std = minimum = maximum = worst = None
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            worst = minimum if key[-1] == "higher" else maximum
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "fold_count": len(group),
                "available_fold_count": len(values),
                "mean": mean,
                "std": std,
                "minimum": minimum,
                "maximum": maximum,
                "worst_fold_value": worst,
                "statistical_unit": "view_fold",
                "window_level_p_value_reported": False,
            }
        )
    return tuple(rows)


def _load_outputs(root, *, seed, fold_id, method):
    base = root / "representations" / f"seed_{seed}" / fold_id / method / fold_id
    return {
        role: load_fusion_stream_batch(base / role)
        for role in ("train", "validation", "held_out")
    }


def _acceptance_rows(*, config, results, metrics, summary):
    expected = len(config.seeds) * len(config.fold_ids) * len(APPLICATION_METHODS)
    outer = [row for row in metrics if row["role"] == "held_out"]
    return (
        _check("all_method_fold_seed_consumers", len(results) == expected, len(results), expected),
        _check("all_consumers_complete", all(row["status"] in {"completed", "resumed"} for row in results), [row["status"] for row in results], "completed_or_resumed"),
        _check("validation_hyperparameter_selection", all(row["hyperparameter_selection_role"] == "validation" for row in results), {row["hyperparameter_selection_role"] for row in results}, {"validation"}),
        _check("formal_metric_scope", bool(metrics) and all(not row["smoke_only"] for row in metrics), len(metrics), ">0 formal rows"),
        _check("outer_test_metrics_available", bool(outer), len(outer), ">0"),
        _check("main_summary_uses_three_view_folds", bool(summary) and all(row["fold_count"] == 3 and row["statistical_unit"] == "view_fold" for row in summary), len(summary), ">0 with 3 folds"),
        _check("no_window_level_p_values", all(not row["window_level_p_value_reported"] for row in summary), False, False),
    )


def _write_outputs(**values):
    root = values["compact_root"]
    paths = {
        "results": root / "consumer_inventory.csv",
        "metrics": root / "metric_long.csv",
        "gains": root / "fusion_gain.csv",
        "summary": root / "main_view_fold_summary.csv",
        "resources": root / "resource_metrics.csv",
        "acceptance": root / "acceptance.csv",
        "protocol": root / "protocol.json",
        "report": root / "report.md",
        "claim": root / "claim_boundary.md",
        "resume": root / "resume_command.txt",
        "evidence": root / "evidence_manifest.json",
    }
    for key, rows in (
        ("results", values["result_rows"]),
        ("metrics", values["metric_rows"]),
        ("gains", values["fusion_gain_rows"]),
        ("summary", values["main_summary_rows"]),
        ("resources", values["resource_rows"]),
        ("acceptance", values["acceptance"]),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    prediction_path = values["heavy_root"] / "prediction_rows.csv"
    pd.DataFrame(values["prediction_rows"]).to_csv(prediction_path, index=False)
    _write_json(paths["protocol"], {
        "format": "chronaris.dingxin_locked_consumer_protocol.v1",
        "config": asdict(values["config"]),
        "fit_role": "train",
        "hyperparameter_selection_role": "validation",
        "evaluation_roles": ["validation", "held_out"],
        "main_statistical_unit": "view_fold",
        "window_level_significance_allowed": False,
    })
    passed = sum(row["passed"] for row in values["acceptance"])
    paths["report"].write_text("\n".join((
        "# 鼎新三随机种子五折锁定下游评估",
        "",
        f"状态：{values['status']}；验收 {passed}/{len(values['acceptance'])}。",
        f"完成 {len(values['result_rows'])} 个方法—折—随机种子 consumer 和 {len(values['metric_rows'])} 条指标。",
        "主协议只以三个留一视图折汇总均值、标准差、范围和最差折；不报告窗口级显著性。",
        "",
    )), encoding="utf-8")
    paths["claim"].write_text(
        "# 结论边界\n\n鼎新结果是现有真实双流上的弱监督任务证据，不等同于人工工作负荷或专家机动科目真值。\n",
        encoding="utf-8",
    )
    paths["resume"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_locked_consumers.py "
        f"--run-id {values['config'].run_id} "
        f"--representation-run-id {values['config'].representation_run_id} "
        f"--minirocket-kernels {values['config'].minirocket_kernels} --resume\n",
        encoding="utf-8",
    )
    _write_json(paths["evidence"], {
        "format": "chronaris.dingxin_locked_consumer_evidence.v1",
        "run_id": values["config"].run_id,
        "status": values["status"],
        "method_fold_seed_count": len(values["result_rows"]),
        "metric_count": len(values["metric_rows"]),
        "fusion_gain_count": len(values["fusion_gain_rows"]),
        "main_summary_count": len(values["main_summary_rows"]),
        "acceptance_pass_count": passed,
        "acceptance_check_count": len(values["acceptance"]),
        "heavy_run_root": str(values["heavy_root"]),
        "prediction_path": str(prediction_path),
        "output_paths": {key: str(path) for key, path in paths.items()},
    })
    return paths


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
