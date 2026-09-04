"""Aggregate frozen outer results without treating cross-validation folds as IID."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from chronaris.evaluation.dingxin.simple_downstream_figures import (
    configure_chinese_matplotlib,
)


COMPLEMENTARY_TASK_METRICS = {
    ("cogpilot_event_response", "rmse"),
    ("clare_cognitive_load", "macro_f1"),
    ("clare_cognitive_load", "rmse"),
}


def build_thesis_outer_summary(
    *,
    native_compact_root: str | Path,
    native_heavy_root: str | Path,
    dingxin_compact_root: str | Path,
    output_root: str | Path,
    bootstrap_repetitions: int = 2_000,
):
    native_compact_root = Path(native_compact_root)
    native_heavy_root = Path(native_heavy_root)
    dingxin_compact_root = Path(dingxin_compact_root)
    _require_complete(
        native_compact_root / "outer_results.json",
        "completed_for_requested_tasks",
    )
    _require_complete(dingxin_compact_root / "outer_results.json", "completed")
    native = pd.read_csv(native_compact_root / "metric_long.csv")
    dingxin = pd.read_csv(dingxin_compact_root / "metric_long.csv")
    native_summary = summarize_metric_panel(
        native,
        group_columns=("task", "scenario", "metric", "direction", "method"),
    )
    dingxin_summary = summarize_metric_panel(
        dingxin,
        group_columns=("fold_kind", "task", "metric", "direction", "method"),
    )
    gains = build_best_single_gain_rows(native)
    application = build_application_target_rows(gains)
    predictions = _native_predictions(native_heavy_root)
    bootstrap = paired_subject_bootstrap_rows(
        predictions,
        repetitions=bootstrap_repetitions,
    )
    audit = {
        "format": "chronaris.thesis_outer_summary.v1",
        "protocol_version": "v3.2.2",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip(),
        "evaluation_code_sha256": _sha256_file(Path(__file__)),
        "source_files": {
            "native_state_sha256": _sha256_file(
                native_compact_root / "outer_results.json"
            ),
            "native_metrics_sha256": _sha256_file(
                native_compact_root / "metric_long.csv"
            ),
            "native_prediction_set_sha256": _file_set_sha256(
                sorted((native_heavy_root / "outer_units").glob("**/result.json")),
                relative_to=native_heavy_root,
            ),
            "dingxin_state_sha256": _sha256_file(
                dingxin_compact_root / "outer_results.json"
            ),
            "dingxin_metrics_sha256": _sha256_file(
                dingxin_compact_root / "metric_long.csv"
            ),
        },
        "application_enhancement_target_passed": any(
            row["passed"] for row in application
        ),
        "application_target_rows": application,
        "bootstrap_repetitions": bootstrap_repetitions,
        "application_figure": "application_enhancement.png",
        "statistical_scope": (
            "paired subject bootstrap; no IID fold t-test; Dingxin is descriptive "
            "because only two sorties are available"
        ),
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "native_metric_summary.csv", native_summary)
    _write_csv(root / "native_best_single_gain.csv", gains)
    _write_csv(root / "native_subject_bootstrap.csv", bootstrap)
    _write_csv(root / "dingxin_metric_summary.csv", dingxin_summary)
    _plot_application_target(root / audit["application_figure"], application)
    _atomic_json(root / "summary.json", audit)
    (root / "report.md").write_text(_report(audit), encoding="utf-8")
    return audit


def summarize_metric_panel(frame, *, group_columns):
    rows = []
    finite = frame[frame["value"].notna()].copy()
    for key, values in finite.groupby(list(group_columns), sort=True):
        direction = values["direction"].iloc[0]
        numbers = values["value"].to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "count": len(numbers),
                "median": float(np.median(numbers)),
                "q25": float(np.quantile(numbers, 0.25)),
                "q75": float(np.quantile(numbers, 0.75)),
                "worst": float(
                    np.min(numbers) if direction == "higher" else np.max(numbers)
                ),
            }
        )
    return rows


def build_best_single_gain_rows(frame):
    index = ("task", "fold", "seed", "scenario", "metric", "direction")
    usable = frame[frame["method"].isin((
        "chronaris",
        "physiology_only",
        "vehicle_only",
    ))]
    pivot = usable.pivot(index=list(index), columns="method", values="value").dropna()
    rows = []
    for key, values in pivot.iterrows():
        metadata = dict(zip(index, key, strict=True))
        singles = {
            "physiology_only": float(values["physiology_only"]),
            "vehicle_only": float(values["vehicle_only"]),
        }
        if metadata["direction"] == "higher":
            best_method = max(singles, key=singles.get)
            gain = float(values["chronaris"] - singles[best_method])
        else:
            best_method = min(singles, key=singles.get)
            gain = float(singles[best_method] - values["chronaris"])
        rows.append(
            {
                **metadata,
                "chronaris_value": float(values["chronaris"]),
                "best_single_method": best_method,
                "best_single_value": singles[best_method],
                "gain_positive_favors_chronaris": gain,
            }
        )
    return rows


def build_application_target_rows(gain_rows):
    frame = pd.DataFrame(gain_rows)
    frame = frame[
        (frame["scenario"] == "full")
        & frame[["task", "metric"]].apply(tuple, axis=1).isin(
            COMPLEMENTARY_TASK_METRICS
        )
    ]
    rows = []
    for (task, metric), values in frame.groupby(["task", "metric"], sort=True):
        fold_gains = values.groupby("fold")[
            "gain_positive_favors_chronaris"
        ].median()
        median_gain = float(values["gain_positive_favors_chronaris"].median())
        positive_fraction = float((fold_gains > 0).mean())
        rows.append(
            {
                "task": task,
                "metric": metric,
                "fold_count": len(fold_gains),
                "fold_seed_count": len(values),
                "median_gain": median_gain,
                "positive_outer_fold_fraction": positive_fraction,
                "passed": (
                    len(fold_gains) == 5
                    and median_gain > 0
                    and positive_fraction >= 0.60
                ),
            }
        )
    return rows


def paired_subject_bootstrap_rows(predictions, *, repetitions=2_000):
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    frame = pd.DataFrame(predictions)
    frame = frame[frame["scenario"] == "full"]
    rows = []
    for (task, target_kind, seed), values in frame.groupby(
        ["task", "target_kind", "seed"], sort=True
    ):
        metric = "macro_f1" if target_kind == "classification" else "rmse"
        for comparison in ("physiology_only", "vehicle_only"):
            pivot = values[values["method"].isin(("chronaris", comparison))].pivot(
                index=["sample_id", "group_id", "truth"],
                columns="method",
                values="prediction",
            ).dropna()
            if pivot.empty:
                continue
            groups = tuple(sorted(pivot.index.get_level_values("group_id").unique()))
            truth = pivot.index.get_level_values("truth").to_numpy()
            labels = tuple(sorted(set(truth.astype(int))))
            point = _prediction_gain(
                truth,
                pivot["chronaris"].to_numpy(),
                pivot[comparison].to_numpy(),
                target_kind=target_kind,
                labels=labels,
            )
            rng = np.random.default_rng(17_000 + int(seed))
            samples = []
            for _ in range(repetitions):
                selected = rng.choice(groups, size=len(groups), replace=True)
                indices = np.concatenate(
                    [
                        np.flatnonzero(
                            pivot.index.get_level_values("group_id") == group
                        )
                        for group in selected
                    ]
                )
                samples.append(
                    _prediction_gain(
                        truth[indices],
                        pivot["chronaris"].to_numpy()[indices],
                        pivot[comparison].to_numpy()[indices],
                        target_kind=target_kind,
                        labels=labels,
                    )
                )
            rows.append(
                {
                    "task": task,
                    "target_kind": target_kind,
                    "metric": metric,
                    "seed": int(seed),
                    "comparison_method": comparison,
                    "independent_subject_count": len(groups),
                    "gain_positive_favors_chronaris": point,
                    "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                }
            )
    return rows


def _prediction_gain(truth, reference, comparison, *, target_kind, labels):
    if target_kind == "classification":
        return float(
            f1_score(truth, reference, labels=labels, average="macro", zero_division=0)
            - f1_score(
                truth,
                comparison,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    return float(
        mean_squared_error(truth, comparison) ** 0.5
        - mean_squared_error(truth, reference) ** 0.5
    )


def _native_predictions(root):
    rows = []
    for path in sorted((root / "outer_units").glob("**/result.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["prediction_rows"])
    return rows


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_set_sha256(paths, *, relative_to):
    if not paths:
        raise RuntimeError("native outer prediction set is empty")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(relative_to).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_complete(path, key):
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get(key) is not True:
        raise RuntimeError(f"outer evidence incomplete: {path}")


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"cannot write empty evidence: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _report(audit):
    task_labels = {
        "cogpilot_event_response": "CogPilot 飞行事件—生理响应回归",
        "clare_cognitive_load": "CLARE 认知负荷评估",
    }
    metric_labels = {"macro_f1": "宏平均 F1", "rmse": "均方根误差"}
    lines = [
        "# 论文主线外层结果汇总",
        "",
        "公开数据按受试者分组评价，并以受试者为单位给出配对自助法区间；五个共享训练集的交叉验证折不作为独立样本进行 t 检验。鼎新仅作协议冻结后的分组确认，不作显著性结论。",
        "",
        "## 双流互补信息应用增强目标",
        "",
        "下图展示 Chronaris 连续语义融合模型相对每个外层结果中较优单流的中位增量，以及五个受试者分组外层折中增量为正的比例，用于判断真实双流互补信息是否形成稳定收益。",
        "",
        f"![真实双流互补任务增量]({audit['application_figure']})",
        "",
        "| 任务 | 指标 | 中位增量 | 正向外层折比例 | 结果 |",
        "|---|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {task_labels[row['task']]} | {metric_labels[row['metric']]} | "
        f"{row['median_gain']:.4f} | "
        f"{row['positive_outer_fold_fraction']:.1%} | "
        f"{'达到' if row['passed'] else '未达到'} |"
        for row in audit["application_target_rows"]
    )
    lines.extend(
        (
            "",
            "总体判断："
            + (
                "至少一个真实双流互补任务达到增强目标。"
                if audit["application_enhancement_target_passed"]
                else "应用增强目标未达到；协议与机制结论不因此失效。"
            ),
            "",
        )
    )
    return "\n".join(lines)


def _plot_application_target(path, rows):
    import matplotlib.pyplot as plt

    configure_chinese_matplotlib()
    labels = [
        {
            ("cogpilot_event_response", "rmse"): "CogPilot\n响应回归均方根误差",
            ("clare_cognitive_load", "macro_f1"): "CLARE\n认知负荷宏平均 F1",
            ("clare_cognitive_load", "rmse"): "CLARE\n认知负荷均方根误差",
        }[(row["task"], row["metric"])]
        for row in rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(labels, [row["median_gain"] for row in rows], color="#2878B5")
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_ylabel("增量（正值表示 Chronaris 更优）")
    axes[0].set_title("相对较优单流的中位增量")
    axes[1].bar(
        labels,
        [row["positive_outer_fold_fraction"] for row in rows],
        color="#2878B5",
    )
    axes[1].axhline(0.60, color="#444444", linestyle="--", linewidth=0.9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("正向外层折比例")
    axes[1].set_title("跨受试者外层折方向一致性")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)
