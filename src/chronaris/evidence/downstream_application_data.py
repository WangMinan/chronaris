"""Reader-facing metric selections for the fixed-data downstream evidence pack."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


METHOD_ORDER = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)
METHOD_LABELS = {
    "physiology_only": "生理单流",
    "vehicle_only": "航电单流",
    "naive_time_sync": "朴素时间同步",
    "mult": "MulT",
    "contiformer": "ContiFormer",
    "chronaris": "Chronaris",
}
ABLATION_LABELS = {
    "chronaris_no_continuous_evolution": "去连续演化",
    "chronaris_no_physics": "去物理约束",
    "chronaris_no_causal_mask": "去因果掩码",
    "chronaris_single_scale_lag": "单尺度时延",
}
STRESS_LABELS = {
    "timestamp_jitter_ms": "时间戳抖动",
    "absolute_clock_offset_s": "时钟偏移",
    "absolute_clock_drift_ppm": "时钟漂移",
    "random_missing_rate": "随机缺失",
    "contiguous_gap_s": "连续缺失",
    "additional_physiology_lag_s": "生理额外时延",
    "snr_degradation_db": "信噪比退化",
}
TARGET_LABELS = {
    "relative_clock_offset_magnitude_s": "时钟偏移幅值",
    "primary_physiology_response_lag_s": "生理响应时延",
}


DINGXIN_PRIMARY = (
    {
        "task": "maneuver_intensity_classification",
        "consumer": "minirocket",
        "metric": "macro_f1",
        "title": "机动强度弱监督分类",
        "metric_label": "Macro-F1（越高越好）",
    },
    {
        "task": "high_physiology_response_classification",
        "consumer": "minirocket",
        "metric": "macro_auprc",
        "title": "高生理响应识别",
        "metric_label": "宏平均 AUPRC（越高越好）",
    },
    {
        "task": "physiology_response_regression",
        "consumer": "minirocket",
        "metric": "rmse",
        "title": "未来生理响应预测",
        "metric_label": "RMSE（越低越好）",
    },
)

SIMULATION_PRIMARY = (
    {
        "task": "simulated_future_workload_classification",
        "consumer": "minirocket",
        "metric": "macro_f1",
        "title": "仿真负荷状态分类",
        "metric_label": "Macro-F1（越高越好）",
    },
    {
        "task": "simulated_future_workload_regression",
        "consumer": "minirocket",
        "metric": "rmse",
        "title": "仿真负荷连续值预测",
        "metric_label": "RMSE（越低越好）",
    },
    {
        "task": "simulated_maneuver_state_segmentation",
        "consumer": "causal_tcn_duration",
        "metric": "frame_macro_f1",
        "title": "仿真机动状态分段",
        "metric_label": "逐时刻 Macro-F1（越高越好）",
    },
)


def select_metric_panel(
    frame: pd.DataFrame,
    specification: Mapping[str, str],
    *,
    value_column: str,
    role: str | None = None,
) -> pd.DataFrame:
    selected = frame.copy()
    for column in ("task", "consumer", "metric"):
        selected = selected[selected[column] == specification[column]]
    if role is not None and "role" in selected:
        selected = selected[selected["role"] == role]
    if selected.empty:
        raise ValueError(
            "primary evidence metric unavailable: "
            f"{specification['task']}/{specification['consumer']}/{specification['metric']}"
        )
    method_column = "method" if "method" in selected else "method_name"
    grouped = (
        selected.groupby(method_column, sort=False)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={method_column: "method"})
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["direction"] = (
        str(selected["direction"].iloc[0]) if "direction" in selected else "higher"
    )
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS)
    grouped["method_order"] = grouped["method"].map(
        {value: index for index, value in enumerate(METHOD_ORDER)}
    )
    return grouped.sort_values("method_order").reset_index(drop=True)


def build_primary_metric_table(
    frame: pd.DataFrame,
    specifications: Sequence[Mapping[str, str]],
    *,
    value_column: str,
    role: str | None = None,
) -> pd.DataFrame:
    rows = []
    for specification in specifications:
        panel = select_metric_panel(
            frame,
            specification,
            value_column=value_column,
            role=role,
        )
        for row in panel.to_dict("records"):
            rows.append(
                {
                    "task": specification["task"],
                    "task_label": specification["title"],
                    "consumer": specification["consumer"],
                    "metric": specification["metric"],
                    "metric_label": specification["metric_label"],
                    **row,
                }
            )
    return pd.DataFrame(rows)


def build_dingxin_transfer_delta_table(
    real_only: pd.DataFrame,
    synthetic_pretrain_adapted: pd.DataFrame,
    specifications: Sequence[Mapping[str, str]] = DINGXIN_PRIMARY,
) -> pd.DataFrame:
    """Build a matched, direction-aware Dingxin transfer comparison."""

    primary_keys = {
        (item["task"], item["consumer"], item["metric"])
        for item in specifications
    }

    def _select(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        required = {
            "seed",
            "method",
            "task",
            "consumer",
            "metric",
            "direction",
            "mean",
            "worst_fold_value",
            "fold_count",
            "available_fold_count",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                "Dingxin transfer comparison is missing columns: "
                + ", ".join(sorted(missing))
            )
        selected = frame[
            frame.apply(
                lambda row: (row["task"], row["consumer"], row["metric"])
                in primary_keys,
                axis=1,
            )
        ].copy()
        selected = selected[
            selected["fold_count"].eq(3)
            & selected["available_fold_count"].eq(3)
        ]
        keys = ["seed", "method", "task", "consumer", "metric", "direction"]
        if selected.duplicated(keys).any():
            raise ValueError("Dingxin transfer comparison keys are not unique")
        return selected[keys + ["mean", "worst_fold_value"]].rename(
            columns={
                "mean": value_name,
                "worst_fold_value": f"{value_name}_worst_fold",
            }
        )

    real = _select(real_only, "real_only_mean")
    transfer = _select(synthetic_pretrain_adapted, "transfer_mean")
    keys = ["seed", "method", "task", "consumer", "metric", "direction"]
    merged = real.merge(
        transfer,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), keys + ["_merge"]]
        raise ValueError(
            "Dingxin real-only and transfer primary keys differ: "
            + missing.to_json(orient="records", force_ascii=False)
        )
    merged = merged.drop(columns="_merge")
    merged["raw_delta_transfer_minus_real"] = (
        merged["transfer_mean"] - merged["real_only_mean"]
    )
    merged["normalized_improvement"] = merged[
        "raw_delta_transfer_minus_real"
    ].where(
        merged["direction"].eq("higher"),
        -merged["raw_delta_transfer_minus_real"],
    )
    merged["transfer_improved"] = merged["normalized_improvement"].gt(0.0)
    merged["method_label"] = merged["method"].map(METHOD_LABELS)
    task_labels = {item["task"]: item["title"] for item in specifications}
    metric_labels = {item["metric"]: item["metric_label"] for item in specifications}
    merged["task_label"] = merged["task"].map(task_labels)
    merged["metric_label"] = merged["metric"].map(metric_labels)
    merged["statistical_unit"] = "view_fold"
    merged["descriptive_only"] = True
    merged["method_order"] = merged["method"].map(
        {value: index for index, value in enumerate(METHOD_ORDER)}
    )
    return (
        merged.sort_values(["task", "method_order", "seed"], kind="stable")
        .drop(columns="method_order")
        .reset_index(drop=True)
    )


def build_stress_heatmap_table(slopes: pd.DataFrame) -> pd.DataFrame:
    primary_keys = {
        (item["task"], item["consumer"], item["metric"])
        for item in SIMULATION_PRIMARY
    }
    selected = slopes[
        slopes.apply(
            lambda row: (row["task"], row["consumer"], row["metric"])
            in primary_keys,
            axis=1,
        )
    ].copy()
    if selected.empty:
        raise ValueError("stress slopes do not contain the locked primary metrics")
    table = selected.pivot_table(
        index="method",
        columns="stress_factor",
        values="degradation_slope",
        aggfunc="mean",
    )
    table = table.reindex(
        index=[value for value in METHOD_ORDER if value in table.index],
        columns=[value for value in STRESS_LABELS if value in table.columns],
    )
    return table


def build_mechanism_mae_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["metric"] == "mae_s"].copy()
    if selected.empty:
        raise ValueError("mechanism recovery MAE rows are unavailable")
    return (
        selected.groupby(["target", "method"], sort=False)["value"]
        .mean()
        .reset_index()
    )


def build_ablation_primary_table(deltas: pd.DataFrame) -> pd.DataFrame:
    primary_keys = {
        (item["task"], item["consumer"], item["metric"])
        for item in SIMULATION_PRIMARY
    }
    selected = deltas[
        deltas.apply(
            lambda row: (row["task"], row["consumer"], row["metric"])
            in primary_keys,
            axis=1,
        )
    ].copy()
    if selected.empty:
        raise ValueError("ablation deltas do not contain the locked primary metrics")
    return (
        selected.groupby(
            ["task", "consumer", "metric", "ablation_method"], sort=False
        )["full_advantage_normalized"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .fillna({"std": 0.0})
    )
