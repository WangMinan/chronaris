"""Compact fixtures for fixed-data application-task tests."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


VEHICLE_FIELDS = (
    "BUS.self_acc_north",
    "BUS.self_acc_west",
    "BUS.self_acc_up",
    "BUS.self_roll",
    "BUS.target_acc_north",
    "BUS.self_acc_quality",
)


def make_records(
    *,
    views: Iterable[tuple[str, str, int]] = (
        ("sortie-a", "view-a1", 1),
        ("sortie-a", "view-a2", 2),
        ("sortie-b", "view-b1", 3),
    ),
    windows_per_view: int = 37,
) -> pd.DataFrame:
    rows = []
    for view_index, (sortie_id, view_id, pilot_id) in enumerate(views):
        for window_index in range(windows_per_view):
            vehicle_features = {}
            for field_index, field_name in enumerate(VEHICLE_FIELDS):
                base = (window_index + 1) * (field_index + 1) + view_index * 0.3
                vehicle_features[field_name] = {
                    "count": 5,
                    "mean": base,
                    "std": 0.2 + abs(base % 5),
                    "delta": ((-1) ** window_index) * (0.1 + abs(base % 7)),
                }
            physiology_features = {
                "eeg.AF3": {
                    "count": 5,
                    "mean": 0.5 * window_index + 0.03 * window_index**2 + view_index,
                    "std": 0.1,
                    "delta": 0.2,
                },
                "spo2.value": {
                    "count": 5,
                    "mean": 96.0 - 0.08 * window_index + 0.005 * window_index**2,
                    "std": 0.1,
                    "delta": -0.1,
                },
            }
            rows.append(
                {
                    "sample_id": f"{view_id}::window_{window_index:04d}",
                    "sortie_id": sortie_id,
                    "view_id": view_id,
                    "pilot_id": pilot_id,
                    "window_index": window_index,
                    "start_offset_ms": window_index * 5_000,
                    "end_offset_ms": (window_index + 1) * 5_000,
                    "raw_vehicle_stats": {"features": vehicle_features},
                    "raw_physiology_stats": {"features": physiology_features},
                    "physiology_point_count": 10,
                    "vehicle_point_count": 30,
                }
            )
    return pd.DataFrame(rows)


def vehicle_labels_by_sortie(records: pd.DataFrame) -> dict[str, dict[str, str]]:
    labels = {
        "BUS.self_acc_north": "[TSPI数据][载机平台系加速度][加速度_北向]",
        "BUS.self_acc_west": "[TSPI数据][载机平台系加速度][加速度_西向]",
        "BUS.self_acc_up": "[TSPI数据][载机平台系加速度][加速度_天向]",
        "BUS.self_roll": "[TSPI数据][载机横滚角]",
        "BUS.target_acc_north": "[TSPI数据][目标平台系加速度][加速度_北向]",
        "BUS.self_acc_quality": "[TSPI数据][载机平台系加速度][精度]",
    }
    return {str(sortie_id): dict(labels) for sortie_id in records["sortie_id"].unique()}
