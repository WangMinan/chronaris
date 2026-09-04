import pandas as pd
import pytest

from chronaris.evaluation.application_tasks import thesis_simulation_gates as gates


def _bypass_rows():
    return [
        {"task": task, "metric": metric, "consumer": consumer, "seed": seed,
         "method": method, "role": "held_out", "status": "available", "value": value}
        for task, metric, value in (
            ("simulated_future_workload_classification", "macro_f1", 0.8),
            ("simulated_future_workload_regression", "rmse", 1.0),
        )
        for consumer in ("linear", "minirocket")
        for seed in (17, 29, 43)
        for method in ("physiology_only", "vehicle_only", "chronaris")
    ]


def test_bypass_gate_requires_every_seed_and_keeps_failure(tmp_path, monkeypatch):
    path = tmp_path / "clean.csv"
    monkeypatch.setattr(gates, "CLEAN_METRICS", path)
    rows = _bypass_rows()
    pd.DataFrame(rows).to_csv(path, index=False)
    details, passed = gates._bypass_gate()
    assert passed and len(details) == 12
    rows[2]["value"] = 0.69
    pd.DataFrame(rows).to_csv(path, index=False)
    details, passed = gates._bypass_gate()
    assert not passed and len(details) == 12
    assert sum(not row["passed"] for row in details) == 1


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "consumer", "nan", "unavailable"])
def test_bypass_gate_fails_closed_on_incomplete_evidence(tmp_path, monkeypatch, defect):
    rows = _bypass_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "seed":
        rows[0]["seed"] = 99
    elif defect == "consumer":
        rows[0]["consumer"] = "unexpected"
    elif defect == "nan":
        rows[0]["value"] = float("nan")
    else:
        rows[0]["status"] = "unavailable"
    path = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(gates, "CLEAN_METRICS", path)
    with pytest.raises(RuntimeError, match="safe bypass comparison"):
        gates._bypass_gate()


def test_bypass_zero_baseline_requires_zero_error(tmp_path, monkeypatch):
    rows = _bypass_rows()
    for row in rows:
        if row["metric"] == "rmse":
            row["value"] = 0.0
    path = tmp_path / "clean.csv"
    monkeypatch.setattr(gates, "CLEAN_METRICS", path)
    pd.DataFrame(rows).to_csv(path, index=False)
    assert gates._bypass_gate()[1]
    rows[-1]["value"] = 0.01
    pd.DataFrame(rows).to_csv(path, index=False)
    details, passed = gates._bypass_gate()
    assert not passed
    assert any(row["degradation"] is None and not row["passed"] for row in details)


def test_bypass_accepts_exact_frozen_boundaries(tmp_path, monkeypatch):
    rows = _bypass_rows()
    for row in rows:
        if row["method"] == "chronaris":
            row["value"] = 0.75 if row["metric"] == "macro_f1" else 1.05
    path = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(gates, "CLEAN_METRICS", path)
    assert gates._bypass_gate()[1]


def test_continuous_gate_requires_two_seed_positive_recovery(tmp_path, monkeypatch):
    path = tmp_path / "full.csv"
    rows = []
    ablated = []
    for seed, gain in ((17, 0.2), (29, 0.1), (43, -0.1)):
        for scenario in ("a", "b"):
            rows.append(
                {
                    "seed": seed,
                    "scenario_id": scenario,
                    "target": "clock",
                    "method": "chronaris",
                    "metric": "mae_s",
                    "value": 1.0,
                }
            )
            ablated.append(
                {
                    "seed": seed,
                    "scenario_id": scenario,
                    "target": "clock",
                    "metric": "mae_s",
                    "value": 1.0 + gain,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(gates, "FULL_MECHANISM", path)

    summary, passed = gates._continuous_gate(ablated)

    assert passed is True
    assert sum(row["positive_gain"] for row in summary) == 2
    with pytest.raises(RuntimeError, match="coverage changed"):
        gates._continuous_gate(ablated[:-1])


def test_physical_gate_uses_available_translation_or_vertical_residual():
    rows = []
    for seed, full, ablated in (
        (17, 0.1, 0.2),
        (29, 0.2, 0.3),
        (43, 0.4, 0.3),
    ):
        rows.extend(
            (
                {
                    "seed": seed,
                    "variant": "full",
                    "component": "vehicle_rigid_body_vertical",
                    "raw_residual": full,
                    "count": 10,
                },
                {
                    "seed": seed,
                    "variant": "no_physics",
                    "component": "vehicle_rigid_body_vertical",
                    "raw_residual": ablated,
                    "count": 10,
                },
            )
        )

    assert gates._physical_gate(rows) is True
