from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronaris.evidence.downstream_application_data import (
    DINGXIN_PRIMARY,
    METHOD_ORDER,
    SIMULATION_PRIMARY,
    STRESS_LABELS,
    TARGET_LABELS,
)
from chronaris.evidence.downstream_application_pack import (
    DownstreamEvidencePackConfig,
    run_downstream_evidence_pack,
)


def test_downstream_evidence_pack_builds_separated_chinese_figures(tmp_path: Path) -> None:
    config = DownstreamEvidencePackConfig(
        compact_output_root=str(tmp_path),
        heavy_output_root=str(tmp_path / "heavy"),
        public_adapter_run_id=None,
        transfer_consumer_run_id=None,
    )
    run_ids = (
        config.dingxin_consumer_run_id,
        config.simulation_consumer_run_id,
        config.stress_consumer_run_id,
        config.mechanism_consumer_run_id,
        config.ablation_consumer_run_id,
        config.finetuning_run_id,
    )
    for run_id in run_ids:
        root = tmp_path / run_id
        root.mkdir(parents=True)
        (root / "evidence_manifest.json").write_text(
            json.dumps(
                {
                    "format": f"test.{run_id}.v1",
                    "run_id": run_id,
                    "status": "completed",
                }
            )
            + "\n"
        )
    dingxin_rows = []
    simulation_rows = []
    for spec_index, specification in enumerate(DINGXIN_PRIMARY):
        for method_index, method in enumerate(METHOD_ORDER):
            dingxin_rows.append(
                {
                    "seed": 17,
                    "method": method,
                    "task": specification["task"],
                    "consumer": specification["consumer"],
                    "metric": specification["metric"],
                    "direction": "lower" if specification["metric"] == "rmse" else "higher",
                    "mean": 0.2 + spec_index * 0.1 + method_index * 0.01,
                }
            )
    for spec_index, specification in enumerate(SIMULATION_PRIMARY):
        for method_index, method in enumerate(METHOD_ORDER):
            simulation_rows.append(
                {
                    "seed": 17,
                    "method": method,
                    "task": specification["task"],
                    "consumer": specification["consumer"],
                    "metric": specification["metric"],
                    "direction": "lower" if specification["metric"] == "rmse" else "higher",
                    "role": "held_out",
                    "value": 0.3 + spec_index * 0.1 + method_index * 0.01,
                }
            )
    pd.DataFrame(dingxin_rows).to_csv(
        tmp_path / config.dingxin_consumer_run_id / "main_view_fold_summary.csv",
        index=False,
    )
    pd.DataFrame(simulation_rows).to_csv(
        tmp_path / config.simulation_consumer_run_id / "metric_long.csv",
        index=False,
    )
    stress_rows = []
    for specification in SIMULATION_PRIMARY:
        for method_index, method in enumerate(METHOD_ORDER):
            for factor_index, factor in enumerate(STRESS_LABELS):
                stress_rows.append(
                    {
                        "seed": 17,
                        "method": method,
                        "task": specification["task"],
                        "consumer": specification["consumer"],
                        "metric": specification["metric"],
                        "stress_factor": factor,
                        "degradation_slope": -0.2 + method_index * 0.01 - factor_index * 0.001,
                    }
                )
    pd.DataFrame(stress_rows).to_csv(
        tmp_path / config.stress_consumer_run_id / "stress_slopes.csv",
        index=False,
    )
    mechanism_rows = [
        {
            "seed": 17,
            "method": method,
            "target": target,
            "metric": "mae_s",
            "value": 0.5 + method_index * 0.1,
        }
        for target in TARGET_LABELS
        for method_index, method in enumerate(METHOD_ORDER[2:])
    ]
    pd.DataFrame(mechanism_rows).to_csv(
        tmp_path / config.mechanism_consumer_run_id / "metric_long.csv",
        index=False,
    )
    variants = (
        "chronaris_no_continuous_evolution",
        "chronaris_no_physics",
        "chronaris_no_causal_mask",
        "chronaris_single_scale_lag",
    )
    ablation_rows = [
        {
            "seed": 17,
            "task": specification["task"],
            "consumer": specification["consumer"],
            "metric": specification["metric"],
            "ablation_method": variant,
            "full_advantage_normalized": 0.02 + variant_index * 0.01,
        }
        for specification in SIMULATION_PRIMARY
        for variant_index, variant in enumerate(variants)
    ]
    pd.DataFrame(ablation_rows).to_csv(
        tmp_path
        / config.ablation_consumer_run_id
        / "full_ablation_metric_delta.csv",
        index=False,
    )
    dingxin_prediction_rows = []
    for task in (
        "maneuver_intensity_classification",
        "physiology_response_regression",
    ):
        for index in range(1, 5):
            dingxin_prediction_rows.append(
                {
                    "seed": 17,
                    "method": "chronaris",
                    "consumer": "minirocket",
                    "fold": "leave_one_view_out__fold01",
                    "role": "held_out",
                    "sample_id": f"view_a::context_end_{index * 5:04d}",
                    "task": task,
                    "truth": float(index % 3),
                    "prediction": float((index + 1) % 3),
                }
            )
    dingxin_heavy = Path(config.heavy_output_root) / config.dingxin_consumer_run_id
    dingxin_heavy.mkdir(parents=True)
    pd.DataFrame(dingxin_prediction_rows).to_csv(
        dingxin_heavy / "prediction_rows.csv", index=False
    )
    simulation_prediction_rows = []
    simulation_manifest_rows = []
    for index, start in enumerate((30.0, 60.0, 90.0, 120.0)):
        sample_id = f"held_out_profile::context_{index}"
        simulation_prediction_rows.append(
            {
                "seed": 17,
                "method": "chronaris",
                "consumer": "minirocket",
                "role": "held_out",
                "sample_id": sample_id,
                "workload_class_true": index % 3,
                "workload_class_pred": (index + 1) % 3,
                "future_workload_true": 0.2 + index * 0.1,
                "future_workload_pred": 0.22 + index * 0.09,
            }
        )
        simulation_manifest_rows.append(
            {
                "sample_id": sample_id,
                "trajectory_id": "trajectory_001",
                "context_start_s": start,
            }
        )
    simulation_heavy = (
        Path(config.heavy_output_root) / config.simulation_consumer_run_id
    )
    simulation_heavy.mkdir(parents=True)
    pd.DataFrame(simulation_prediction_rows).to_csv(
        simulation_heavy / "workload_predictions.csv", index=False
    )
    representation_root = tmp_path / config.simulation_representation_run_id
    representation_root.mkdir(parents=True)
    pd.DataFrame(simulation_manifest_rows).to_csv(
        representation_root / "data_manifest.csv", index=False
    )

    result = run_downstream_evidence_pack(config)

    root = tmp_path / config.run_id
    assert result.status == "completed"
    assert result.figure_count == 7
    assert result.acceptance_pass_count == result.acceptance_check_count == 8
    assert len(list((root / "figures").glob("*.png"))) == 7
    assert "真实双流上的弱监督" in (root / "claim_boundary.md").read_text()
    evidence = pd.read_csv(root / "evidence_matrix.csv")
    assert evidence["data_layer"].nunique() >= 2
    assert not evidence["claim_boundary"].isna().any()
