import pandas as pd

from chronaris.evaluation.application_tasks.thesis_final_reporting import (
    build_application_target_rows,
    build_best_single_gain_rows,
    paired_subject_bootstrap_rows,
)


def test_outer_gain_and_group_bootstrap_use_positive_chronaris_direction():
    metrics = []
    predictions = []
    for fold, group in (
        ("fold1", "s1"),
        ("fold2", "s2"),
        ("fold3", "s3"),
        ("fold4", "s4"),
        ("fold5", "s5"),
    ):
        for seed in (17, 29, 43):
            for method, value in (
                ("chronaris", 0.8),
                ("physiology_only", 0.6),
                ("vehicle_only", 0.7),
            ):
                metrics.append(
                    {
                        "task": "clare_cognitive_load",
                        "fold": fold,
                        "seed": seed,
                        "scenario": "full",
                        "metric": "macro_f1",
                        "direction": "higher",
                        "method": method,
                        "value": value,
                    }
                )
                predictions.extend(
                    {
                        "task": "clare_cognitive_load",
                        "target_kind": "classification",
                        "fold": fold,
                        "seed": seed,
                        "scenario": "full",
                        "method": method,
                        "sample_id": f"{group}-{truth}",
                        "group_id": group,
                        "truth": truth,
                        "prediction": truth if method == "chronaris" else 0,
                    }
                    for truth in (0, 1)
                )

    gains = build_best_single_gain_rows(pd.DataFrame(metrics))
    target = build_application_target_rows(gains)
    bootstrap = paired_subject_bootstrap_rows(predictions, repetitions=20)

    assert target[0]["passed"] is True
    assert target[0]["positive_outer_fold_fraction"] == 1.0
    assert all(row["gain_positive_favors_chronaris"] > 0 for row in bootstrap)
