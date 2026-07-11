from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.evaluation.application_tasks.dingxin_consumer_models import (
    DingxinConsumerConfig,
    fit_dingxin_task_consumer,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_runtime import (
    run_dingxin_method_consumers,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    DingxinFoldConsumerTargets,
)
from chronaris.evaluation.application_tasks.dingxin_locked_consumer_run import (
    aggregate_dingxin_main_fold_metrics,
)
from chronaris.representation import FusionStreamBatch


def _targets():
    roles = {
        **{f"train_{index}": "train" for index in range(12)},
        **{f"validation_{index}": "validation" for index in range(6)},
        **{f"held_{index}": "held_out" for index in range(6)},
    }
    maneuver = {
        sample_id: index % 3
        for index, sample_id in enumerate(roles)
    }
    response = {
        sample_id: float(index) / 10.0
        for index, sample_id in enumerate(roles)
        if not sample_id.endswith("_5")
    }
    high = {
        sample_id: index % 2
        for index, sample_id in enumerate(response)
    }
    return DingxinFoldConsumerTargets(
        fold_id="fold_test",
        role_by_sample_id=roles,
        maneuver_class_by_sample_id=maneuver,
        response_value_by_sample_id=response,
        high_response_by_sample_id=high,
        target_source_sha256="a" * 64,
    )


def _output(role, sample_ids, *, seed=17):
    rng = np.random.default_rng(seed + len(sample_ids))
    sequence = torch.from_numpy(
        rng.normal(size=(len(sample_ids), 96, 64)).astype(np.float32)
    )
    return FusionStreamBatch(
        sample_ids=tuple(sample_ids),
        timestamps_s=torch.linspace(0, 30, 96).repeat(len(sample_ids), 1),
        sequence_embedding=sequence,
        valid_mask=torch.ones(len(sample_ids), 96, dtype=torch.bool),
        pooled_embedding=sequence.mean(dim=1),
        method_name="chronaris",
        fold_id="fold_test",
        checkpoint_sha256="b" * 64,
        source_sample_hashes=tuple(
            hashlib.sha256(sample_id.encode()).hexdigest()
            for sample_id in sample_ids
        ),
    )


def test_dingxin_task_consumers_fit_response_subset_and_predict_all():
    targets = _targets()
    train_ids = targets.sample_ids(role="train", task="maneuver_intensity_classification")
    output = _output("train", train_ids)
    config = DingxinConsumerConfig(n_kernels=84)

    for consumer_name in ("linear", "minirocket"):
        consumer = fit_dingxin_task_consumer(
            consumer_name=consumer_name,
            pooled_embedding=output.pooled_embedding.numpy(),
            sequence_embedding=output.sequence_embedding.numpy(),
            sample_ids=train_ids,
            targets=targets,
            config=config,
        )
        values = (
            output.pooled_embedding.numpy()
            if consumer_name == "linear"
            else output.sequence_embedding.numpy()
        )
        prediction = consumer.predict(values)

        assert prediction["maneuver_prediction"].shape == (12,)
        assert prediction["response_prediction"].shape == (12,)
        assert len(consumer.fit_response_sample_ids) == 11


def test_dingxin_method_consumer_resume_keeps_metrics_and_predictions(tmp_path):
    targets = _targets()
    outputs = {
        role: _output(
            role,
            tuple(
                sample_id
                for sample_id, sample_role in targets.role_by_sample_id.items()
                if sample_role == role
            ),
            seed=20 + index,
        )
        for index, role in enumerate(("train", "validation", "held_out"))
    }
    config = DingxinConsumerConfig(n_kernels=84)

    first = run_dingxin_method_consumers(
        method_name="chronaris",
        fold_id="fold_test",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        config=config,
        resume=True,
    )
    resumed = run_dingxin_method_consumers(
        method_name="chronaris",
        fold_id="fold_test",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        config=config,
        resume=True,
    )

    assert first.status == "completed"
    assert resumed.status == "resumed"
    assert len(first.metric_rows) == 56
    assert first.metric_rows == resumed.metric_rows
    assert first.manifest["prediction_sha256"] == resumed.manifest[
        "prediction_sha256"
    ]


def test_dingxin_formal_consumer_selects_validation_grid_and_marks_formal(tmp_path):
    targets = _targets()
    outputs = {
        role: _output(
            role,
            tuple(
                sample_id
                for sample_id, sample_role in targets.role_by_sample_id.items()
                if sample_role == role
            ),
            seed=40 + index,
        )
        for index, role in enumerate(("train", "validation", "held_out"))
    }
    config = DingxinConsumerConfig(n_kernels=84, tune_on_validation=True)

    result = run_dingxin_method_consumers(
        method_name="chronaris",
        fold_id="fold_test",
        outputs=outputs,
        targets=targets,
        output_root=tmp_path,
        config=config,
        evaluation_roles=("held_out",),
        smoke_only=False,
    )

    assert all(not row["smoke_only"] for row in result.metric_rows)
    assert result.manifest["smoke_only"] is False
    for consumer in result.manifest["consumer_files"].values():
        model = consumer["model"]
        assert model["hyperparameter_selection_role"] == "validation"
        assert model["selected_maneuver_c"] in config.classification_c_grid
        assert model["selected_response_alpha"] in config.regression_alpha_grid


def test_dingxin_main_summary_uses_view_folds_not_windows() -> None:
    rows = [
        {
            "seed": 17,
            "method": "chronaris",
            "task": "physiology_response_regression",
            "consumer": "linear",
            "metric": "rmse",
            "direction": "lower",
            "role": "held_out",
            "fold": f"leave_one_view_out__fold0{index}",
            "value": value,
        }
        for index, value in enumerate((2.0, 1.0, 3.0), start=1)
    ]

    summary = aggregate_dingxin_main_fold_metrics(rows)

    assert len(summary) == 1
    assert summary[0]["fold_count"] == 3
    assert summary[0]["mean"] == 2.0
    assert summary[0]["worst_fold_value"] == 3.0
    assert summary[0]["statistical_unit"] == "view_fold"
    assert summary[0]["window_level_p_value_reported"] is False
