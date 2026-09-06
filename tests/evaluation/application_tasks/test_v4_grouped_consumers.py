from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_squared_error

from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier, fit_regressor, grouped_score
from chronaris.evaluation.application_tasks.v4_grouped_consumers import (
    fit_native_consumers, evaluate_native_consumers, regression_metrics, run_native_method_consumers)
from chronaris.representation import FusionStreamBatch
from chronaris.representation.contracts import QUERY_POINT_COUNT


def _output(ids, values, *, empty_last=False):
    sequence = torch.tensor(values, dtype=torch.float32)
    if sequence.ndim == 2:
        sequence = sequence[:, None, :].repeat(1, QUERY_POINT_COUNT, 1)
    mask = torch.ones(len(ids), QUERY_POINT_COUNT, dtype=torch.bool)
    if empty_last:
        sequence[-1] = 0
        mask[-1] = False
    return FusionStreamBatch(tuple(ids), torch.linspace(0., 30., QUERY_POINT_COUNT).repeat(len(ids), 1), sequence, mask,
        (sequence * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1),
        "chronaris", "test", "a" * 64, tuple("b" * 64 for _ in ids))


def test_weighted_fit_matches_deduplicated_views_and_group_scores_ignore_window_counts():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(12, 4)), np.arange(12) % 3
    repeats = np.array([3, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 1])
    duplicated = np.repeat(np.arange(12), repeats)
    weights = 1. / repeats[duplicated]
    for fit, kwargs in ((fit_classifier, dict(c_values=(1.,), random_state=17)),
                        (fit_regressor, dict(alpha_values=(1.,)))):
        model, _ = fit(x, y, None, None, scaler_with_mean=True, **kwargs)
        duplicate, _ = fit(x[duplicated], y[duplicated], None, None, scaler_with_mean=True,
                           train_sample_weight=weights, **kwargs)
        np.testing.assert_allclose(model[0].mean_, duplicate[0].mean_, atol=1e-14)
        np.testing.assert_allclose(model[-1].coef_, duplicate[-1].coef_, atol=1e-9)
    metric = lambda a, b: mean_squared_error(a, b) ** .5
    assert grouped_score(metric, np.zeros(101), np.r_[np.ones(100), 10.], np.array(["a"] * 100 + ["b"])) == 5.5


def _public_inputs():
    rng = np.random.default_rng(29)
    ids = tuple(f"s{i}" for i in range(18))
    outputs = {"train": _output(ids[:12], rng.normal(size=(12, QUERY_POINT_COUNT, 64))),
               "validation": _output(ids[12:], rng.normal(size=(6, QUERY_POINT_COUNT, 64)), empty_last=True)}
    values = {"classify": torch.arange(18) % 3, "fields": torch.arange(36.).reshape(18, 2)}
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
    masks["fields"][::2, 0] = False
    masks["fields"][1::2, 1] = False
    values["fields"][~masks["fields"]] = float("nan")
    tasks = (ApplicationTaskDefinition("classify", "classification", 3), ApplicationTaskDefinition("fields", "regression", 2))
    targets = ApplicationTaskTargets(ids, values, masks, {})
    context = {"domain": "clare", "groups": {s: "train" if i < 12 else "a" if i < 17 else "b" for i, s in enumerate(ids)},
               "vehicle_groups": {}, "regression": {}}
    return outputs, targets, tasks, context


@pytest.mark.parametrize("family", ["linear", "minirocket"])
def test_partial_fields_zero_observations_grouped_scores_and_frozen_reuse(family):
    outputs, targets, tasks, context = _public_inputs()
    bundle = fit_native_consumers(outputs=outputs, targets=targets, definitions=tasks, context=context,
                                  family=family, minirocket_kernels=84)
    assert [len(row["train_sample_ids"]) for row in bundle["fit_rows"]] == [12, 6, 6]
    result = evaluate_native_consumers(bundle, output=outputs["validation"], targets=targets)
    assert result["sample_count"] == 6 and result["no_observation_fraction"] == pytest.approx(1 / 6)
    assert len(result["prediction_rows"]) == 18
    assert sum(row["target_valid"] for row in result["prediction_rows"]) == 12
    regression = next(row for row in result["task_summary"] if row["task"] == "fields")
    rows = [row for row in result["group_metrics"] if row["task"] == "fields" and row["status"] == "completed"]
    expected = np.mean([np.mean([row["rmse"] for row in rows if row["group_id"] == group]) for group in ("a", "b")])
    assert regression["value"] == expected
    train = outputs["train"]
    with pytest.raises(ValueError, match="training windows"):
        evaluate_native_consumers(bundle, output=train, targets=targets)
    leaked = dict(context, groups=dict(context["groups"], s12="train"))
    with pytest.raises(ValueError, match="subjects overlap"):
        fit_native_consumers(outputs=outputs, targets=targets, definitions=tasks, context=leaked)
    # No fit is called when evaluating another observation condition of these windows.
    for model in bundle["models"].values():
        model.fit = lambda *args, **kwargs: pytest.fail("stress evaluation refitted a consumer")
    assert evaluate_native_consumers(bundle, output=outputs["validation"], targets=targets) == result


def test_tail_keeps_missing_baselines_and_rejects_nonfinite_predictions():
    result = regression_metrics([1., 2., 100.], [0., 0., 0.], sample_ids=("a", "b", "tail"),
                                persistence=[0., 0., np.nan], no_observation=[False, False, True])
    assert result["support"] == 3 and result["persistence_support"] == 2
    assert result["rmse"] == pytest.approx(np.sqrt(10005 / 3))
    assert result["skill_vs_persistence"] == 0
    assert result["top_five_windows"][0]["sample_id"] == "tail"
    with pytest.raises(ValueError, match="must be finite"):
        regression_metrics([1.], [np.nan], sample_ids=("a",))


def test_dingxin_uses_fixed_parameters_view_weights_and_vehicle_aggregation():
    rng = np.random.default_rng(43)
    ids = tuple(f"s{i}" for i in range(24))
    outputs = {"train": _output(ids[:18], rng.normal(size=(18, 64))),
               "validation": _output(ids[18:], rng.normal(size=(6, 64)), empty_last=True)}
    values = {"maneuver_regression": torch.arange(12.).repeat_interleave(2),
              "maneuver_classification": torch.arange(12).remainder(3).repeat_interleave(2)}
    targets = ApplicationTaskTargets(ids, values, {n: torch.ones_like(v, dtype=torch.bool) for n, v in values.items()},
                                    {"fit_sample_ids": list(ids[:18])}, torch.full((24,), .5))
    tasks = (ApplicationTaskDefinition("maneuver_regression", "regression", 1),
             ApplicationTaskDefinition("maneuver_classification", "classification", 3))
    context = {"domain": "dingxin", "groups": {sample: "sortie" for sample in ids},
        "vehicle_groups": {sample: f"v{i // 2}" for i, sample in enumerate(ids)},
        "regression": {"maneuver_regression": {"scale": [2.], "fields": ["maneuver"],
            "persistence": {sample: [0.] for sample in ids}}}}
    bundle = fit_native_consumers(outputs=outputs, targets=targets, definitions=tasks, context=context)
    assert all(row["selected_parameter"] == 1. and row["train_weight_sum"] == 9. for row in bundle["fit_rows"])
    result = evaluate_native_consumers(bundle, output=outputs["validation"], targets=targets)
    assert len(result["prediction_rows"]) == 12 and len(result["vehicle_prediction_rows"]) == 6
    assert all(row["support"] == 3 for row in result["group_metrics"])
    assert result["independent_unit"] == "sortie_descriptive_only"
    with pytest.raises(ValueError, match="calibration"):
        fit_native_consumers(outputs=outputs, targets=replace(targets, manifest={"fit_sample_ids": ids}), definitions=tasks, context=context)
    with pytest.raises(ValueError, match="sum to one"):
        fit_native_consumers(outputs=outputs, targets=replace(targets, sample_weights=torch.ones(24)), definitions=tasks, context=context)


def test_consumer_artifacts_resume_and_reject_mutated_targets(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks import v4_grouped_consumers as module
    outputs, targets, tasks, context = _public_inputs()
    kwargs = dict(outputs=outputs, targets=targets, definitions=tasks, context=context,
                  output_root=tmp_path, label_used_for_encoder_training=True, minirocket_kernels=84)
    first = run_native_method_consumers(**kwargs)
    def forbidden(**kwargs):
        pytest.fail("cached consumer must not refit")
    monkeypatch.setattr(module, "fit_native_consumers", forbidden)
    assert run_native_method_consumers(**kwargs) == first
    changed = dict(targets.values)
    changed["classify"] = (changed["classify"] + 1).remainder(3)
    with pytest.raises(ValueError, match="source/data/config changed"):
        run_native_method_consumers(**(kwargs | {"targets": replace(targets, values=changed)}))
