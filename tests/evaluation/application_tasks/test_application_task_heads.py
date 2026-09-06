import copy
from dataclasses import replace

import pytest
import torch

from chronaris.evaluation.application_tasks.application_task_heads import (
    ApplicationTaskDefinition, ApplicationTaskTargets, application_task_losses,
    fit_application_task_parameters, select_application_targets,
    effective_task_counts,
)
from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method,
)
from chronaris.evaluation.application_tasks.application_finetuning_export import export_finetuned_application_representations
from chronaris.modeling.training import build_trainable_fusion_encoder
from chronaris.representation import collate_observation_samples, TrainOnlyRobustNormalizer
from chronaris.representation.contracts import RepresentationContractError
from tests.representation.test_contracts import _sample


def test_partial_fields_use_valid_sample_means_and_equal_tasks():
    definitions = (ApplicationTaskDefinition("difficulty", "classification", 2),
                   ApplicationTaskDefinition("physiology", "regression", 3))
    targets = ApplicationTaskTargets(("a", "b"),
        {"difficulty": torch.tensor([0, 1]), "physiology": torch.tensor([[1., float("nan"), 3.], [0., 2., 0.]])},
        {"difficulty": torch.tensor([True, False]), "physiology": torch.tensor([[True, False, True], [False, True, False]])},
        {}, sample_weights=torch.tensor([.5, .5]))
    parameters = {"tasks": {"difficulty": {"class_weights": [1., 1.]},
                            "physiology": {"center": [0., 0., 0.], "scale": [1., 1., 1.]}}}
    logits = torch.zeros(2, 2, requires_grad=True)
    fields = torch.zeros(2, 3, requires_grad=True)
    output = {"task_predictions": {"difficulty": logits, "physiology": fields}}
    losses = application_task_losses(output, select_application_targets(targets, targets.sample_ids, "cpu"), definitions, parameters)
    assert losses["physiology"].item() == 4.5
    assert losses["difficulty"].item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert losses["total"].item() == pytest.approx((4.5 + torch.log(torch.tensor(2.)).item()) / 2)
    losses["total"].backward()
    assert fields.grad[0, 1] == fields.grad[1, 0] == fields.grad[1, 2] == 0
    assert logits.grad[1].abs().sum() == 0
    with pytest.raises(RepresentationContractError, match="finite"):
        replace(targets, valid_masks=targets.valid_masks | {"physiology": torch.ones(2, 3, dtype=torch.bool)})


def test_custom_task_heads_train_export_and_reject_changed_label_sources(tmp_path):
    ids = ("train_a", "train_b", "train_c", "validation", "held_out")
    batch = collate_observation_samples([_sample(name, shift=i) for i, name in enumerate(ids)])
    roles = dict(train=ids[:3], validation=ids[3:4], held_out=ids[4:])
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=roles["train"],
        held_out_sample_ids=roles["validation"] + roles["held_out"])
    definitions = (ApplicationTaskDefinition("difficulty", "classification", 4),
        ApplicationTaskDefinition("response", "regression", 2),
        ApplicationTaskDefinition("events", "sequence_classification", 2))
    values = dict(difficulty=torch.tensor([0, 1, 2, 3, 0]), response=torch.arange(10.).reshape(5, 2),
                  events=torch.arange(96)[None].expand(5, -1) % 2)
    masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}
    masks["response"][1, 1] = False
    targets = ApplicationTaskTargets(ids, values, masks, {"smoke_only": True})
    parameters = fit_application_task_parameters(targets, definitions, roles["train"])
    changed = copy.deepcopy(targets)
    changed.values["response"][3:] += 100000
    assert parameters == fit_application_task_parameters(changed, definitions, roles["train"])
    source = tmp_path / "source.pt"
    torch.save({"label_used_for_encoder_training": False}, source)
    model = EndToEndApplicationModel(method_name="physiology_only",
        encoder=build_trainable_fusion_encoder("physiology_only",
            physiology_feature_names=("physiology.a", "physiology.b"), vehicle_feature_names=("vehicle.a",)),
        normalizer=normalizer, naive_encoder=None, task_definitions=definitions)
    config = EndToEndFineTuningConfig(max_epochs=1, batch_size=3)
    args = dict(model=model, batch=batch, targets=targets, role_sample_ids=roles,
        source_checkpoint_path=source, output_root=tmp_path / "models", config=config)
    result = train_end_to_end_application_method(**args)
    payload = torch.load(result.best_checkpoint_path, weights_only=True)
    assert payload["task_parameters"] == parameters
    assert payload["selection_uses_validation_labels"] is True
    assert payload["encoder_backprop_uses_labels"] is True
    assert model(batch)["task_predictions"]["response"].shape == (5, 2)
    exports = export_finetuned_application_representations(model=model, checkpoint_path=result.best_checkpoint_path,
        batch=batch, role_sample_ids=roles, output_root=tmp_path / "representations", batch_size=2)
    assert exports["held_out"].pooled_embedding.shape == (1, 64)
    with pytest.raises(RepresentationContractError, match="protocol changed"):
        train_end_to_end_application_method(**(args | {"targets": changed}))


def test_task_reduction_across_accumulation_matches_a_whole_batch():
    definitions = tuple(ApplicationTaskDefinition(name, "regression", 1) for name in ("a", "b"))
    targets = ApplicationTaskTargets(("s0", "s1", "s2", "s3"),
        {"a": torch.tensor([1., 1., 0., 0.]), "b": torch.tensor([0., 2., 4., 6.])},
        {"a": torch.tensor([True, True, False, False]), "b": torch.tensor([False, True, True, True])}, {})
    parameters = {"tasks": {name: {"center": [0.], "scale": [1.]} for name in ("a", "b")}}
    batches = (targets.sample_ids[:2], targets.sample_ids[2:])
    counts = effective_task_counts(targets, definitions, batches, batch=None, provider=None, method_name="chronaris")
    parameter = torch.tensor(0., requires_grad=True)
    total = 0
    for ids in batches:
        output = {"task_predictions": {name: parameter.expand(len(ids), 1) for name in ("a", "b")}}
        losses = application_task_losses(output, select_application_targets(targets, ids, "cpu"), definitions, parameters)
        total = total + sum(losses[name] * losses["counts"][name] / count for name, count in counts.items()) / 2
    full = application_task_losses({"task_predictions": {name: parameter.expand(4, 1) for name in ("a", "b")}},
        select_application_targets(targets, targets.sample_ids, "cpu"), definitions, parameters)["total"]
    torch.testing.assert_close(total, full)
    total.backward()
    assert parameter.grad.item() == pytest.approx(-5.)
