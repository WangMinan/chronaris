import json
from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_development_data as data_module
from chronaris.evaluation.application_tasks import v4_smoke_run as run
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
from chronaris.representation import FoldLineage


def test_engineering_subset_crops_partial_targets_and_rejects_unused_subject_windows(tmp_path, monkeypatch):
    ids = tuple(f"sample_{i}" for i in range(60))
    values = {"response": torch.arange(120.).reshape(60, 2)}
    masks = {"response": values["response"].remainder(3) != 0}
    targets = ApplicationTaskTargets(ids, values, masks, {})
    fold = FoldLineage("development", ids[:40], ids[40:], (), development_only=True)
    dataset = SimpleNamespace(batch_provider=lambda selected: tuple(selected), schema=None)
    data = SimpleNamespace(dataset=dataset, targets=targets, task_definitions=(), prepared_manifest_sha256="a" * 64,
        fold=lambda _: fold, sampling_hierarchy=lambda selected: {s: (s,) for s in selected.train_sample_ids})
    monkeypatch.setattr(data_module, "load_prepared_public_development", lambda *args, **kwargs: data)
    registry = tmp_path / "subjects.json"
    registry.write_text(json.dumps({"domains": {"clare": {"folds": {"development": [{}]}}}}))
    provider, _, actual_fold, _, _, actual_targets, _, _ = run._smoke_inputs("clare", tmp_path, registry)
    selected = actual_fold.train_sample_ids + actual_fold.validation_sample_ids
    assert actual_targets.sample_ids == selected and len(selected) == 40
    index = [ids.index(sample) for sample in selected]
    assert torch.equal(actual_targets.values["response"], values["response"][index])
    assert torch.equal(actual_targets.valid_masks["response"], masks["response"][index])
    omitted = next(sample for sample in ids if sample not in selected)
    with pytest.raises(ValueError, match="confirmation observations"):
        provider((omitted,))


def test_public_confirmation_input_provider_and_training_targets_exclude_outer_holdout(tmp_path, monkeypatch):
    ids = ("train", "validation", "held")
    targets = ApplicationTaskTargets(ids, {"response": torch.tensor([1., 2., 1000.])},
        {"response": torch.ones(3, dtype=torch.bool)}, {"source_role": "fixed_confirmation_subjects"})
    fold = FoldLineage("public_confirmation", ids[:1], ids[1:2], ids[2:])
    data = SimpleNamespace(dataset=SimpleNamespace(batch_provider=lambda selected: tuple(selected), schema=None),
        targets=targets, task_definitions=(), prepared_manifest_sha256="a" * 64,
        fold=lambda _: fold, sampling_hierarchy=lambda selected: {s: (s,) for s in selected.train_sample_ids})
    requested_roles = []
    def load(*args, **kwargs):
        requested_roles.append(kwargs["role"])
        return data
    monkeypatch.setattr(data_module, "load_prepared_public_development", load)
    registry = tmp_path / "subjects.json"
    registry.write_text(json.dumps({"domains": {"clare": {"folds": {"confirmation": [{}]}}}}))
    provider, _, actual_fold, _, _, actual_targets, _, full_data = data_module.load_development_inputs(
        "clare", tmp_path, registry, subject_role="confirmation")
    assert requested_roles == ["confirmation"] and actual_fold == fold
    assert actual_targets.sample_ids == ids[:2] and full_data.targets.sample_ids == ids
    assert provider(ids[:2]) == ids[:2]
    with pytest.raises(ValueError, match="confirmation observations"):
        provider(ids[2:])
    with pytest.raises(ValueError, match="complete public fold"):
        data_module.load_development_inputs("clare", tmp_path, registry, subject_role="confirmation", smoke=True)
