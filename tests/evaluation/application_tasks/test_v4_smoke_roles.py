import json
from types import SimpleNamespace

import pytest
import torch

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
    monkeypatch.setattr(run, "load_prepared_public_development", lambda *args, **kwargs: data)
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
