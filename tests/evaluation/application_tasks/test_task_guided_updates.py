import copy

import pytest
import torch

from chronaris.modeling.training import (CandidateScreenConfig, EncoderCandidateConfig,
    train_pretext_candidate, load_common_pretraining_checkpoint)
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.representation import collate_observation_samples, TrainOnlyRobustNormalizer, FoldLineage
from chronaris.evaluation.application_tasks import application_finetuning as tuning
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets, ApplicationTaskDefinition
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_head_warmup_and_joint_updates_resume_without_confirmation_labels(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ids = tuple(f"sample_{i}" for i in range(5))
    batch = collate_observation_samples([_sample(name, shift=i) for i, name in enumerate(ids)])
    roles = dict(train=ids[:4], validation=ids[4:], held_out=("sealed_confirmation",))
    fold = FoldLineage(fold_id="guided_test", train_sample_ids=roles["train"],
        validation_sample_ids=roles["validation"], held_out_sample_ids=roles["held_out"])
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=roles["train"], held_out_sample_ids=roles["validation"])
    source = train_pretext_candidate("physiology_only", candidate=EncoderCandidateConfig(candidate_id="D", hidden_dim=32, dropout=.2),
        batch=batch, fold=fold, physiology_feature_names=("physiology.a", "physiology.b"), vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(), normalizer=normalizer, output_root=tmp_path / "source",
        config=CandidateScreenConfig(max_updates=2, batch_size=2, effective_batch_size=4,
            validation_interval=2, early_stopping=False, device=device))
    encoder, _, normalizer, source_payload = load_common_pretraining_checkpoint(source.best_checkpoint_path)
    definitions = (ApplicationTaskDefinition("classification", "classification", 2),
                   ApplicationTaskDefinition("regression", "regression", 2))
    model = tuning.EndToEndApplicationModel(method_name="physiology_only", encoder=encoder,
        normalizer=normalizer, naive_encoder=None, task_definitions=definitions)
    values = {"classification": torch.tensor([0, 1, 0, 1, 0]), "regression": torch.arange(10.).reshape(5, 2)}
    targets = ApplicationTaskTargets(ids, values, {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}, {})
    config = tuning.EndToEndFineTuningConfig(max_updates=3, head_warmup_updates=2,
        batch_size=2, effective_batch_size=4, checkpoint_interval=1,
        validation_interval=1, early_stopping=False, device=device)
    arguments = dict(batch=batch, targets=targets, role_sample_ids=roles,
        source_checkpoint_path=source.best_checkpoint_path, config=config)
    save = tuning._atomic_save
    warmup_snapshots = []

    def record_warmup(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["step_count"] == 2:
            warmup_snapshots.append(copy.deepcopy(payload))

    monkeypatch.setattr(tuning, "_atomic_save", record_warmup)
    complete = tuning.train_end_to_end_application_method(model=copy.deepcopy(model), output_root=tmp_path / "complete", **arguments)
    warmup = warmup_snapshots[0]
    for name, value in source_payload["encoder_state_dict"].items():
        torch.testing.assert_close(value, warmup["model_state_dict"]["encoder." + name].cpu(), atol=0, rtol=0)
    assert warmup["encoder_backprop_uses_labels"] is False
    assert complete.optimizer_updates == 5 and complete.head_warmup_updates == 2 and complete.joint_updates == 3
    original_step = tuning.pretext_micro_step
    calls = 0

    def interrupt(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("interrupted joint accumulation")
        return original_step(**kwargs)

    monkeypatch.setattr(tuning, "pretext_micro_step", interrupt)
    with pytest.raises(RuntimeError, match="joint accumulation"):
        tuning.train_end_to_end_application_method(model=copy.deepcopy(model), output_root=tmp_path / "resumed", **arguments)
    monkeypatch.setattr(tuning, "pretext_micro_step", original_step)
    resumed = tuning.train_end_to_end_application_method(model=copy.deepcopy(model), output_root=tmp_path / "resumed", **arguments)
    left, right = (torch.load(path, map_location="cpu", weights_only=True) for path in (complete.last_checkpoint_path, resumed.last_checkpoint_path))
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "data_cursor", "update_rows"):
        assert canonical_training_state_sha256(left[key]) == canonical_training_state_sha256(right[key]), key
    assert right["data_cursor"]["samples_seen"] == 20
    assert right["data_cursor"]["micro_batches_seen"] == 10
    assert len(right["data_cursor"]["sampling_order_sha256"]) == 64
    assert right["encoder_backprop_uses_labels"] is True
    assert right["representation_family"] == "task_guided_v4"
    assert any(not torch.equal(value, right["model_state_dict"]["encoder." + name]) for name, value in source_payload["encoder_state_dict"].items())
    assert all(row["public_weight"] == .2 and row["public_loss"] is not None for row in right["update_rows"] if row["stage"] == "joint_adaptation")
