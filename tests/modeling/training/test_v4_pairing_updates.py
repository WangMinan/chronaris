import pytest
import torch

from chronaris.modeling.training import candidate_screen as screen
from chronaris.modeling.training import EncoderCandidateConfig, load_common_pretraining_checkpoint
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.representation import AugmentationPolicy, FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples
from tests.modeling.training.test_candidate_screen import _sample


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pairing_warmup_activation_and_resume_preserve_projection_optimizer_states(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    samples = [_sample(f"sample_{i}", i) for i in range(6)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage("pair_updates", batch.sample_ids[:4], batch.sample_ids[4:], (), development_only=True)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids)
    config = screen.CandidateScreenConfig(max_updates=53, batch_size=2, effective_batch_size=4, validation_interval=50,
        checkpoint_interval=25, early_stopping=False, independent_pairing_enabled=True, independent_pair_weight=.1, device=device)
    kwargs = dict(method_name="chronaris", candidate=EncoderCandidateConfig(hidden_dim=8), batch=batch, fold=fold,
        physiology_feature_names=samples[0].schema.physiology_feature_names, vehicle_feature_names=samples[0].schema.vehicle_feature_names,
        vehicle_field_labels=(), normalizer=normalizer, config=config, chronaris_fusion_kind="safe_lag",
        augmentation_policy=AugmentationPolicy(point_dropout_probability=0., modality_dropout_probability=0.,
            block_duration_min_s=.1, block_duration_max_s=.1, timestamp_jitter_sigma_s=.005, clock_offset_limit_s=0.))
    complete = screen.train_pretext_candidate(output_root=tmp_path / "complete", **kwargs)
    original = screen.pretext_micro_step
    calls = 0
    def interrupt(**args):
        nonlocal calls
        calls += 1
        if calls == 102:
            raise RuntimeError("interrupt during first active pairing update")
        return original(**args)
    monkeypatch.setattr(screen, "pretext_micro_step", interrupt)
    with pytest.raises(RuntimeError, match="first active pairing"):
        screen.train_pretext_candidate(output_root=tmp_path / "resumed", **kwargs)
    monkeypatch.setattr(screen, "pretext_micro_step", original)
    resumed = screen.train_pretext_candidate(output_root=tmp_path / "resumed", **kwargs)
    a = torch.load(complete.last_checkpoint_path, map_location="cpu", weights_only=True)
    b = torch.load(resumed.last_checkpoint_path, map_location="cpu", weights_only=True)
    for name in ("encoder_state_dict", "optimizer_state_dict", "rng_state", "data_cursor", "training_rows"):
        assert canonical_training_state_sha256(a[name]) == canonical_training_state_sha256(b[name]), name
    terms = [row for row in b["training_rows"] if row["term_name"] == "independent_window_pairing"]
    assert all(row["status"] == "scheduled_zero" and row["related_parameter_gradient_norm"] == 0 for row in terms if row["step"] <= 50)
    assert all(row["count"] == 2 and row["related_parameter_gradient_norm"] > 0 for row in terms if row["step"] > 50)
    assert b["epoch_rows"][-1]["mechanism_validation"]["independent_pairing"]["valid_pair_count"] == 2
    restored, _, _, payload = load_common_pretraining_checkpoint(resumed.last_checkpoint_path)
    assert restored.backbone.config.independent_pairing_enabled and payload["config"]["independent_pair_weight"] == .1
    if device == "cpu":
        _check_guided_pairing(tmp_path, restored, normalizer, batch, fold, resumed.last_checkpoint_path)


def _check_guided_pairing(root, encoder, normalizer, batch, fold, checkpoint):
    from chronaris.evaluation.application_tasks.application_finetuning import (
        EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method)
    from chronaris.evaluation.application_tasks.application_finetuning_export import export_finetuned_application_representations
    from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
    from chronaris.modeling.training.rng import isolated_training_rng
    roles = {role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")}
    targets = ApplicationTaskTargets(batch.sample_ids, {"classify": torch.arange(len(batch.sample_ids)).remainder(3)},
        {"classify": torch.ones(len(batch.sample_ids), dtype=torch.bool)}, {"domain": "unit_test"})
    with isolated_training_rng(17):
        model = EndToEndApplicationModel(method_name="chronaris", encoder=encoder, normalizer=normalizer,
            naive_encoder=None, task_definitions=(ApplicationTaskDefinition("classify", "classification", 3),))
    before = {name: value.clone() for name, value in model.encoder.backbone.independent_pairing.state_dict().items()}
    result = train_end_to_end_application_method(model=model, batch=batch, targets=targets, role_sample_ids=roles,
        source_checkpoint_path=checkpoint, output_root=root / "guided", config=EndToEndFineTuningConfig(
            max_updates=2, head_warmup_updates=1, batch_size=2, effective_batch_size=4, validation_interval=1, early_stopping=False))
    assert result.optimizer_updates == 3
    assert any(not torch.equal(value, model.encoder.backbone.independent_pairing.state_dict()[name]) for name, value in before.items())
    output = export_finetuned_application_representations(model=model, checkpoint_path=result.last_checkpoint_path,
        batch=batch, role_sample_ids=roles, output_root=root / "guided_exports", batch_size=2, export_roles=("train", "validation"))
    assert output["validation"].sample_ids == fold.validation_sample_ids and output["validation"].sequence_embedding.shape[-1] == 64
