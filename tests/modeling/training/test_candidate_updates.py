from dataclasses import replace

import pytest
import torch

from chronaris.modeling.training import candidate_screen as screen
from chronaris.modeling.training import EncoderCandidateConfig
from chronaris.modeling.training.pretext import chronaris_auxiliary_weight_schedule
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples
from chronaris.representation.contracts import RepresentationContractError
from tests.modeling.training.test_candidate_screen import _sample


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_update_accumulation_replays_partial_update_and_data_cursor(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    samples = [_sample(f"sample_{i}", i) for i in range(9)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage(fold_id="updates", train_sample_ids=batch.sample_ids[:7],
                       validation_sample_ids=batch.sample_ids[7:8], held_out_sample_ids=batch.sample_ids[8:])
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids)
    config = screen.CandidateScreenConfig(max_updates=5, batch_size=2, effective_batch_size=6,
        validation_interval=3, checkpoint_interval=2, seed=17, device=device, early_stopping=False)
    arguments = dict(method_name="physiology_only", candidate=EncoderCandidateConfig(
        candidate_id="D", hidden_dim=32, dropout=.2), batch=batch, fold=fold,
        physiology_feature_names=("physiology.a",), vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(), normalizer=normalizer, config=config)
    complete = screen.train_pretext_candidate(output_root=tmp_path / "continuous", **arguments)
    original = screen.pretext_micro_step
    calls = 0

    def interrupt(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 9:
            raise RuntimeError("simulated process interruption inside accumulation")
        return original(**kwargs)

    monkeypatch.setattr(screen, "pretext_micro_step", interrupt)
    with pytest.raises(RuntimeError, match="simulated process"):
        screen.train_pretext_candidate(output_root=tmp_path / "resumed", **arguments)
    partial = torch.load(tmp_path / "resumed/physiology_only/D/last.pt", weights_only=True)
    assert partial["optimizer_updates"] == partial["step_count"] == 2
    assert partial["data_cursor"]["micro_batches_seen"] == 6
    assert partial["data_cursor"]["samples_seen"] == 12
    monkeypatch.setattr(screen, "pretext_micro_step", original)
    resumed = screen.train_pretext_candidate(output_root=tmp_path / "resumed", **arguments)
    assert complete.optimizer_updates == resumed.optimizer_updates == 5
    assert [row["optimizer_updates"] for row in resumed.epoch_rows] == [3, 5]
    left = torch.load(complete.last_checkpoint_path, map_location="cpu", weights_only=True)
    right = torch.load(resumed.last_checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("encoder_state_dict", "head_state_dict", "optimizer_state_dict", "rng_state",
                "data_cursor", "training_rows", "augmentation_rows"):
        assert canonical_training_state_sha256(left[key]) == canonical_training_state_sha256(right[key]), key
    assert right["data_cursor"]["micro_batches_seen"] == 15
    assert right["data_cursor"]["samples_seen"] == 30
    assert all(int(state["step"]) == 5 for state in right["optimizer_state_dict"]["state"].values())
    assert right["stage_update_counts"] == dict(pretraining=5, head_warmup=0, joint_adaptation=0)
    with pytest.raises(RepresentationContractError, match="protocol changed"):
        screen.train_pretext_candidate(output_root=tmp_path / "resumed",
            **(arguments | {"config": replace(config, effective_batch_size=4)}))


def test_update_schedule_has_zero_then_linear_mechanisms():
    for updates, fraction in ((0, 0), (50, 0), (51, 1 / 150), (125, .5), (200, 1), (500, 1)):
        weights = chronaris_auxiliary_weight_schedule(1, optimizer_updates=updates)
        assert weights.continuous_alignment == pytest.approx(.2 * fraction)
        assert weights.physical_consistency == pytest.approx(.1 * fraction)
    with pytest.raises(ValueError, match="multiple"):
        screen.CandidateScreenConfig(max_updates=10, batch_size=4, effective_batch_size=7)
