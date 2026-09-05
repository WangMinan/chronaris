from __future__ import annotations

import json
import copy
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeTargets,
)
from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel,
    EndToEndFineTuningConfig,
    train_end_to_end_application_method,
)
from chronaris.evaluation.application_tasks.application_finetuning_export import (
    export_finetuned_application_representations,
)
from chronaris.evaluation.application_tasks.simulation_finetuning_run import (
    _evaluate_seed,
)
from chronaris.modeling.training import build_trainable_fusion_encoder
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from tests.representation.test_contracts import _sample


def test_end_to_end_finetuning_declares_labels_and_resumes(tmp_path: Path) -> None:
    ids = ("train_a", "train_b", "train_c", "validation", "test_a", "test_b")
    batch = collate_observation_samples(
        [_sample(sample_id, shift=float(index)) for index, sample_id in enumerate(ids)]
    )
    roles = {
        "train": ids[:3],
        "validation": ids[3:4],
        "held_out": ids[4:],
    }
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=roles["train"],
        held_out_sample_ids=roles["validation"] + roles["held_out"],
    )
    state_row = torch.arange(96, dtype=torch.long) % 5
    targets = ApplicationConsumerSmokeTargets(
        sample_ids=ids,
        roles=("train", "train", "train", "validation", "held_out", "held_out"),
        future_workload_mean=torch.tensor([0.1, 0.5, 0.9, 0.6, 0.4, 0.8]),
        workload_class=torch.tensor([0, 1, 2, 1, 0, 2]),
        maneuver_state=torch.stack([state_row.roll(index) for index in range(len(ids))]),
        boundary_mask=torch.zeros((len(ids), 96), dtype=torch.bool),
        manifest={"smoke_only": False, "allowed_oracle_fields": ["workload", "maneuver_state"]},
    )
    source = tmp_path / "source.pt"
    torch.save({"label_used_for_encoder_training": False}, source)

    def build_model() -> EndToEndApplicationModel:
        return EndToEndApplicationModel(
            method_name="physiology_only",
            encoder=build_trainable_fusion_encoder(
                "physiology_only",
                physiology_feature_names=("physiology.a", "physiology.b"),
                vehicle_feature_names=("vehicle.a",),
            ),
            normalizer=normalizer,
            naive_encoder=None,
        )

    model = build_model()
    initial_model = copy.deepcopy(model)
    config = EndToEndFineTuningConfig(
        max_epochs=1,
        patience=1,
        batch_size=3,
        seed=17,
        device="cpu",
    )
    first = train_end_to_end_application_method(
        model=model,
        batch=batch,
        targets=targets,
        role_sample_ids=roles,
        source_checkpoint_path=source,
        output_root=tmp_path / "checkpoints",
        config=config,
    )
    resumed = train_end_to_end_application_method(
        model=build_model(),
        batch=batch,
        targets=targets,
        role_sample_ids=roles,
        source_checkpoint_path=source,
        output_root=tmp_path / "checkpoints",
        config=config,
    )
    outputs = export_finetuned_application_representations(
        model=model,
        checkpoint_path=first.best_checkpoint_path,
        batch=batch,
        role_sample_ids=roles,
        output_root=tmp_path / "representations",
        batch_size=2,
    )
    checkpoint = torch.load(first.best_checkpoint_path, map_location="cpu", weights_only=True)
    model._finetune_regression_mean = checkpoint["regression_train_mean"]
    model._finetune_regression_std = checkpoint["regression_train_std"]
    metrics, units = _evaluate_seed(
        seed=17,
        models={"physiology_only": model},
        outputs={"physiology_only": outputs},
        targets=targets,
    )
    manifest = json.loads(
        (tmp_path / "representations/physiology_only/held_out/representation_manifest.json").read_text()
    )

    assert first.status == "completed"
    assert resumed.status == "resumed"
    assert first.encoder_update_mode == "full_backbone"
    assert outputs["held_out"].sequence_embedding.shape == (2, 96, 64)
    assert manifest["label_used_for_encoder_training"] is True
    assert metrics and units
    assert {row["role"] for row in metrics} == {"validation", "held_out"}
    assert checkpoint["representation_family"] == "end_to_end_finetuned_v1"
    assert checkpoint["label_used_for_encoder_training"] is True

    # A real interruption must match uninterrupted optimizer/dropout updates,
    # not merely return a cached "completed" result.
    import chronaris.evaluation.application_tasks.application_finetuning as module
    from chronaris.modeling.training.rng import canonical_training_state_sha256
    longer = replace(config, max_epochs=3, patience=3)
    kwargs = dict(batch=batch, targets=targets, role_sample_ids=roles,
                  source_checkpoint_path=source, config=longer)
    continuous = train_end_to_end_application_method(
        model=copy.deepcopy(initial_model), output_root=tmp_path / "continuous", **kwargs,
    )
    save = module._atomic_save
    def interrupt(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["completed_epochs"] == 1:
            raise RuntimeError("intentional interruption")
    try:
        with patch.object(module, "_atomic_save", side_effect=interrupt):
            train_end_to_end_application_method(
                model=copy.deepcopy(initial_model), output_root=tmp_path / "interrupted", **kwargs,
            )
    except RuntimeError as error:
        assert str(error) == "intentional interruption"
    else:
        raise AssertionError("interruption was not exercised")
    torch.rand(1000)
    resumed_training = train_end_to_end_application_method(
        model=copy.deepcopy(initial_model), output_root=tmp_path / "interrupted", **kwargs,
    )
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in (
        continuous.last_checkpoint_path, resumed_training.last_checkpoint_path,
    )]
    assert states[0]["step_count"] == states[1]["step_count"] == 3
    assert canonical_training_state_sha256(states[0]["model_state_dict"], states[0]["optimizer_state_dict"], states[0]["rng_state"]) == canonical_training_state_sha256(states[1]["model_state_dict"], states[1]["optimizer_state_dict"], states[1]["rng_state"])
