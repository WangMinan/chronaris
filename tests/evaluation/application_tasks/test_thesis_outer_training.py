import hashlib

import numpy as np
import pytest

from chronaris.evaluation.application_tasks.thesis_native_outer_run import (
    ThesisNativeOuterConfig,
    _guarded_provider,
)
from chronaris.evaluation.application_tasks.thesis_dingxin_outer_run import (
    ThesisDingxinOuterConfig,
)
from chronaris.evaluation.application_tasks.thesis_outer_training import (
    train_frozen_outer_adapters,
)
from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
    select_observation_batch,
)


def _sample(sample_id, group_id, offset):
    schema = ObservationSchema(
        schema_id="thesis_outer_training_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=group_id,
        schema=schema,
        physiology_values=np.asarray([[1 + offset], [2 + offset]], np.float32),
        physiology_timestamps_s=np.asarray([0.0, 10.0]),
        physiology_feature_mask=np.ones((2, 1), dtype=bool),
        vehicle_values=np.asarray([[3 + offset], [4 + offset]], np.float32),
        vehicle_timestamps_s=np.asarray([0.0, 12.0]),
        vehicle_feature_mask=np.ones((2, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_frozen_outer_training_reuses_one_canonical_loop(tmp_path):
    samples = (
        _sample("train_a", "g1", 0),
        _sample("train_b", "g2", 1),
        _sample("validation", "g3", 2),
        _sample("held_out", "g4", 3),
    )
    batch = collate_observation_samples(samples)
    fold = FoldLineage(
        fold_id="outer_fold",
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    def provider(ids):
        return select_observation_batch(batch, ids)
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    selected = {
        method: {"candidate_id": "A"}
        for method in (
            "physiology_only",
            "vehicle_only",
            "mult",
            "contiformer",
            "chronaris",
        )
    }

    adapters, hashes, rows = train_frozen_outer_adapters(
        provider=provider,
        fold=fold,
        schema=samples[0].schema,
        normalizer=normalizer,
        selected_models=selected,
        output_root=tmp_path,
        seed=17,
        max_epochs=1,
        patience=1,
        batch_size=2,
        learning_rate=3e-4,
        weight_decay=1e-4,
        device="cpu",
        resume=True,
    )

    assert set(adapters) == set(hashes) == {
        "physiology_only",
        "vehicle_only",
        "naive_time_sync",
        "mult",
        "contiformer",
        "chronaris",
    }
    assert len(rows) == 6
    assert all(len(value) == 64 for value in hashes.values())


def test_native_outer_freeze_and_provider_reject_protocol_drift(monkeypatch):
    guarded = _guarded_provider(
        lambda sample_ids: tuple(sample_ids),
        allowed=("train", "validation"),
        forbidden=("held_out",),
    )
    assert guarded(("train",)) == ("train",)
    with pytest.raises(ValueError, match="outer-test"):
        guarded(("held_out",))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(ValueError, match="seeds or tasks changed"):
        ThesisNativeOuterConfig(
            runner_sha256="a" * 64,
            seeds=(17,),
            resume=True,
        )
    with pytest.raises(ValueError, match="resume enabled"):
        ThesisNativeOuterConfig(
            runner_sha256="a" * 64,
            resume=False,
        )
    with pytest.raises(ValueError, match="training budget"):
        ThesisDingxinOuterConfig(
            runner_sha256="a" * 64,
            max_epochs=0,
            resume=True,
        )
