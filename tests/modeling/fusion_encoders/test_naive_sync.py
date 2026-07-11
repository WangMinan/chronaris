from __future__ import annotations

import torch

from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.representation import collate_observation_samples
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.modeling.fusion_encoders.test_single_stream import _future_sample
from tests.representation.test_contracts import _sample


def test_naive_sync_fits_normalizer_and_pca_on_train_sample_only():
    batch = collate_observation_samples(
        [_sample("train"), _sample("test", shift=1000.0)]
    )
    encoder = NaiveTimeSyncEncoder().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )

    assert encoder.normalizer.fit_sample_ids == ("train",)
    assert encoder.projector.fit_sample_ids == ("train",)
    assert "test" not in encoder.to_manifest()["pca_projector"]["fit_sample_ids"]


def test_naive_sync_produces_fixed_representation_and_valid_mean_pooling():
    batch = collate_observation_samples([_sample("train"), _sample("test", shift=2.0)])
    encoder = NaiveTimeSyncEncoder().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    output = NaiveTimeSyncFusionAdapter(
        encoder=encoder,
        fold_id="fold_a",
        checkpoint_sha256="3" * 64,
    )(batch)

    assert output.sequence_embedding.shape == (2, 96, 64)
    expected = output.sequence_embedding.sum(dim=1) / output.valid_mask.sum(
        dim=1,
        keepdim=True,
    )
    assert torch.allclose(output.pooled_embedding, expected)


def test_naive_sync_future_change_does_not_change_past_queries():
    train = _future_sample("train", 3.0)
    base = _future_sample("test", 9.0)
    changed = _future_sample("test", 9999.0)
    fit_batch = collate_observation_samples([train, base])
    encoder = NaiveTimeSyncEncoder().fit(
        fit_batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    adapter = NaiveTimeSyncFusionAdapter(
        encoder=encoder,
        fold_id="fold_a",
        checkpoint_sha256="4" * 64,
    )
    first = adapter(collate_observation_samples([base]))
    second = adapter(collate_observation_samples([changed]))
    past = first.timestamps_s[0] < 20.0

    assert torch.allclose(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        atol=1e-6,
        rtol=1e-6,
    )


def test_naive_sync_checkpoint_round_trips_projection_arrays(tmp_path):
    batch = collate_observation_samples([_sample("train"), _sample("test", shift=2.0)])
    encoder = NaiveTimeSyncEncoder().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    path = save_naive_time_sync_checkpoint(tmp_path / "naive.pt", encoder=encoder)
    loaded = load_naive_time_sync_checkpoint(path)
    checkpoint_hash = sha256_file(path)
    first = NaiveTimeSyncFusionAdapter(
        encoder=encoder,
        fold_id="fold_a",
        checkpoint_sha256=checkpoint_hash,
    )(batch)
    second = NaiveTimeSyncFusionAdapter(
        encoder=loaded,
        fold_id="fold_a",
        checkpoint_sha256=checkpoint_hash,
    )(batch)

    assert loaded.projector.fit_sample_ids == ("train",)
    assert loaded.projector.solver == "randomized"
    assert torch.equal(first.sequence_embedding, second.sequence_embedding)


def test_naive_sync_batch_provider_matches_materialized_fit():
    samples = {
        "train": _sample("train"),
        "test": _sample("test", shift=2.0),
    }
    batch = collate_observation_samples(tuple(samples.values()))
    expected = NaiveTimeSyncEncoder().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    normalizer = expected.normalizer
    actual = NaiveTimeSyncEncoder().fit_from_batch_provider(
        lambda sample_ids: collate_observation_samples(
            tuple(samples[sample_id] for sample_id in sample_ids)
        ),
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
        normalizer=normalizer,
        batch_size=1,
    )

    assert actual.projector.solver == "randomized"
    assert torch.equal(
        NaiveTimeSyncFusionAdapter(
            encoder=actual,
            fold_id="fold_a",
            checkpoint_sha256="5" * 64,
        )(batch).sequence_embedding,
        NaiveTimeSyncFusionAdapter(
            encoder=expected,
            fold_id="fold_a",
            checkpoint_sha256="5" * 64,
        )(batch).sequence_embedding,
    )
