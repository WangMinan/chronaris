from __future__ import annotations

import numpy as np
import pytest
import torch

from chronaris.representation import (
    TrainOnlyPCAProjector,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.representation.contracts import RepresentationContractError
from tests.representation.test_contracts import _sample


def test_exact_native_quantiles_beyond_torch_size_limit_keep_boundary_observations():
    from chronaris.representation.normalization import _fit_stream_statistics, _fit_stream_statistics_from_observed_parts
    count = 2**24 + 2
    values = torch.ones(count)
    values[:count // 2] = 0
    values[count // 2 - 1:count // 2 + 1] = torch.tensor([.25, .75])
    # Both central observations matter: decimation can change this median to .25.
    streamed = _fit_stream_statistics_from_observed_parts([[values[:count // 2], values[count // 2:]]], minimum_scale=1e-6)
    materialized = _fit_stream_statistics(values.reshape(1, -1, 1), torch.ones((1, count, 1), dtype=torch.bool), minimum_scale=1e-6)
    for statistics in (streamed, materialized):
        assert statistics.center.tolist() == [.5]
        assert statistics.scale.tolist() == [1.]
        assert statistics.valid_count.tolist() == [count]
        assert statistics.active_mask.tolist() == [True]


def test_robust_normalizer_fits_only_explicit_train_samples():
    batch = collate_observation_samples(
        [_sample("train_a"), _sample("train_b", shift=2.0), _sample("test", shift=1000.0)]
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train_a", "train_b"),
        held_out_sample_ids=("test",),
    )

    assert normalizer.fit_sample_ids == ("train_a", "train_b")
    assert "test" not in normalizer.to_manifest()["fit_sample_ids"]
    assert normalizer.physiology is not None
    assert normalizer.physiology.center.tolist() == pytest.approx([2.0, 3.0])
    transformed = normalizer.transform(batch)
    assert transformed.sample_ids == batch.sample_ids
    assert transformed.physiology_feature_mask.equal(batch.physiology_feature_mask)


def test_robust_normalizer_rejects_held_out_fit_overlap():
    batch = collate_observation_samples([_sample("train"), _sample("test")])

    with pytest.raises(RepresentationContractError, match="held-out"):
        TrainOnlyRobustNormalizer().fit(
            batch,
            train_sample_ids=("train", "test"),
            held_out_sample_ids=("test",),
        )


def test_robust_normalizer_batch_provider_matches_materialized_fit():
    samples = {
        "train_a": _sample("train_a"),
        "train_b": _sample("train_b", shift=2.0),
        "test": _sample("test", shift=1000.0),
    }
    batch = collate_observation_samples(tuple(samples.values()))
    expected = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train_a", "train_b"),
        held_out_sample_ids=("test",),
    )
    actual = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        lambda sample_ids: collate_observation_samples(
            tuple(samples[sample_id] for sample_id in sample_ids)
        ),
        train_sample_ids=("train_b", "train_a"),
        held_out_sample_ids=("test",),
        batch_size=1,
    )

    assert actual.fit_sample_ids == ("train_a", "train_b")
    assert actual.physiology is not None
    assert expected.physiology is not None
    assert torch.equal(actual.physiology.center, expected.physiology.center)
    assert torch.equal(actual.physiology.scale, expected.physiology.scale)


def test_pca_ignores_held_out_extreme_values_and_zero_pads_to_64():
    train = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    held_out = np.asarray([[1_000_000.0, -1_000_000.0]])
    values = np.vstack([train, held_out])
    projector = TrainOnlyPCAProjector().fit(
        values,
        row_sample_ids=("train", "train", "train", "test"),
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )

    assert projector.center is not None
    assert projector.center.tolist() == pytest.approx([1.0, 1.0])
    transformed = projector.transform(held_out)
    assert transformed.shape == (1, 64)
    assert np.all(transformed[:, 2:] == 0)
    assert projector.to_manifest()["fit_sample_ids"] == ["train"]


def test_pca_rejects_train_test_overlap():
    with pytest.raises(RepresentationContractError, match="disjoint"):
        TrainOnlyPCAProjector().fit(
            np.ones((2, 3)),
            row_sample_ids=("a", "b"),
            train_sample_ids=("a", "b"),
            held_out_sample_ids=("b",),
        )
