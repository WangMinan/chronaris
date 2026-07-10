from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders import causal_query_stream
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
)
from third_party.contiformer.physiopro.network import ContiFormerEncoder


def _sample(*, future_value: float = 9.0) -> ObservedDualStreamSample:
    schema = ObservationSchema(
        schema_id="causal_query_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    return ObservedDualStreamSample(
        sample_id="sample_a",
        group_id="group_a",
        schema=schema,
        physiology_values=np.asarray([[1.0], [2.0], [3.0], [future_value]], dtype=np.float32),
        physiology_timestamps_s=np.asarray([1.0, 5.0, 5.0, 20.0], dtype=np.float64),
        physiology_feature_mask=np.ones((4, 1), dtype=bool),
        vehicle_values=np.asarray([[10.0]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([2.0], dtype=np.float64),
        vehicle_feature_mask=np.ones((1, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(b"sample_a").hexdigest(),
    )


def test_causal_query_uses_latest_feature_observation_and_stable_duplicate_order():
    queried = causal_query_stream(
        collate_observation_samples([_sample()]),
        stream_name="physiology",
    )
    query = queried.timestamps_s[0]
    before_first = query < 1.0
    after_duplicate = (query >= 5.0) & (query < 20.0)

    assert not queried.feature_mask[0, before_first, 0].any()
    assert torch.all(queried.values[0, after_duplicate, 0] == 3.0)
    assert torch.all(queried.observation_age_s[0, after_duplicate, 0] >= 0)


def test_future_observation_change_does_not_change_past_query_values():
    first = causal_query_stream(
        collate_observation_samples([_sample(future_value=9.0)]),
        stream_name="physiology",
    )
    changed = causal_query_stream(
        collate_observation_samples([_sample(future_value=9999.0)]),
        stream_name="physiology",
    )
    past = first.timestamps_s[0] < 20.0

    assert torch.equal(first.values[0, past], changed.values[0, past])
    assert torch.equal(first.feature_mask[0, past], changed.feature_mask[0, past])


def test_causal_contiformer_attention_blocks_future_query_leakage():
    torch.manual_seed(7)
    encoder = ContiFormerEncoder(
        input_dim=2,
        model_dim=8,
        num_heads=2,
        depth=2,
        dropout=0.0,
        causal=True,
    ).eval()
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    changed = values.clone()
    changed[:, 2] = 9999.0
    time = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.ones((1, 3), dtype=torch.bool)

    first, _ = encoder(values, time_axis=time, mask=mask)
    second, _ = encoder(changed, time_axis=time, mask=mask)

    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-6)
