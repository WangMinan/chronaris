from __future__ import annotations

import torch

from chronaris.modeling.fusion_encoders.multiscale_causal import (
    MultiScaleCausalFusionConfig,
    MultiScaleCausalFusionInput,
    MultiScaleCausalLagFusion,
    build_seconds_lag_mask,
)


def test_seconds_lag_boundaries_are_non_overlapping() -> None:
    queries = torch.tensor([[30.0]])
    keys = torch.tensor([[30.0, 25.0, 15.0, 0.0, -1.0]])
    query_valid = torch.ones_like(queries, dtype=torch.bool)
    key_valid = torch.ones_like(keys, dtype=torch.bool)
    ranges = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))
    masks = [
        build_seconds_lag_mask(
            queries,
            keys,
            query_valid_mask=query_valid,
            key_valid_mask=key_valid,
            lower_s=lower,
            upper_s=upper,
            range_index=index,
            use_causal_mask=True,
        )[0, 0]
        for index, (lower, upper) in enumerate(ranges)
    ]
    assert masks[0].tolist() == [True, True, False, False, False]
    assert masks[1].tolist() == [False, False, True, False, False]
    assert masks[2].tolist() == [False, False, False, True, False]
    assert torch.stack(masks).sum(dim=0).max().item() == 1


def test_causal_mask_excludes_future_but_ablation_sees_it() -> None:
    query = torch.tensor([[10.0]])
    keys = torch.tensor([[9.0, 11.0]])
    valid_q = torch.ones_like(query, dtype=torch.bool)
    valid_k = torch.ones_like(keys, dtype=torch.bool)
    causal = build_seconds_lag_mask(
        query,
        keys,
        query_valid_mask=valid_q,
        key_valid_mask=valid_k,
        lower_s=0.0,
        upper_s=5.0,
        range_index=0,
        use_causal_mask=True,
    )
    symmetric = build_seconds_lag_mask(
        query,
        keys,
        query_valid_mask=valid_q,
        key_valid_mask=valid_k,
        lower_s=0.0,
        upper_s=5.0,
        range_index=0,
        use_causal_mask=False,
    )
    assert causal[0, 0].tolist() == [True, False]
    assert symmetric[0, 0].tolist() == [True, True]


def test_gate_normalizes_only_over_available_scales() -> None:
    torch.manual_seed(17)
    model = MultiScaleCausalLagFusion(
        MultiScaleCausalFusionConfig(hidden_dim=4, output_dim=4)
    ).eval()
    states = torch.randn(1, 4, 4)
    output = model(
        MultiScaleCausalFusionInput(
            physiology_states=states,
            vehicle_states=states,
            physiology_valid_mask=torch.ones(1, 4, dtype=torch.bool),
            vehicle_valid_mask=torch.tensor([[False, False, False, True]]),
            query_timestamps_s=torch.tensor([[0.0, 5.0, 15.0, 30.0]]),
        )
    )
    assert torch.allclose(
        output.scale_gate_weights.sum(dim=-1),
        output.scale_available_mask.any(dim=-1).to(torch.float32),
    )
    assert torch.equal(output.scale_gate_weights == 0, ~output.scale_available_mask)
    assert torch.isfinite(output.sequence_embedding).all()


def test_empty_windows_produce_zero_context_and_weights() -> None:
    model = MultiScaleCausalLagFusion(
        MultiScaleCausalFusionConfig(hidden_dim=4, output_dim=4)
    ).eval()
    states = torch.randn(1, 3, 4)
    output = model(
        MultiScaleCausalFusionInput(
            physiology_states=states,
            vehicle_states=states,
            physiology_valid_mask=torch.ones(1, 3, dtype=torch.bool),
            vehicle_valid_mask=torch.zeros(1, 3, dtype=torch.bool),
            query_timestamps_s=torch.tensor([[0.0, 1.0, 2.0]]),
        )
    )
    assert not output.scale_available_mask.any()
    assert torch.count_nonzero(output.scale_gate_weights) == 0
    assert torch.count_nonzero(output.attended_vehicle_states) == 0
