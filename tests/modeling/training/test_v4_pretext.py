from dataclasses import replace

import pytest
import torch

from chronaris.modeling.training.pretext import CommonPretextHeadBundle
from chronaris.modeling.training.candidate_validation import _empty_loss_totals, _accumulate_loss_terms, _finalize_loss_totals
from chronaris.representation.pretext_targets import CommonPretextTargets


def _targets(values, mask=None):
    mask = torch.ones_like(values, dtype=torch.bool) if mask is None else mask
    return CommonPretextTargets(values, mask, values, mask, values.shape[-1], tuple(str(i) for i in range(len(values))))


def _forward(heads, sequence, targets, valid=None):
    valid = torch.ones(sequence.shape[:2], dtype=torch.bool) if valid is None else valid
    return heads(sequence, sequence, targets, positive_valid_mask=valid,
                 negative_valid_mask=valid, lag_valid_mask=valid.any(dim=1))


def test_equal_modality_loss_and_single_stream_gradients():
    sequence = torch.randn(2, 3, 8)
    values = torch.tensor([1., 3., 3., 3.]).expand(2, 3, -1).clone()
    heads = CommonPretextHeadBundle(representation_dim=8, target_feature_count=4, modality_feature_counts=(1, 3))
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()
    output = _forward(heads, sequence, _targets(values))
    assert output.terms[0].raw_loss.item() == 1.5  # mean of modal Huber losses .5 and 2.5
    output.terms[0].raw_loss.backward()
    torch.testing.assert_close(heads.reconstruction_head.bias.grad, torch.tensor([-.5, -1/6, -1/6, -1/6]))

    single = CommonPretextHeadBundle(representation_dim=8, target_feature_count=4,
        modality_feature_counts=(1, 3), input_streams=("physiology",))
    first = _forward(single, sequence, _targets(values))
    values[..., 1:] += 100000
    second = _forward(single, sequence, _targets(values))
    torch.testing.assert_close(first.total_loss, second.total_loss, atol=0, rtol=0)
    assert first.terms[2].count == 0 and first.terms[2].raw_loss is None
    first.total_loss.backward()
    assert single.reconstruction_head.weight.grad[1:].abs().sum() == 0
    assert all(parameter.grad is None for parameter in single.lag_head.parameters())


def test_validation_modal_aggregation_and_masked_pool_are_padding_invariant():
    torch.manual_seed(17)
    heads = CommonPretextHeadBundle(representation_dim=8, target_feature_count=4, modality_feature_counts=(1, 3))
    sequence = torch.randn(3, 5, 8)
    values, mask = torch.randn(3, 5, 4), torch.ones(3, 5, 4, dtype=torch.bool)
    mask[1:, :, 0] = False
    values[~mask] = torch.nan
    targets = _targets(values, mask)
    output = _forward(heads, sequence, targets)
    one, split = _empty_loss_totals(), _empty_loss_totals()
    _accumulate_loss_terms(one, output.terms)
    for i in range(3):
        _accumulate_loss_terms(split, _forward(heads, sequence[i:i+1], _targets(values[i:i+1], mask[i:i+1])).terms)
    assert _finalize_loss_totals(one) == pytest.approx(_finalize_loss_totals(split))
    padded_values = torch.cat((values, torch.zeros(3, 2, 4)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(3, 2, 4, dtype=torch.bool)), dim=1)
    padded = _forward(heads, torch.cat((sequence, 10000 * torch.randn(3, 2, 8)), dim=1),
        _targets(padded_values, padded_mask), torch.tensor([[True] * 5 + [False] * 2] * 3))
    torch.testing.assert_close(output.positive_lag_logits, padded.positive_lag_logits)
    torch.testing.assert_close(output.total_loss, padded.total_loss)
    output.total_loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in heads.parameters())
    with pytest.raises(ValueError, match="non-finite"):
        _forward(heads, sequence, replace(targets, reconstruction_mask=torch.ones_like(mask)))
