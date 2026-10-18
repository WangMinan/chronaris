from dataclasses import replace

import pytest
import torch

from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.modeling.fusion_encoders.deep_baselines import (
    CausalContiFormerFusionEncoder, CausalMulTFusionEncoder, DeepBaselineEncoderConfig,
)
from chronaris.modeling.fusion_encoders.same_backbone import SameBackboneFusion
from chronaris.modeling.fusion_encoders.single_stream import ContinuousTimeSingleStreamEncoder, SingleStreamEncoderConfig
from chronaris.evaluation.application_tasks.performance_pilot_data import future_query_variants
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize('method', ['physiology', 'vehicle', 'mult', 'contiformer', 'same_backbone'])
def test_cached_queries_match_raw_forward_and_preserve_history(method):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    batch = collate_observation_samples([_sample('a'), _sample('b', shift=2.)])
    p = causal_query_stream(batch, stream_name='physiology')
    v = causal_query_stream(batch, stream_name='vehicle')
    if method in ('physiology', 'vehicle'):
        model = ContinuousTimeSingleStreamEncoder(SingleStreamEncoderConfig(method, 2 if method == 'physiology' else 1))
        encode = lambda a, b: model.forward_queried(a if method == 'physiology' else b)
    elif method in ('mult', 'contiformer'):
        cls = CausalMulTFusionEncoder if method == 'mult' else CausalContiFormerFusionEncoder
        model = cls(DeepBaselineEncoderConfig(method, 2, 1))
        encode = model.forward_queried
    else:
        model = SameBackboneFusion(2, 1); encode = model.forward_queried
    model.eval()
    with torch.no_grad():
        raw = model(batch).sequence_embedding
        cached = encode(p, v).sequence_embedding
        torch.testing.assert_close(raw, cached, atol=0, rtol=0)
        past = batch.query_timestamps_s <= 15
        for a, b in zip(future_query_variants(p), future_query_variants(v)):
            actual = encode(a, b).sequence_embedding
            torch.testing.assert_close(cached[past], actual[past], atol=1e-5, rtol=1e-5)
    if method == 'same_backbone':
        encode(p, v).sequence_embedding.square().mean().backward()
        for branch in (model.physiology, model.vehicle):
            assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in branch.parameters())
