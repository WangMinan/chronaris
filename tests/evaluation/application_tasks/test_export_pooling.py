from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks.application_finetuning import FINETUNING_FORMAT, EndToEndApplicationModel
from chronaris.evaluation.application_tasks.application_finetuning_export import (
    export_finetuned_application_representations, export_loaded_application_encoder,
)
from chronaris.evaluation.application_tasks.v4_pressure_run import _paired_representation
from chronaris.representation import FoldLineage, collate_observation_samples, select_observation_batch
from chronaris.representation.contracts import pool_exported_sequence
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_all_export_paths_share_masked_pooling(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    batch = collate_observation_samples([_sample(str(i)) for i in range(5)])
    fold = FoldLineage(fold_id='pooling', train_sample_ids=('0',),
        validation_sample_ids=('1', '2', '3', '4'), held_out_sample_ids=(), development_only=True)
    sequence = torch.randn(5, 96, 64, generator=torch.Generator().manual_seed(17))
    valid = torch.ones(5, 96, dtype=torch.bool)
    valid[2, ::3] = False
    valid[3] = False
    sequence[~valid] = float('nan')

    class Encoder(torch.nn.Module):
        method_name = 'physiology_only'
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def forward(self, raw):
            indices = [int(i) for i in raw.sample_ids]
            return SimpleNamespace(sequence_embedding=sequence[indices].to(self.weight.device),
                modality_available_mask=valid[indices].to(self.weight.device))

    class Normalizer:
        def transform(self, raw):
            return raw
        def to_manifest(self):
            return {}

    encoder, normalizer = Encoder().to(device), Normalizer()
    model = EndToEndApplicationModel(method_name=encoder.method_name, encoder=encoder,
        normalizer=normalizer, naive_encoder=None).to(device)
    roles = {role: getattr(fold, role + '_sample_ids') for role in ('train', 'validation', 'held_out')}
    checkpoint = tmp_path/'encoder.pt'
    torch.save(dict(format=FINETUNING_FORMAT, training_status='completed', label_used_for_encoder_training=True,
        role_sample_ids={k: list(v) for k, v in roles.items()}, normalizer={}, fold_id=fold.fold_id,
        model_state_dict=model.state_dict()), checkpoint)
    provider = lambda ids: select_observation_batch(batch, ids)
    guided = export_finetuned_application_representations(model=model, checkpoint_path=checkpoint,
        batch=batch, role_sample_ids=roles, output_root=tmp_path/'guided', batch_size=3)['validation']
    formal = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
        provider=provider, fold=fold, root=tmp_path/'formal', export_roles=('validation',),
        export_prefix='test', label_used_for_encoder_training=True, batch_size=2)['validation']
    pressure = _paired_representation(encoder, normalizer, checkpoint, fold,
        provider(fold.validation_sample_ids), tmp_path/'pressure', label_used_for_encoder_training=True)
    expected = sequence[1:].masked_fill(~valid[1:].unsqueeze(-1), 0)
    expected_pool = expected.sum(1) / valid[1:].sum(1, keepdim=True).clamp_min(1)
    for output in (guided, formal, pressure):
        assert torch.equal(output.sequence_embedding.cpu(), expected)
        assert torch.equal(output.pooled_embedding.cpu(), expected_pool)
        assert torch.equal(output.valid_mask.cpu(), valid[1:])
    assert torch.equal(pool_exported_sequence(sequence.to(device), valid.to(device)).cpu()[1:], expected_pool)
