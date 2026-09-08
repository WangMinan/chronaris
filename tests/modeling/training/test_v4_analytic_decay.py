import pytest
import torch

from chronaris.modeling.training import candidate_screen as screen
from chronaris.modeling.training import EncoderCandidateConfig, load_common_pretraining_checkpoint
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.models.alignment.ode_cells import ODERNNCell
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples
from tests.modeling.training.test_candidate_screen import _sample


def test_analytic_decay_is_bounded_composable_and_has_trainable_positive_rates():
    cell = ODERNNCell(4, hidden_dim=4, dynamics_hidden_dim=8, ode_method='analytic_decay')
    state = torch.tensor([[1., -2., 3., -4.]], requires_grad=True)
    a = cell.evolve_hidden_state(state, torch.tensor([2.]))
    b = cell.evolve_hidden_state(a, torch.tensor([5.]))
    torch.testing.assert_close(b, cell.evolve_hidden_state(state, torch.tensor([7.])))
    assert torch.equal(state, cell.evolve_hidden_state(state, torch.zeros(1)))
    long = cell.evolve_hidden_state(state, torch.tensor([1e9]))
    assert torch.isfinite(long).all() and torch.all(long.abs() <= state.abs())
    a.square().sum().backward()
    assert cell.decay_logits.grad.abs().sum() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_analytic_decay_training_resume_and_checkpoint_loading(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    samples = [_sample(f'sample_{i}', i) for i in range(6)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage('decay_updates', batch.sample_ids[:4], batch.sample_ids[4:], (), development_only=True)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
                                                held_out_sample_ids=fold.validation_sample_ids)
    config = screen.CandidateScreenConfig(max_updates=4, batch_size=2, effective_batch_size=4,
        checkpoint_interval=1, validation_interval=2, early_stopping=False, ode_method='analytic_decay', device=device)
    kwargs = dict(method_name='chronaris', candidate=EncoderCandidateConfig(hidden_dim=8), batch=batch, fold=fold,
        physiology_feature_names=samples[0].schema.physiology_feature_names,
        vehicle_feature_names=samples[0].schema.vehicle_feature_names, vehicle_field_labels=(),
        normalizer=normalizer, config=config, chronaris_fusion_kind='safe_lag')
    complete = screen.train_pretext_candidate(output_root=tmp_path/'complete', **kwargs)
    original = screen.pretext_micro_step
    calls = 0
    def interrupt(**args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('interrupted accumulated update')
        return original(**args)
    monkeypatch.setattr(screen, 'pretext_micro_step', interrupt)
    with pytest.raises(RuntimeError, match='accumulated update'):
        screen.train_pretext_candidate(output_root=tmp_path/'resumed', **kwargs)
    monkeypatch.setattr(screen, 'pretext_micro_step', original)
    resumed = screen.train_pretext_candidate(output_root=tmp_path/'resumed', **kwargs)
    a = torch.load(complete.last_checkpoint_path, map_location='cpu', weights_only=True)
    b = torch.load(resumed.last_checkpoint_path, map_location='cpu', weights_only=True)
    for name in ('encoder_state_dict', 'optimizer_state_dict', 'rng_state', 'data_cursor', 'training_rows'):
        assert canonical_training_state_sha256(a[name]) == canonical_training_state_sha256(b[name]), name
    encoder, _, _, _ = load_common_pretraining_checkpoint(resumed.last_checkpoint_path, device=device)
    assert encoder.backbone.config.ode_method == 'analytic_decay'
    assert any('decay_logits' in key for key in encoder.state_dict())
