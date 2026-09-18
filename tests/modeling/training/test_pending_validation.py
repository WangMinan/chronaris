from dataclasses import replace

import numpy as np
import pytest
import torch

from chronaris.modeling.training import candidate_screen as screen, EncoderCandidateConfig, load_common_pretraining_checkpoint
from chronaris.modeling.training.rng import canonical_training_state_sha256, isolated_training_rng
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples
from chronaris.evaluation.application_tasks import application_finetuning as fine
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from tests.modeling.training.test_candidate_screen import _sample


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pending_validation_resumes_both_trainers_without_duplicate_updates(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    samples = [_sample(f'sample_{i}', i) for i in range(7)]
    if device == 'cuda':
        samples = [replace(s, physiology_values=np.linspace(i, i+3, 513, dtype=np.float32)[:, None],
            physiology_timestamps_s=np.linspace(0., 20., 513), physiology_feature_mask=np.ones((513, 1), dtype=bool))
            for i,s in enumerate(samples)]
    batch = collate_observation_samples(samples)
    fold = FoldLineage(fold_id='pending', train_sample_ids=batch.sample_ids[:4],
                      validation_sample_ids=batch.sample_ids[4:6], held_out_sample_ids=batch.sample_ids[6:])
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids)
    config = screen.CandidateScreenConfig(max_updates=3, batch_size=2, effective_batch_size=4,
        validation_interval=2, checkpoint_interval=2, device=device, early_stopping=False,
        cuda_graph_recurrence=device == 'cuda', retained_updates=(2,))
    kwargs = dict(method_name='chronaris', candidate=EncoderCandidateConfig(candidate_id='C', hidden_dim=32),
        batch=batch, fold=fold, physiology_feature_names=('physiology.a',), vehicle_feature_names=('vehicle.a',),
        vehicle_field_labels=(), normalizer=normalizer, config=config, chronaris_fusion_kind='safe_lag')
    reference = screen.train_pretext_candidate(output_root=tmp_path/'pre_full', **kwargs)
    evaluate = screen.evaluate_candidate_mechanisms
    def interrupted(**kw):
        evaluate(**kw)
        raise RuntimeError('interrupted validation')
    monkeypatch.setattr(screen, 'evaluate_candidate_mechanisms', interrupted)
    with pytest.raises(RuntimeError, match='interrupted validation'):
        screen.train_pretext_candidate(output_root=tmp_path/'pre_resume', **kwargs)
    pending = torch.load(tmp_path/'pre_resume/chronaris/C/last.pt', map_location='cpu', weights_only=True)
    assert pending['optimizer_updates'] == 2 and pending['pending_validation']
    monkeypatch.setattr(screen, 'evaluate_candidate_mechanisms', evaluate)
    restored = screen.train_pretext_candidate(output_root=tmp_path/'pre_resume', **kwargs)
    def equal_checkpoints(a, b, keys):
        left, right = [torch.load(p, map_location='cpu', weights_only=True) for p in (a,b)]
        assert 'pending_validation' not in right
        for key in keys:
            assert canonical_training_state_sha256(left[key]) == canonical_training_state_sha256(right[key]), key
    equal_checkpoints(reference.last_checkpoint_path, restored.last_checkpoint_path,
        ('encoder_state_dict','head_state_dict','optimizer_state_dict','rng_state','data_cursor','epoch_rows','training_rows'))
    source = reference.best_checkpoint_path
    def model():
        encoder, _, norm, _ = load_common_pretraining_checkpoint(source)
        with isolated_training_rng(17):
            return fine.EndToEndApplicationModel(method_name='chronaris', encoder=encoder, normalizer=norm,
                naive_encoder=None, task_definitions=(ApplicationTaskDefinition('classify','classification',2),))
    targets = ApplicationTaskTargets(batch.sample_ids, {'classify': torch.arange(7).remainder(2)},
        {'classify': torch.ones(7, dtype=torch.bool)}, {'domain':'test'})
    kwargs = dict(batch=batch, targets=targets, source_checkpoint_path=source,
        role_sample_ids={r:getattr(fold,r+'_sample_ids') for r in ('train','validation','held_out')},
        config=fine.EndToEndFineTuningConfig(max_updates=3, head_warmup_updates=1, batch_size=2,
            effective_batch_size=4, validation_interval=2, device=device, early_stopping=False))
    reference = fine.train_end_to_end_application_method(model=model(), output_root=tmp_path/'fine_full', **kwargs)
    evaluate_fine = fine._evaluate_losses
    def interrupted_fine(**kw):
        evaluate_fine(**kw)
        raise RuntimeError('interrupted validation')
    monkeypatch.setattr(fine, '_evaluate_losses', interrupted_fine)
    with pytest.raises(RuntimeError, match='interrupted validation'):
        fine.train_end_to_end_application_method(model=model(), output_root=tmp_path/'fine_resume', **kwargs)
    pending = torch.load(tmp_path/'fine_resume/chronaris/last.pt', map_location='cpu', weights_only=True)
    assert pending['step_count'] == 3 and pending['pending_validation']
    monkeypatch.setattr(fine, '_evaluate_losses', evaluate_fine)
    restored = fine.train_end_to_end_application_method(model=model(), output_root=tmp_path/'fine_resume', **kwargs)
    equal_checkpoints(reference.last_checkpoint_path, restored.last_checkpoint_path,
        ('model_state_dict','optimizer_state_dict','rng_state','data_cursor','epoch_rows','update_rows'))


def test_first_joint_ordinary_execution_restores_configuration_on_failure():
    from chronaris.models.alignment.config import AlignmentPrototypeConfig
    from chronaris.models.alignment.prototype import SingleStreamODERNNPrototype
    from chronaris.models.alignment.cuda_recurrence import ordinary_recurrence
    model = SingleStreamODERNNPrototype(1, config=AlignmentPrototypeConfig(ode_method='euler', cuda_graph_recurrence=True))
    original = model.config
    with ordinary_recurrence(model, False):
        assert model.config is original
    with pytest.raises(RuntimeError, match='interrupt'):
        with ordinary_recurrence(model, True):
            assert not model.config.cuda_graph_recurrence
            raise RuntimeError('interrupt')
    assert model.config is original
