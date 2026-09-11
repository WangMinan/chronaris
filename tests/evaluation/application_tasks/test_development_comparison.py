from pathlib import Path
import shutil

import pytest
import torch
from torch import nn

from chronaris.evaluation.application_tasks.development_comparison import comparison_budget, comparison_units
from chronaris.evaluation.application_tasks.v4_pipeline import pipeline_groups
from chronaris.evaluation.application_tasks.recent_model_training import short_fit
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS


def test_comparison_is_complete_development_only_and_recent_budgets_cover_training():
    stages = [s for group in pipeline_groups('comparison') for s, _ in group]
    assert len(comparison_units()) == 25 and len(set(stages)) == 28
    assert stages[0:2] == ['cuda_validation', 'comparison_plan'] and stages[-1] == 'comparison_costs'
    assert not any(s in stages for g in pipeline_groups('confirmation')[2:] for s, _ in g)
    assert [u['domain'] for u in comparison_units() if u['method']=='sensorllm_deepseek'] == ['clare']
    for count in (24, 500, 2524):
        for method in ('timecma', 'chronos2', 'sensorllm_deepseek'):
            budget = comparison_budget(method, count)
            assert budget['adaptation_updates'] * (4 if method=='timecma' else 1) >= count
            assert budget['adaptation_updates'] >= 200 and budget['replay_updates'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA adapter resume required')
def test_cuda_full_adapter_schedule_preserves_every_window_and_exact_last_step_resume(tmp_path):
    class Encoder(nn.Module):
        feature_dim = 3
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 3).cuda()
            self.dropout = nn.Dropout(.2)
        def forward(self, values, mask, prompts=None):
            return self.dropout(self.layer(values.cuda()))
    ids = tuple(f's{i}' for i in range(23))
    definitions = PUBLIC_TASKS['clare']
    targets = ApplicationTaskTargets(ids,
        {'workload_classification': torch.arange(23) % 2, 'workload_regression': torch.arange(23.)},
        {'workload_classification': torch.arange(23) < 9, 'workload_regression': torch.arange(23) >= 9}, {})
    with isolated_training_rng(17):
        encoder = Encoder()
        initial = encoder.layer.weight.detach().clone()
        kwargs = dict(values=torch.randn(23, 3), mask=torch.ones(23, 3, dtype=torch.bool), prompts=None,
            targets=targets, definitions=definitions, train_ids=ids,
            metadata={'method_name':'timecma', 'target_supervision':'summary_labels',
                      'full_training_role':True, 'checkpoint_interval':25, 'training_budget':7})
        short_fit(encoder, **kwargs, root=tmp_path/'primary', stop_after=6)
        (tmp_path/'replay').mkdir()
        shutil.copyfile(tmp_path/'primary/training.pt', tmp_path/'replay/training.pt')
        primary = short_fit(encoder, **kwargs, root=tmp_path/'primary', stop_after=7)
        replay = short_fit(Encoder(), **kwargs, root=tmp_path/'replay', stop_after=7)
        assert primary['state_sha256'] == replay['state_sha256'] and primary['history'] == replay['history']
        assert {s for h in primary['history'] for s in h['sample_ids']} == set(ids)
        assert not torch.equal(initial, encoder.layer.weight)
        assert all(sum(h['task_counts'][t.name] for h in primary['history']) > 0 for t in definitions)
        assert all(h['parameters_with_gradient'] > 0 for h in primary['history'])
        assert torch.load(tmp_path/'primary/training.pt', weights_only=True)['update'] == 7
