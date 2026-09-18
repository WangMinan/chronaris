"""Real short-entry execution comparison, plus independent interrupted guided resume."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values
from chronaris.evaluation.application_tasks.common_downstream_smoke import (
    contract_development_inputs, run_common_contract_smoke)
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, train_end_to_end_application_method)
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.training.pretraining_encoders import TrainableFusionEncoder
from chronaris.modeling.training.rng import isolated_training_rng


@contextmanager
def execution_trace(root):
    """Diagnostic instrumentation only; retain optimizer-input gradients after the existing clip and every native forward."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    original_forward, original_step = TrainableFusionEncoder.forward, torch.optim.AdamW.step
    counters = dict(forward=0, step=0)
    def forward(self, *args, **kwargs):
        out = original_forward(self, *args, **kwargs)
        torch.save(dict(sequence=out.sequence_embedding.detach().cpu(), mask=out.modality_available_mask.cpu()),
                   root/f'forward_{counters["forward"]:05d}.pt')
        counters['forward'] += 1
        return out
    def step(self, *args, **kwargs):
        parameters = [p for g in self.param_groups for p in g['params']]
        gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]
        result = original_step(self, *args, **kwargs)
        torch.save(dict(gradients=gradients, parameters=[p.detach().cpu() for p in parameters],
                        optimizer=self.state_dict()), root/f'step_{counters["step"]:05d}.pt')
        counters['step'] += 1
        return result
    TrainableFusionEncoder.forward, torch.optim.AdamW.step = forward, step
    try:
        yield
    finally:
        TrainableFusionEncoder.forward, torch.optim.AdamW.step = original_forward, original_step


def performance_entry(config, domain, recipe, mode):
    root = Path(config['root'])/'performance'/domain/recipe/mode
    if mode in ('eager', 'graph'):
        with execution_trace(root/'trace'):
            return run_common_contract_smoke(domain=domain, output_root=root, methods=('chronaris',),
                recipe=recipe, cuda_graph_recurrence=mode == 'graph',
                data_root=config['data_root'], registry_path=config['registry_path'])
    from chronaris.evaluation.application_tasks import application_finetuning as guided_module
    graph_root = root.parent/'graph'/domain
    summary = json.loads((graph_root/'summary.json').read_text())
    source = next(r['training']['best_checkpoint_path'] for r in summary['results']
                  if r.get('route') == 'self_supervised' and 'training' in r)
    provider, _, fold, digest, targets, definitions, _, _ = contract_development_inputs(domain,
        data_root=config['data_root'], registry_path=config['registry_path'])
    encoder, normalizer, _ = load_frozen_application_encoder(source, route='self_supervised', fold=fold, device='cuda')
    calibration = json.loads((graph_root/'recipe.json').read_text())['pretraining']['physics_calibration']
    _, _, guided, _, _ = training_recipe(recipe, method='chronaris', full=False, seed=17,
        digest=digest, train_count=len(fold.train_sample_ids), graph=True, calibration=calibration)
    guided = replace(guided, checkpoint_interval=1)
    with isolated_training_rng(17):
        model = EndToEndApplicationModel(method_name='chronaris', encoder=encoder, normalizer=normalizer,
                                        naive_encoder=None, task_definitions=definitions)
    resume_root = root.parent/'interrupted_guided'
    original_save = guided_module._atomic_save
    class ExpectedInterruption(Exception):
        pass
    def save(path, payload):
        original_save(path, payload)
        if mode == 'interrupt' and Path(path).name == 'last.pt' and payload.get('step_count') == 3:
            raise ExpectedInterruption('saved first joint update before independent restart')
    guided_module._atomic_save = save
    try:
        trained = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider, targets=targets,
            role_sample_ids={r: getattr(fold, r+'_sample_ids') for r in ('train', 'validation', 'held_out')},
            source_checkpoint_path=source, output_root=resume_root, config=guided)
    except ExpectedInterruption:
        return dict(status='completed', expected_interruption=True, confirmation_opened=False)
    finally:
        guided_module._atomic_save = original_save
    reference = next(r['training']['last_checkpoint_path'] for r in summary['results']
                     if r.get('route') == 'task_guided' and 'training' in r)
    left, right = [torch.load(p, map_location='cpu', weights_only=True) for p in (reference, trained.last_checkpoint_path)]
    checks = {k: compare_values(left[k], right[k], atol=0., rtol=0.) for k in
              ('model_state_dict', 'optimizer_state_dict', 'rng_state', 'data_cursor', 'update_rows', 'epoch_rows')}
    result = dict(status='completed', passed=all(c['close'] for c in checks.values()), checks=checks,
                  confirmation_opened=False)
    return write_result(root.parent/'resume_check.json', result)


def compare_execution(root, *, revised=False):
    if revised:
        from chronaris.evaluation.application_tasks.execution_equivalence import POLICY, compare_runtime
    root = Path(root)
    left, right = root/'eager/trace', root/'graph/trace'
    paths = sorted(p.name for p in left.glob('*.pt'))
    if not paths or paths != sorted(p.name for p in right.glob('*.pt')):
        raise ValueError('execution trace incomplete')
    checks = {}
    for name in paths:
        a, b = [torch.load(p/name, map_location='cpu', weights_only=True) for p in (left, right)]
        if revised:
            checks[name] = compare_runtime(a, b, representation=name.startswith('forward'))
        else:
            checks[name] = compare_values(a, b, rtol=0. if name.startswith('forward') else 1e-5)
    resume = json.loads((root/'resume_check.json').read_text())
    impact = fixed_consumer_impact(root) if revised else None
    return dict(status='completed', passed=all(c['close'] for c in checks.values()) and resume['passed']
                and (impact is None or impact['passed']),
        checks=checks, independent_resume=resume,
        **({'execution_policy': POLICY, 'consumer_impact': impact} if revised else dict(output_atol=1e-6, state_atol=1e-6, state_rtol=1e-5)),
        interpretation='short native entry only; full-unit costs require completed screen units',
        confirmation_opened=False)


def fixed_consumer_impact(root):
    """Apply the SAME fitted consumers to both exports; never refit to hide drift."""
    import joblib
    import numpy as np
    from chronaris.representation import load_fusion_stream_batch
    from chronaris.evaluation.application_tasks.execution_equivalence import POLICY
    root = Path(root)
    paths = list((root/'eager').glob('*/summary.json'))
    if len(paths) != 1:
        raise ValueError('fixed consumer comparison requires one complete native entry')
    left = json.loads(paths[0].read_text())
    right = json.loads((root/'graph'/paths[0].parent.name/'summary.json').read_text())
    routes = lambda d: {r['route']: r for r in d['results'] if 'training' in r}
    a, b = routes(left), routes(right)
    if set(a) != {'self_supervised', 'task_guided'} or a.keys() != b.keys():
        raise ValueError('execution comparison lacks both routes')
    checks = {}
    for route, row in a.items():
        outputs = [load_fusion_stream_batch(root/m/left['domain']/row['method']/route/'representations/validation')
                   for m in ('eager','graph')]
        if (outputs[0].sample_ids != outputs[1].sample_ids
            or not torch.equal(outputs[0].valid_mask, outputs[1].valid_mask)):
            raise ValueError('execution export roles or masks changed')
        selected = [r[route]['training'] for r in (a,b)]
        checks[f'{route}/selection'] = dict(passed=all(selected[0].get(k) == selected[1].get(k)
            for k in ('best_update','best_epoch')), values=[{k:t.get(k) for k in ('best_update','best_epoch')} for t in selected])
        for family, component in row['consumers']['components'].items():
            bundle = joblib.load(component['model_path'])['consumer']
            transformer = bundle['transformer']
            x = [(o.pooled_embedding.numpy() if transformer is None else
                  transformer.transform_features(o.sequence_embedding.numpy())) for o in outputs]
            kinds = {t.name:t.kind for t in bundle['definitions']}
            for (task, field), model in bundle['models'].items():
                p, q = [model.predict(v).astype(np.float64) for v in x]
                if not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise ValueError('nonfinite fixed consumer prediction')
                if kinds[task] == 'classification':
                    changed = int(np.count_nonzero(p != q))
                    check = dict(passed=changed == POLICY['classification_changes'], changed_predictions=changed)
                else:
                    scale = max(float(np.sqrt(np.mean(p**2))), 1e-3)
                    rms = float(np.sqrt(np.mean((p-q)**2)))
                    check = dict(passed=rms <= POLICY['prediction_atol'] + POLICY['prediction_relative_rms']*scale,
                                 relative_rms=rms/scale, rms=rms, scale=scale,
                                 max_abs=float(np.max(np.abs(p-q))))
                checks[f'{route}/{family}/{task}/{field}'] = check
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                interpretation='fixed eager consumers on identical development validation windows')
