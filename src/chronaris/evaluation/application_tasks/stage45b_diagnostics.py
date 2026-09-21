"""Read-only normalization probes and isolated, weighted backbone gradients."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.application_tasks.application_finetuning import EndToEndApplicationModel
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.application_task_heads import (fit_application_task_parameters,
    select_application_targets, application_task_losses)
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.training import load_common_pretraining_checkpoint, CandidateScreenConfig
from chronaris.modeling.training.candidate_step import pretext_micro_step
from chronaris.modeling.training.pretext import ExplicitTimeShiftHead
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.models.alignment.cuda_recurrence import ordinary_recurrence
from chronaris.representation import AugmentationPolicy
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def feature_statistics(outputs, context):
    train = outputs['train'].pooled_embedding.numpy()
    scaler = StandardScaler().fit(train)
    result = {}
    for role, output in outputs.items():
        x = output.pooled_embedding.numpy()
        scaled = scaler.transform(x)
        groups = np.array([context['groups'][s] for s in output.sample_ids])
        result[role] = dict(raw_singular_values=np.linalg.svd(x-x.mean(0), compute_uv=False).tolist(),
            scaled_singular_values=np.linalg.svd(scaled-scaled.mean(0), compute_uv=False).tolist(),
            mean_norm=float(np.linalg.norm(x, axis=1).mean()),
            group_mean_shift={str(g): float(np.linalg.norm(scaled[groups == g].mean(0))) for g in sorted(set(groups))})
    return result


def normalization_trigger(probes):
    reasons = []
    for probe in probes:
        branches = probe['branches']
        for stream in ('physiology', 'vehicle'):
            keys = (stream+'_before_projection_common_mask', stream+'_after_normalization_common_mask')
            a, b = [{r['task']: r for r in branches[k]['components']['linear']['task_summary']['validation']} for k in keys]
            for task, row in a.items():
                before, after = row['value'], b[task]['value']
                if before is None or after is None:
                    continue
                if (before-after >= .02 if row['metric'] == 'macro_f1' else before > 0 and after/before >= 1.05):
                    reasons.append(dict(route=probe['route'], stream=stream, task=task, before=before, after=after,
                        checkpoint=probe['checkpoint']))
    return dict(enabled=bool(reasons), reasons=reasons, diagnostic_only=True,
        policy='same 32 dimensions and common query mask: F1 loss >= .02 or RMSE increase >= 5 percent')


def isolated_gradients(losses, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    vectors, records = {}, {}
    for name, loss in losses.items():
        if loss is None or not loss.requires_grad:
            records[name] = dict(status='unavailable', norm=None)
            continue
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        present = [g for g in grads if g is not None]
        vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
                            for p, g in zip(parameters, grads)])
        vectors[name] = vector
        records[name] = dict(status='measured' if present else 'disconnected', norm=float(vector.norm()), loss=float(loss.detach()))
    cosines = {}
    for i, (a, left) in enumerate(vectors.items()):
        for b, right in list(vectors.items())[i+1:]:
            denominator = float(left.norm()*right.norm())
            cosines[a+' / '+b] = float(torch.dot(left, right)/denominator) if denominator > 0 else None
    return dict(terms=records, cosines=cosines, scope='isolated_weighted_loss_gradients_on_shared_continuous_backbone')


def checkpoint_gradients(*, checkpoint, route, provider, fold, targets, definitions, output_root):
    with isolated_training_rng(17):
        encoder, normalizer, payload = load_frozen_application_encoder(checkpoint, route=route, fold=fold,
            device='cuda', allow_diagnostic_snapshot=True)
        source_path = payload['source_checkpoint_path'] if route == 'task_guided' else checkpoint
        _, heads, _, source = load_common_pretraining_checkpoint(source_path, device='cuda', allow_diagnostic_snapshot=True)
        model = EndToEndApplicationModel(method_name='chronaris', encoder=encoder, normalizer=normalizer,
            naive_encoder=None, task_definitions=definitions).cuda()
        if route == 'task_guided':
            model.task_heads.load_state_dict({n.removeprefix('task_heads.'): v for n,v in payload['model_state_dict'].items()
                                             if n.startswith('task_heads.')})
            heads.load_state_dict({n.removeprefix('pretext_heads.'): v for n,v in payload['model_state_dict'].items()
                                   if n.startswith('pretext_heads.')})
        else:
            # A fixed linear readout is fitted on frozen training features, never on validation labels.
            from chronaris.evaluation.application_tasks.checkpoint_selection import GroupedCheckpointSelector
            selector = GroupedCheckpointSelector(provider=provider, fold=fold, targets=targets, definitions=definitions,
                context=output_root['context'], output_root=Path(output_root['path'])/'readout')
            selected = selector(encoder, normalizer, 1)
            import joblib
            bundle = joblib.load(selected['consumer']['components']['linear']['model_path'])['consumer']
            parameters = fit_application_task_parameters(targets, definitions, fold.train_sample_ids)
            for task in definitions:
                estimator = bundle['models'][(task.name, 0)]
                scaler, head = estimator.steps[0][1], estimator.steps[-1][1]
                weight = np.atleast_2d(head.coef_) / scaler.scale_[None, :]
                bias = np.atleast_1d(head.intercept_) - weight @ scaler.mean_
                if task.kind == 'classification' and task.output_dim == 2:
                    weight = np.concatenate([np.zeros_like(weight), weight]); bias = np.concatenate([np.zeros_like(bias), bias])
                if task.kind == 'regression':
                    calibration = parameters['tasks'][task.name]
                    weight = weight / np.asarray(calibration['scale'])[:, None]
                    bias = (bias-np.asarray(calibration['center'])) / np.asarray(calibration['scale'])
                model.task_heads[task.name].weight.data.copy_(torch.as_tensor(weight, device='cuda'))
                model.task_heads[task.name].bias.data.copy_(torch.as_tensor(bias, device='cuda'))
        shift = None
        if source.get('chronaris_explicit_shift_enabled'):
            shift = ExplicitTimeShiftHead(64).cuda()
            state = ({n.removeprefix('explicit_shift_head.'):v for n,v in payload['model_state_dict'].items()
                      if n.startswith('explicit_shift_head.')} if route == 'task_guided' else source['explicit_time_shift_head_state_dict'])
            shift.load_state_dict(state)
        config = replace(CandidateScreenConfig(**source['config']), device='cuda')
        policy = AugmentationPolicy(**source['augmentation_policy'])
        updates = payload.get('optimizer_updates', 300) if route == 'self_supervised' else 200
        rows = []
        # Two fixed batches test repeatability without any optimizer updates.
        for offset in (0, 4):
            ids = fold.train_sample_ids[offset:offset+4]
            with ordinary_recurrence(model, True):
                model.eval()
                output = model(provider(ids))
                tasks = application_task_losses(output, select_application_targets(targets, ids, 'cuda'), definitions,
                    fit_application_task_parameters(targets, definitions, fold.train_sample_ids))
                public, mechanism, _, _ = pretext_micro_step(encoder=encoder, heads=heads, shift_head=shift,
                    batch=None, batch_provider=provider, sample_ids=ids, normalizer=normalizer, resolved=config,
                    policy=policy, method_name='chronaris', epoch=1, optimizer_updates=updates,
                    chronaris_lag_aware_weight=source.get('chronaris_lag_aware_weight', 0.),
                    chronaris_mechanism_enabled=source.get('chronaris_mechanism_enabled', False),
                    chronaris_explicit_shift_weight=source.get('chronaris_explicit_shift_weight', 0.),
                    chronaris_event_pair_weight=source.get('chronaris_event_pair_weight', 0.))
                losses = {t.name: tasks[t.name]/len(definitions) for t in definitions}
                losses['public'] = public.total_loss * (.2 if route == 'task_guided' else 1.)
                losses.update(mechanism.weighted_terms)
                rows.append(dict(sample_ids=ids, schedule_update=updates, mechanisms=list(mechanism.rows),
                    **isolated_gradients(losses, encoder.backbone.continuous_backbone.parameters())))
        return write_result(Path(output_root['path'])/'gradients.json', dict(status='completed', checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint), route=route, batches=rows, optimizer_steps=0,
            task_readout='trained_task_heads' if route == 'task_guided' else 'training_fitted_common_linear_readout',
            confirmation_opened=False))
