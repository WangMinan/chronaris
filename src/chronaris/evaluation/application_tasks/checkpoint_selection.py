"""Prospective grouped, label-assisted checkpoint selection with shared consumers."""
from dataclasses import replace
import hashlib
import time
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_task_heads import select_application_targets
from chronaris.evaluation.application_tasks.application_finetuning import _task_target_sha256
from chronaris.evaluation.application_tasks.v4_grouped_consumers import run_native_method_consumers
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation.contracts import pool_exported_sequence
from chronaris.representation.window_features import WindowFeatureBatch


def subset_targets(targets, ids):
    return replace(targets, sample_ids=tuple(ids), **select_application_targets(targets, ids, 'cpu'))


def selection_split(fold, targets, definitions, context, *, inner_index=0):
    """Two reciprocal CLARE splits; a fixed two-subject holdout for CogPilot."""
    groups = sorted({context['groups'][s] for s in fold.train_sample_ids})
    if context['domain'] == 'clare':
        if len(groups) != 2 or inner_index not in (0, 1):
            raise ValueError('CLARE requires the two fixed reciprocal training-subject splits')
        selected = {groups[inner_index]}
    elif context['domain'] == 'cogpilot':
        if len(groups) < 4 or inner_index != 0:
            raise ValueError('CogPilot requires at least four training subjects and split zero')
        selected = set(groups[:2])
    else:
        raise ValueError('checkpoint selection only supports public development')
    train = tuple(s for s in fold.train_sample_ids if context['groups'][s] not in selected)
    selection = tuple(s for s in fold.train_sample_ids if context['groups'][s] in selected)
    name = fold.fold_id + f'__selection{inner_index}'
    training_fold = replace(fold, fold_id=name, train_sample_ids=train, validation_sample_ids=selection)
    evaluation_fold = replace(fold, fold_id=name, train_sample_ids=train)
    index = {s: i for i, s in enumerate(targets.sample_ids)}
    support = {}
    for role, ids in (('train', train), ('selection', selection)):
        support[role] = {}
        for task in definitions:
            positions = [index[s] for s in ids]
            values = targets.values[task.name][positions]
            valid = targets.valid_masks[task.name][positions]
            selected_values = values[valid]
            if not len(selected_values):
                raise ValueError(f'{role} lacks valid targets: {task.name}')
            if task.kind == 'classification' and set(selected_values.tolist()) != set(range(task.output_dim)):
                raise ValueError(f'{role} lacks declared classes: {task.name}; no window-level fallback')
            support[role][task.name] = dict(count=len(selected_values), classes=sorted(set(selected_values.tolist()))
                if task.kind == 'classification' else None)
    return training_fold, evaluation_fold, support


class GroupedCheckpointSelector:
    def __init__(self, *, provider, fold, targets, definitions, context, output_root, seed=17):
        self.provider, self.fold = provider, fold
        self.targets = subset_targets(targets, fold.train_sample_ids + fold.validation_sample_ids)
        self.definitions, self.context = definitions, context
        self.root, self.seed = Path(output_root), seed
        train_groups = {context['groups'][s] for s in fold.train_sample_ids}
        select_groups = {context['groups'][s] for s in fold.validation_sample_ids}
        if train_groups & select_groups:
            raise ValueError('checkpoint-selection subjects overlap training')
        index = {s: i for i, s in enumerate(self.targets.sample_ids)}
        positions = [index[s] for s in fold.train_sample_ids]
        self.scales = {}
        for task in definitions:
            if task.kind == 'regression':
                values, valid = self.targets.values[task.name][positions], self.targets.valid_masks[task.name][positions]
                scale = float(values[valid].std(unbiased=False))
                if not scale > 1e-8:
                    raise ValueError('checkpoint selection requires nonconstant training regression targets')
                self.scales[task.name] = scale
        from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
        self.manifest = dict(strategy='grouped_common_linear_v1', fold=fold.to_dict(), source_code_sha256=v4_workflow_source_sha256(),
            target_sha256=_task_target_sha256(self.targets), regression_scales=self.scales,
            classification_weight=.5, regression_weight=.5, tie_break='earliest_update',
            gradient_supervision='declared_per_route', selection_supervision='summary_labels',
            consumer='existing_linear_grid_and_train_fitted_StandardScaler', seed=seed,
            selection_subjects=sorted(select_groups), training_subjects=sorted(train_groups))

    def __call__(self, encoder, normalizer, update):
        started = time.perf_counter()
        if set(normalizer.fit_sample_ids) != set(self.fold.train_sample_ids):
            raise ValueError('selection normalizer must fit only encoder-training samples')
        states = {module: module.training for module in encoder.modules()}
        digest = hashlib.sha256()
        for name, value in encoder.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        fingerprint = digest.hexdigest()
        outputs = {}
        try:
            with isolated_training_rng(self.seed), torch.inference_mode():
                encoder.eval()
                device = next(encoder.parameters()).device
                for role in ('train', 'validation'):
                    ids = getattr(self.fold, role+'_sample_ids')
                    values, validities, durations, hashes = [], [], [], []
                    for offset in range(0, len(ids), 4):
                        raw = self.provider(ids[offset:offset+4])
                        output = encoder(move_observation_batch(normalizer.transform(raw), device=device))
                        valid = output.modality_available_mask.cpu()
                        values.append(pool_exported_sequence(output.sequence_embedding.cpu(), valid))
                        validities.append(valid.any(1)); durations.append(raw.context_durations_s)
                        hashes.extend(raw.source_sample_hashes)
                    outputs[role] = WindowFeatureBatch(ids, torch.cat(durations), torch.cat(values), torch.cat(validities),
                        encoder.method_name, self.fold.fold_id, fingerprint, tuple(hashes))
            result = run_native_method_consumers(outputs=outputs, targets=self.targets, definitions=self.definitions,
                context=self.context | {'checkpoint_selection': self.manifest}, output_root=self.root/f'update_{update:06d}',
                label_used_for_encoder_training=self.root.name == 'task_guided', seed=self.seed, families=('linear',))
            rows = result['components']['linear']['task_summary']['validation']
            losses = [1-r['value'] if r['metric'] == 'macro_f1' else r['value']/self.scales[r['task']] for r in rows]
            score = sum(losses)/len(losses)
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError('nonfinite checkpoint selection score')
            return dict(score=score, metrics=rows, consumer=result, manifest=self.manifest, elapsed_s=time.perf_counter()-started)
        finally:
            for module, training in states.items():
                module.training = training
