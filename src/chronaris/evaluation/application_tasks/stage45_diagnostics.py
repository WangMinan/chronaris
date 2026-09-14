"""Branch probes and fixed-checkpoint consumers, never used as a new main ranking."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_finetuning_export import (
    load_frozen_application_encoder, export_loaded_application_encoder)
from chronaris.evaluation.application_tasks.common_downstream_contract import run_common_downstream
from chronaris.evaluation.application_tasks.v4_grouped_consumers import run_native_method_consumers
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation.contracts import pool_exported_sequence
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def branch_features(output):
    alignment, fusion = output.auxiliary['alignment_output'], output.auxiliary['fusion_output']
    result = {}
    for stream in ('physiology', 'vehicle'):
        state = getattr(alignment, stream)
        result[stream+'_before_projection'] = (state.reference_hidden_states, state.reference_valid_mask)
        result[stream+'_after_projection'] = (getattr(fusion, stream+'_private'), state.reference_valid_mask)
    result['branches_concatenated'] = (torch.cat((fusion.physiology_private, fusion.vehicle_private), -1),
                                       output.modality_available_mask)
    result['full'] = (output.sequence_embedding, output.modality_available_mask)
    return result


def run_branch_probes(*, checkpoint, route, provider, fold, targets, definitions, context, output_root, seed=17, allow_diagnostic_snapshot=False):
    encoder, normalizer, payload = load_frozen_application_encoder(checkpoint, route=route, fold=fold, device='cuda',
        allow_diagnostic_snapshot=allow_diagnostic_snapshot)
    encoder.eval()
    collected, gates = {}, []
    for role in ('train', 'validation'):
        ids = getattr(fold, role+'_sample_ids')
        pieces, hashes, times = {}, [], []
        for offset in range(0, len(ids), 4):
            raw = provider(ids[offset:offset+4])
            with torch.inference_mode():
                output = encoder(move_observation_batch(normalizer.transform(raw), device='cuda'))
                for name, (sequence, valid) in branch_features(output).items():
                    # The same CPU pooling as the main exported representation.
                    values = pool_exported_sequence(sequence.cpu(), valid.cpu())
                    pieces.setdefault(name, []).append((values, valid.any(1).cpu()))
                fusion = output.auxiliary['fusion_output']
                available = fusion.scale_available_mask.any(-1) & output.auxiliary['alignment_output'].physiology.reference_valid_mask
                if available.any():
                    gates.append(fusion.cross_gate[available].detach().cpu().flatten())
            hashes.extend(raw.source_sample_hashes)
            times.append(raw.context_durations_s)
        for name, rows in pieces.items():
            collected.setdefault(name, {})[role] = WindowFeatureBatch(ids, torch.cat(times),
                torch.cat([r[0] for r in rows]), torch.cat([r[1] for r in rows]),
                'chronaris', fold.fold_id, sha256_file(checkpoint), tuple(hashes))
    results = {}
    for name, outputs in collected.items():
        results[name] = run_native_method_consumers(outputs=outputs, targets=targets, definitions=definitions,
            context=context | {'diagnostic_branch': name, 'checkpoint_sha256': sha256_file(checkpoint)},
            output_root=Path(output_root)/name, label_used_for_encoder_training=route == 'task_guided', seed=seed,
            families=('linear',))
    gate_values = torch.cat(gates) if gates else torch.empty(0)
    result = dict(status='completed', route=route, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        branches=results, gate_count=len(gate_values), source_training_status=payload['training_status'],
        source_optimizer_updates=payload.get('optimizer_updates'),
        gate_quantiles=gate_values.quantile(torch.tensor([0., .1, .5, .9, 1.])).tolist() if gates else [],
        diagnostic_only=True, dimensions={k: v['train'].pooled_embedding.shape[1] for k,v in collected.items()},
        confirmation_opened=False)
    del encoder, output, collected
    gc.collect(); torch.cuda.empty_cache()
    return write_result(Path(output_root)/'summary.json', result)


def checkpoint_diagnostics(*, root, domain, provider, fold, contract, targets, definitions, context,
                           observations, digest, results):
    records = []
    for result in results:
        if 'training' not in result or result.get('training') is None:
            continue
        route, training = result['route'], result['training']
        last = Path(training['last_checkpoint_path'])
        payload = torch.load(last, map_location='cpu', weights_only=True)
        rows = payload.get('training_rows', payload.get('update_rows', []))
        terms = {}
        for row in rows:
            for term in [row] if 'term_name' in row else row.get('mechanism_terms', []):
                name = term['term_name']
                item = terms.setdefault(name, dict(effective_count=0, weighted_loss_sum=0., max_related_gradient=0.))
                if term.get('weight', 0) > 0:
                    item['effective_count'] += term.get('count', 0)
                    item['weighted_loss_sum'] += term.get('weighted_loss') or 0.
                item['max_related_gradient'] = max(item['max_related_gradient'], term.get('related_parameter_gradient_norm') or 0.)
        record = dict(route=route, best_update=payload.get('best_update'), terms=terms,
            best_checkpoint=str(training['best_checkpoint_path']), last_checkpoint=str(last),
            gradient_scope='related_parameters_under_combined_objective', snapshots=[])
        common_path = root/result['method']/route/'evaluation/common_contract.json'
        declaration = json.loads(common_path.read_text())['declaration']
        pattern = 'joint_update_*.pt' if route == 'task_guided' else 'update_*.pt'
        for checkpoint in sorted(last.parent.glob(pattern)):
            encoder, normalizer, _ = load_frozen_application_encoder(checkpoint, route=route, fold=fold,
                device='cuda', allow_diagnostic_snapshot=True)
            directory = root/'checkpoint_diagnostics'/route/checkpoint.stem
            outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
                provider=provider, fold=fold, root=directory/'representations', export_roles=('train', 'validation'),
                export_prefix='stage45_snapshot', label_used_for_encoder_training=route == 'task_guided')
            declared = declaration | dict(checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
                evidence_files={str(checkpoint.resolve()): sha256_file(checkpoint), str(Path(__file__).resolve()): sha256_file(__file__)})
            evaluated = run_common_downstream(contract=contract, outputs=outputs, declaration=declared,
                targets=targets, definitions=definitions, context=context, observations=observations,
                fold=fold, data_manifest_sha256=digest, output_root=directory/'evaluation', families=('linear',))
            record['snapshots'].append(dict(checkpoint=str(checkpoint), evaluation=evaluated))
            del encoder, outputs
            gc.collect(); torch.cuda.empty_cache()
        records.append(record)
    return write_result(root/'checkpoint_diagnostics/summary.json', dict(status='completed', records=records,
        diagnostic_only=True, checkpoint_selection_unchanged=True, confirmation_opened=False))


def fidelity_trigger(probes):
    """Any meaningful projection loss in public tasks opens the predeclared candidate."""
    reasons = []
    for probe in probes:
        branches = probe['branches']
        for stream in ('physiology', 'vehicle'):
            def scores(name):
                return {r['task']: r for r in branches[name]['components']['linear']['task_summary']['validation']}
            before, after = scores(stream+'_before_projection'), scores(stream+'_after_projection')
            for task, row in before.items():
                a, b = row['value'], after[task]['value']
                if a is None or b is None:
                    continue
                if (a-b >= .02 if row['metric'] == 'macro_f1' else a > 0 and b/a >= 1.05):
                    reasons.append(dict(route=probe['route'], stream=stream, task=task, before=a, after=b))
    return dict(enabled=bool(reasons), reasons=reasons, policy='F1 loss >= 0.02 or RMSE increase >= 5 percent')
