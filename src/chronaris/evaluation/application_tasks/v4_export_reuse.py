"""Read-only compatibility inventory for the frozen v4 initial export repair."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from chronaris.evaluation.application_tasks.v4_candidate_results import _completed_scores, _pressure_p95
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.representation import FoldLineage, load_fusion_stream_batch
from chronaris.representation.contracts import pool_exported_sequence
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _definitions(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f'{node.name}.{child.name}'] = ast.dump(child, include_attributes=False)
    return result


def audit_initial_export_reuse(*, run_root, frozen_project, output_root):
    """Verify original lineage and decide reuse; never rewrite or resume the old run."""
    run, frozen, output = (Path(p).resolve() for p in (run_root, frozen_project, output_root))
    if output == run or run in output.parents or output == frozen or frozen in output.parents:
        raise ValueError('reuse audit requires a separate output root')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    project = Path(__file__).parents[4]
    source_digest = v4_workflow_source_sha256()
    current_files = {p.relative_to(project): p for p in (project/'src/chronaris').rglob('*.py')}
    old_files = {p.relative_to(frozen): p for p in (frozen/'src/chronaris').rglob('*.py')}
    differences = []
    for name in sorted(current_files.keys() | old_files.keys()):
        before, after = old_files.get(name), current_files.get(name)
        if before and after and sha256_file(before) == sha256_file(after):
            continue
        old_defs, new_defs = _definitions(before) if before else {}, _definitions(after) if after else {}
        differences.append(dict(path=str(name), before_sha256=sha256_file(before) if before else None,
            after_sha256=sha256_file(after) if after else None,
            changed_definitions=[k for k in sorted(old_defs.keys() | new_defs.keys()) if old_defs.get(k) != new_defs.get(k)]))
    # This audit is intentionally specific to this repair, not a general source-compatibility bypass.
    allowed = {
        'src/chronaris/representation/contracts.py': {'pool_exported_sequence'},
        'src/chronaris/modeling/training/common_pretraining.py': {'TrainedFusionAdapter.__call__'},
        'src/chronaris/evaluation/application_tasks/application_finetuning_export.py': {'export_finetuned_application_representations'},
        'src/chronaris/evaluation/application_tasks/v4_pipeline_steps.py': {'run_pipeline_step'},
        'src/chronaris/evaluation/application_tasks/v4_pipeline.py': {'main'},
    }
    for row in differences:
        if row['path'] == str(Path(__file__).relative_to(project)):
            continue
        if row['path'] not in allowed or not set(row['changed_definitions']) <= allowed[row['path']]:
            raise ValueError(f"unreviewed source change needs a new compatibility decision: {row['path']}")
    inventory = {}
    def bind(path):
        path = str(Path(path).resolve())
        inventory[path] = sha256_file(path)
        return inventory[path]
    cohort = json.loads((run/'initial/cohort_state.json').read_text())
    config = json.loads((run/'pipeline_config.json').read_text())
    old_digest = hashlib.sha256()
    for path in sorted(old_files.values()) + [frozen/'scripts/evaluation/application_tasks/run_thesis_v4.py']:
        old_digest.update(str(path.relative_to(frozen)).encode())
        old_digest.update(path.read_bytes())
    if old_digest.hexdigest() != cohort['source_code_sha256'] or config['source_code_sha256'] != cohort['source_code_sha256']:
        raise ValueError('frozen source no longer matches the original run')
    for path, digest in config['input_files'].items():
        if bind(path) != digest:
            raise ValueError(f'original input changed: {path}')
    # Bind all old evidence and checkpoints, including failed attempts, before reading any model.
    for name in ('initial', 'initial_pressure'):
        for path in (run/name).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.npz', '.pt', '.joblib', '.log'}:
                bind(path)
    for name in ('pipeline_state.json', 'pipeline_config.json'):
        bind(run/name)
    records, pressure_records = [], []
    for method, candidate in cohort['units']:
        unit = run/'initial/simulation'/method/candidate
        state = json.loads((unit/'run_state.json').read_text())
        if state['source_code_sha256'] != cohort['source_code_sha256'] or state['confirmation_opened']:
            raise ValueError('initial source or role changed')
        fold = FoldLineage(fold_id=state['fold']['fold_id'],
            **{role+'_sample_ids': tuple(state['fold'][role+'_sample_ids']) for role in ('train', 'validation', 'held_out')},
            development_only=state['fold'].get('development_only', False))
        if fold.to_dict() != state['fold']:
            raise ValueError('stored fold hashes changed')
        for route, update in (('self_supervised', 300), ('task_guided', 200)):
            scores = _completed_scores(unit, state, route, update, method, candidate)
            checkpoint = state[route+'_training']['best_checkpoint_path']
            encoder, normalizer, payload = load_frozen_application_encoder(checkpoint, route=route,
                fold=fold, device='cpu', allow_diagnostic_snapshot=True)
            nonfinite = [name for name, value in encoder.state_dict().items()
                if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()]
            if any(not name.endswith('._float_tensor') for name in nonfinite):
                raise ValueError('non-finite encoder parameters require investigation')
            result = json.loads((unit/f'{route}_{update}_consumers.json').read_text())
            manifest = result['model_manifest']
            consumer = joblib.load(manifest['model_files']['linear']['path'])['consumer']
            roles = {}
            directory = unit/'representations'/route/str(update)
            if route == 'task_guided':
                directory /= method
            for role in ('train', 'validation'):
                batch = load_fusion_stream_batch(directory/role)
                masked = batch.sequence_embedding.masked_fill(~batch.valid_mask.unsqueeze(-1), 0)
                pooled = pool_exported_sequence(masked, batch.valid_mask)
                roles[role] = dict(samples=len(batch.sample_ids), archive=str(directory/role/'fusion_stream.npz'),
                    sequence_equal=torch.equal(masked, batch.sequence_embedding),
                    pooled_equal=torch.equal(pooled, batch.pooled_embedding),
                    pooled_max_delta=float((pooled-batch.pooled_embedding).abs().max()))
                if role == 'validation':
                    prediction = consumer.predict(pooled.numpy())
                    with np.load(manifest['prediction_path'], allow_pickle=False) as old:
                        delta = np.abs(prediction['regression_prediction']-old['validation_linear_regression'])
                        roles[role].update(linear_regression_max_delta=float(delta.max()),
                            linear_regression_over_1e_6=int((delta > 1e-6).sum()),
                            linear_class_equal=np.array_equal(prediction['class_prediction'], old['validation_linear_class']))
            record = dict(method=method, candidate=candidate, route=route, checkpoint=str(checkpoint),
                checkpoint_sha256=bind(checkpoint), checkpoint_reusable=True, ignored_dtype_buffers=nonfinite,
                normalizer_reusable=True, roles=roles,
                linear_action='reuse' if all(r['pooled_equal'] for r in roles.values()) else 'refit_from_repooled_features',
                sequence_consumers_action='reuse' if all(r['sequence_equal'] for r in roles.values()) else 'refit_from_masked_sequence',
                representations_action='reuse' if all(r['pooled_equal'] and r['sequence_equal'] for r in roles.values()) else 'rebuild_from_saved_sequence',
                evidence_scope='development_only', source_code_sha256=state['source_code_sha256'])
            records.append(record)
            p95 = _pressure_p95(run/'initial_pressure', state, scores, method, candidate, route, update)
            if p95 is not None:
                pressure_records.append(dict(method=method, candidate=candidate, route=route,
                    completed_conditions=8, old_p95=p95, action='retain_historical_repool_and_reevaluate_in_new_root'))
            del encoder, normalizer, payload, consumer
    changed = [path for path, digest in inventory.items() if sha256_file(path) != digest]
    if changed:
        raise ValueError(f'original evidence changed during audit: {changed}')
    if v4_workflow_source_sha256() != source_digest:
        raise ValueError('source changed during audit; rerun the inventory')
    summary = dict(format='chronaris.v4_export_reuse.v1', status='completed',
        original_run_root=str(run), frozen_project=str(frozen),
        source_code_sha256=source_digest, source_differences=differences,
        original_source_code_sha256=cohort['source_code_sha256'], records=records,
        pressure_records=pressure_records, input_and_evidence_sha256=inventory,
        original_files_unchanged=True, checkpoint_reusable_count=len(records),
        linear_actions=dict(Counter(r['linear_action'] for r in records)),
        sequence_consumer_actions=dict(Counter(r['sequence_consumers_action'] for r in records)),
        representation_actions=dict(Counter(r['representations_action'] for r in records)),
        training_updates_retained=sum(json.loads((run/'initial/simulation'/m/c/'run_state.json').read_text())[route+'_training']['optimizer_updates']
            for m, c in cohort['units'] for route in ('self_supervised', 'task_guided')),
        automatic_old_queue_resume_allowed=False, confirmation_opened=False,
        reuse_scope='same_inputs_roles_training_and_task_contract_only; future_method_protocol_requires_revalidation')
    return write_result(output/'reuse_inventory.json', summary)
