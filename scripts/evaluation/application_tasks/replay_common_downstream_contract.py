"""Revalidate unchanged real exports under strengthened common-contract checks."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT/'src'))

import numpy as np
import torch

from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract, run_common_downstream
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.representation import load_fusion_stream_batch
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--frozen-runner', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('revalidation requires a separate destination')
    torch.set_num_threads(1)
    evidence = {}
    summaries = []
    for domain in ('dingxin', 'cogpilot', 'clare'):
        before = json.loads((source/domain/'summary.json').read_text())
        if before['status'] != 'completed' or before['confirmation_opened']:
            raise ValueError('only completed development exports can be revalidated')
        _, _, fold, digest, targets, definitions, context, observations = contract_development_inputs(domain,
            data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
            registry_path='docs/requirements/thesis-v4-public-subjects.json')
        kwargs = dict(fold=fold, data_manifest_sha256=digest, targets=targets, definitions=definitions,
            context=context, observations=observations)
        contract = build_common_contract(domain=domain, **kwargs)
        previous = json.loads((source/domain/'data_contract.json').read_text())
        ignored = {'source_code_sha256', 'contract_sha256'}
        if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in contract.items() if k not in ignored}:
            raise ValueError('revalidation cannot change tasks, observations, roles or algorithm rules')
        results = []
        for entry in before['results']:
            directory = Path(entry['consumers']['manifest_path']).parent.parent
            recorded = json.loads((directory/'common_contract.json').read_text())
            declaration = recorded['declaration']
            runner_hash = next(v for k, v in declaration['evidence_files'].items() if k.endswith('common_downstream_smoke.py'))
            if sha256_file(args.frozen_runner) != runner_hash:
                raise ValueError('frozen extraction runner does not match the original evidence')
            representations = {r: load_fusion_stream_batch(directory.parent/'representations'/r) for r in ('train', 'validation')}
            for path in directory.parent.rglob('*'):
                if path.is_file():
                    evidence[str(path)] = sha256_file(path)
            window = declaration['kind'] == 'window_end'
            if window:
                representations = {r: WindowFeatureBatch(o.sample_ids, observations[r].context_durations_s,
                    o.pooled_embedding, o.valid_mask.any(1), o.method_name, o.fold_id, o.checkpoint_sha256,
                    o.source_sample_hashes) for r, o in representations.items()}
                for role, batch in representations.items():
                    with np.load(directory.parent/'window_features'/f'{role}.npz') as stored:
                        for name in ('pooled_embedding', 'valid_mask', 'timestamps_s'):
                            if not np.array_equal(stored[name], getattr(batch, name).numpy()):
                                raise ValueError('stored window features differ from the original hidden-state mean')
            naive = declaration['method'] == 'naive_time_sync'
            declaration = declaration | dict(feature_origin='unsupervised_projection' if naive else 'hidden_state',
                extraction_location='pool_exported_sequence' if window else 'NaiveTimeSyncFusionAdapter.__call__' if naive else 'TrainedFusionAdapter.__call__',
                evidence_files={str(args.frozen_runner.resolve()): runner_hash,
                    declaration['checkpoint_path']: declaration['checkpoint_sha256'],
                    str(directory/'common_contract.json'): sha256_file(directory/'common_contract.json')})
            destination = output/domain/directory.relative_to(source/domain)
            result = run_common_downstream(**kwargs, contract=contract, outputs=representations,
                declaration=declaration, output_root=destination, families=recorded['families'])
            if result != run_common_downstream(**kwargs, contract=contract, outputs=representations,
                    declaration=declaration, output_root=destination, families=recorded['families']):
                raise ValueError('revalidation resume changed results')
            for family, component in result['consumers']['components'].items():
                original = entry['consumers']['components'][family]
                if sha256_file(original['result_path']) != original['result_sha256'] or sha256_file(original['model_path']) != original['model_sha256']:
                    raise ValueError('original fitted consumer evidence changed')
                old_scores = json.loads(Path(original['result_path']).read_text())['evaluations']
                new_scores = json.loads(Path(component['result_path']).read_text())['evaluations']
                if old_scores != new_scores:
                    raise ValueError('new contract checks changed predictions or metrics')
            results.append(result | dict(predictions_and_metrics_identical=True, resume_identical=True))
        summary = dict(status='completed', domain=domain, results=results, encoder_updates_added=0,
            source_training_summary=str(source/domain/'summary.json'), sample_counts=before['sample_counts'],
            previous_source_code_sha256=previous['source_code_sha256'], source_code_sha256=contract['source_code_sha256'])
        write_result(output/domain/'summary.json', summary)
        summaries.append(summary)
        print(domain, 'verified', flush=True)
    if any(sha256_file(p) != h for p, h in evidence.items()):
        raise ValueError('original artifacts changed during revalidation')
    write_result(output/'revalidation.json', dict(status='completed', summaries=summaries,
        preserved_files=evidence, encoder_updates_added=0, original_files_unchanged=True))


if __name__ == '__main__':
    main()
