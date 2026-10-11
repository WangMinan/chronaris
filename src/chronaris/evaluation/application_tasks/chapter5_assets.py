"""Read and verify the completed main-model inventory without opening outer outcomes."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)
    return value


def bind(files, path, digest):
    path = str(Path(path).resolve())
    if path in files and files[path] != digest:
        raise ValueError(f'conflicting source hashes: {path}')
    files[path] = digest


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError(f'chapter 5 source changed: {path}')


def main_inventory(run_root):
    root = Path(run_root).resolve()
    state, plan, models = (read(root/name) for name in ('pipeline_state.json', 'formal_plan.json', 'main_models.json'))
    if (state['status'] != 'stage5_formal_train_completed' or not models['main_models_frozen']
        or models['confirmation_opened'] or models['core_models_frozen']
        or plan['model_units'] != 279 or plan['representation_routes'] != 477
        or models['plan_sha256'] != plan['plan_sha256']):
        raise ValueError('requires the complete, unopened main-model training matrix')
    files = {}
    for name, stage in [('formal_plan.json', 'stage5_formal_plan'), ('main_models.json', 'stage5_formal_models')]:
        receipt = state['completed'][stage]
        verify({receipt['path']: receipt['sha256']})
        if read(root/name) != read(receipt['path']):
            raise ValueError('main-model plan/freeze differs from its committed receipt')
        bind(files, root/name, sha(root/name))
        bind(files, receipt['path'], receipt['sha256'])
    bind(files, root/'pipeline_state.json', sha(root/'pipeline_state.json'))
    for path, digest in plan['input_files'].items():
        bind(files, path, digest)
    routes, units, found_models, found_consumers = [], [], {}, {}
    for index, spec in enumerate(plan['units']):
        receipt = state['completed'][f'stage5_formal_unit__{index}']
        if models['unit_receipts'].get(receipt['path']) != receipt['sha256']:
            raise ValueError('unit receipt differs from the frozen model inventory')
        verify({receipt['path']: receipt['sha256']})
        bind(files, receipt['path'], receipt['sha256'])
        unit = read(receipt['path'])
        if (unit['status'] != 'completed' or unit['unit'] != spec or unit['confirmation_opened']
            or [r['route'] for r in unit['results']] != spec['routes']):
            raise ValueError('unit identity, routes or completion differ from the plan')
        units.append(dict(index=index, spec=spec))
        for key, found in [('model_files', found_models), ('consumer_files', found_consumers)]:
            for path, digest in unit[key].items():
                bind(files, path, digest)
                bind(found, path, digest)
        for row in unit['results']:
            evaluation = row.get('evaluation', row)
            manifest = Path(evaluation['consumers']['manifest_path'])
            contract_path = manifest.parents[1]/'common_contract.json'
            verify({str(contract_path): evaluation['contract_file_sha256']})
            bind(files, contract_path, evaluation['contract_file_sha256'])
            contract = read(contract_path)
            declaration = contract['declaration']
            for path, digest in declaration['evidence_files'].items():
                bind(files, path, digest)
            checkpoint = declaration['checkpoint_path']
            if unit['model_files'].get(checkpoint) != declaration['checkpoint_sha256']:
                raise ValueError('representation checkpoint is outside the frozen inventory')
            if spec['method']=='naive_time_sync':
                fit_path=Path(checkpoint).parent/'fit_manifest.json'
                if read(fit_path)!=row['training']:
                    raise ValueError('fixed synchronization fit manifest differs from its signed unit')
                bind(files,fit_path,sha(fit_path))
            if set(evaluation['consumers']['components']) != set(spec['consumers']):
                raise ValueError('missing declared downstream family')
            if contract['contract']['seed'] != spec['seed'] or contract['contract']['data_manifest_sha256'] != spec['data_manifest_sha256']:
                raise ValueError('contract seed or data manifest changed')
            for component in evaluation['consumers']['components'].values():
                for kind in ('model', 'result'):
                    if unit['consumer_files'].get(component[kind+'_path']) != component[kind+'_sha256']:
                        raise ValueError('downstream file is outside the frozen inventory')
            routes.append(dict(unit_index=index, route=row['route'], contract_path=str(contract_path)))
    if found_models != models['model_files'] or found_consumers != models['consumer_files'] or len(routes) != 477:
        raise ValueError('model/consumer coverage differs from the complete matrix')
    return dict(root=str(root), files=files, plan=plan, models=models, state=state, units=units, routes=routes)


def audit_main_models(run_root, output_path):
    inventory = main_inventory(run_root)
    verify(inventory['files'])
    return write(output_path, dict(status='completed', format='chronaris.chapter5_asset_audit.v1',
        run_root=inventory['root'], model_units=len(inventory['units']), routes=len(inventory['routes']),
        model_files=len(inventory['models']['model_files']), consumer_files=len(inventory['models']['consumer_files']),
        source_files=inventory['files'], confirmation_opened=False,
        checkpoint_replay_completed=False, historical_failures=inventory['state']['failures']))
