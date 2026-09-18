"""Qualify the execution revision using preserved eager references and fresh graph runs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT/'src'))

from chronaris.evaluation.application_tasks.stage45_performance import performance_entry, compare_execution
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--eager-parent', required=True)
    parser.add_argument('--domain', choices=('clare','cogpilot','dingxin'))
    parser.add_argument('--recipe', choices=('stage4_reference','thesis_reference'))
    parser.add_argument('--mode', choices=('graph','interrupt','resume','check'))
    args = parser.parse_args()
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    config = dict(root=str(root), data_root=str(PROJECT/'artifacts/application_evaluation/2026-09-06_v4-public-development'),
                  registry_path=str(PROJECT/'docs/requirements/thesis-v4-public-subjects.json'))
    if args.mode:
        if not args.domain or not args.recipe:
            parser.error('worker mode requires domain and recipe')
        if args.mode == 'check':
            directory = root/'performance'/args.domain/args.recipe
            result = compare_execution(directory, revised=True)
            result['workflow_source_sha256'] = v4_workflow_source_sha256()
            write_result(directory/'check.json', result)
            if not result['passed']:
                raise SystemExit(1)
        else:
            performance_entry(config, args.domain, args.recipe, args.mode)
        return
    source = v4_workflow_source_sha256()
    state = dict(status='running', started=time.time(), workflow_source_sha256=source, completed=[])
    try:
        for domain in ('clare','cogpilot','dingxin'):
            for recipe in ('stage4_reference','thesis_reference'):
                directory = root/'performance'/domain/recipe
                directory.mkdir(parents=True, exist_ok=True)
                reference = (Path(args.eager_parent)/domain/recipe/'eager').resolve()
                if not (reference/'trace').is_dir():
                    raise ValueError('preserved ordinary execution trace is missing')
                target = directory/'eager'
                if not target.exists():
                    target.symlink_to(reference, target_is_directory=True)
                if target.resolve() != reference:
                    raise ValueError('ordinary execution reference changed')
                write_result(directory/'eager_source.json', dict(root=str(reference), files={str(p):sha256_file(p)
                    for p in reference.rglob('*') if p.is_file()}))
                for mode in ('graph','interrupt','resume','check'):
                    state['current'] = f'{domain}/{recipe}/{mode}'
                    write_result(root/'qualification_state.json', state)
                    if source != v4_workflow_source_sha256():
                        raise ValueError('qualification source changed')
                    with (directory/f'{mode}.log').open('w') as log:
                        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),
                            '--eager-parent',args.eager_parent,'--domain',domain,'--recipe',recipe,'--mode',mode],
                            cwd=PROJECT, env=os.environ | {'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    state['completed'].append(state['current'])
        state['status'] = 'completed'
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['finished'] = time.time()
        write_result(root/'qualification_state.json', state)


if __name__ == '__main__':
    main()
