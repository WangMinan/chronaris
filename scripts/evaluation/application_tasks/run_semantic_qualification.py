"""Run short native entries with a shared two-hour deadline."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.semantic_qualification import compare_entries, mature_probe, run_bounded, short_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--domain', choices=('clare', 'cogpilot'))
    parser.add_argument('--mode', choices=('legacy', 'candidate', 'interrupt', 'resume'))
    parser.add_argument('--mature', action='store_true')
    parser.add_argument('--resume-state')
    args = parser.parse_args()
    if args.mode:
        if not args.domain:
            parser.error('worker requires a domain')
        if args.mature:
            print(mature_probe(root=args.root, domain=args.domain, mode=args.mode, resume=args.resume_state))
        else:
            print(short_entry(root=args.root, domain=args.domain, mode=args.mode))
        return
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = Path(args.root)
    for domain in ('clare', 'cogpilot'):
        for mode in ('legacy', 'candidate', 'interrupt', 'resume'):
            print(f'qualification {domain} {mode}', flush=True)
            run_bounded([sys.executable, __file__, '--root', str(root), '--deadline', args.deadline,
                         '--domain', domain, '--mode', mode], deadline_path=args.deadline,
                        log_path=root/'logs'/f'{domain}_{mode}.log', max_seconds=900)
        result = compare_entries(root, domain)
        if not result['passed']:
            raise RuntimeError(f'{domain} execution or resume comparison failed')
        print(f'{domain} comparison passed', flush=True)


if __name__ == '__main__':
    main()
