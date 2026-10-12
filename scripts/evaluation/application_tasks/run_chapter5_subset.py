"""Prepare, train and freeze the predeclared 15+3 thesis subset."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))


def main():
    from chronaris.evaluation.application_tasks import chapter5_subset as subset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'dry-run', 'train', 'freeze', 'predict', 'generate-simulation', 'evaluate-simulation'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--main-root')
    parser.add_argument('--core-root')
    parser.add_argument('--legacy-workspace')
    parser.add_argument('--pause-receipt')
    parser.add_argument('--unit', type=int)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--role', choices=('validation', 'held_out'), default='validation')
    parser.add_argument('--sample-limit', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not all((args.main_root, args.core_root, args.legacy_workspace, args.pause_receipt)):
            parser.error('prepare requires main/core roots, legacy workspace and pause receipt')
        result = subset.prepare_subset(main_root=args.main_root, core_root=args.core_root,
            legacy_workspace=args.legacy_workspace, pause_receipt=args.pause_receipt, output_root=args.root)
    elif args.action == 'dry-run': result = subset.verified_subset(args.root)
    elif args.action == 'freeze': result = subset.freeze_subset(args.root)
    elif args.action == 'generate-simulation':
        from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import generate_simulation
        result = generate_simulation(args.root)
    else:
        if args.unit is None: parser.error('action requires --unit')
        if args.action == 'train': result = subset.train_subset_unit(args.root, args.unit)
        elif args.action == 'predict':
            from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import evaluate_native
            result = evaluate_native(args.root, args.unit, reference=args.reference, role=args.role, sample_limit=args.sample_limit)
        else:
            from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import evaluate_simulation
            result = evaluate_simulation(args.root, args.unit)
    print(json.dumps({k:v for k,v in result.items() if k in ('status', 'model_units', 'public_units', 'plan_sha256',
        'source_code_sha256', 'confirmation_opened', 'subset_models_frozen', 'resume_counts')}, indent=2))
    if result.get('status') == 'waiting_gpu': raise SystemExit(75)


if __name__ == '__main__': main()
