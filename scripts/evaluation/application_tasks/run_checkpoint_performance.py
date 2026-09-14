"""Run or compare bounded preserved-checkpoint execution trials in separate directories."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_trials, trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--pipeline-config')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--domain', choices=('clare','cogpilot','dingxin'))
    parser.add_argument('--torch-profile', action='store_true', help='Profile one microbatch with PyTorch CPU/CUDA tracing')
    parser.add_argument('--profile', action='store_true', help='Capture the first microbatch using Nsight CUDA profiler API range')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--resume-state')
    parser.add_argument('--compare-with')
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()
    if args.compare_with:
        result = compare_trials(args.compare_with, args.output_root, exact=args.exact)
        name = 'exact_comparison.json' if args.exact else 'comparison.json'
        (Path(args.output_root)/name).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not result['passed']:
            raise SystemExit(1)
    else:
        if not args.checkpoint or not args.pipeline_config:
            parser.error('trials require --checkpoint and --pipeline-config')
        print(json.dumps(trial(checkpoint=args.checkpoint, pipeline_config=args.pipeline_config,
            output_root=args.output_root, graph=args.graph, threads=args.threads,
            updates=args.updates, resume_state=args.resume_state, domain=args.domain, profile=args.profile, torch_profile=args.torch_profile), indent=2))


if __name__ == '__main__':
    main()
