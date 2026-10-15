"""Restart the three declared simulations with an explicitly qualified audit policy."""
import argparse
import importlib.util
from pathlib import Path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--action', choices=('controller', 'legacy', 'native'), default='controller')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.workspace)/'src'))
    source = Path(__file__).resolve().parents[3]/'src/chronaris/evaluation/application_tasks/causality_recovery.py'
    spec = importlib.util.spec_from_file_location('causality_recovery_runner', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if args.action == 'controller':
        module.continue_subset(args.policy, Path(__file__).resolve())
    else:
        module.worker(policy_path=args.policy, index=args.index, legacy=args.action == 'legacy')
