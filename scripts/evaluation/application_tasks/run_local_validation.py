"""Run local replay, independent resume, two complete folds and the declared subset."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))

if __name__ == '__main__':
    from chronaris.evaluation.application_tasks.local_validation import run_validation
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'replay-root', 'execution-receipt', 'main-root', 'prior-budget'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--replay-pid', type=int, required=True)
    run_validation(**vars(parser.parse_args()))
