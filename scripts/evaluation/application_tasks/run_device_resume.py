"""Execute source-recorded legacy device qualification in an isolated workspace."""
import argparse
import importlib.util
from pathlib import Path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--mode', choices=('continuous', 'resumed'), required=True)
    parser.add_argument('--index', type=int, default=18)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.workspace)/'src'))
    path = Path(__file__).resolve().parents[3]/'src/chronaris/evaluation/application_tasks/device_resume.py'
    spec = importlib.util.spec_from_file_location('device_resume_runner', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.run_legacy(receipt_path=args.receipt, mode=args.mode, index=args.index, stop_at=args.stop_at)
