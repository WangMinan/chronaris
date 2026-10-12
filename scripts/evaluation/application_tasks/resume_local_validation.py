"""Resume a preserved full fold and then rejoin its original frozen controller."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))

if __name__ == '__main__':
    from chronaris.evaluation.application_tasks.local_validation_recovery import resume_validation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    resume_validation(parser.parse_args().manifest)
