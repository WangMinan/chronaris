"""Launch the existing v4 stages from one resumable entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.v4_pipeline import main

if __name__ == '__main__':
    main()
