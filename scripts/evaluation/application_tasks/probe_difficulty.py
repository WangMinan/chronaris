"""Run the preregistered source-bound CPU classification feasibility probe."""
import argparse
from chronaris.evaluation.application_tasks.difficulty_feasibility import run_difficulty_feasibility

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    print(run_difficulty_feasibility(a.output))
