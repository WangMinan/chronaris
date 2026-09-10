"""Inventory the stage-1 export repair without mutating the frozen run."""
import argparse
import json

from chronaris.evaluation.application_tasks.v4_export_reuse import audit_initial_export_reuse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--frozen-project', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    result = audit_initial_export_reuse(**vars(args))
    print(json.dumps({k: result[k] for k in ('status', 'checkpoint_reusable_count', 'linear_actions',
        'sequence_consumer_actions', 'representation_actions', 'training_updates_retained')}, indent=2))
