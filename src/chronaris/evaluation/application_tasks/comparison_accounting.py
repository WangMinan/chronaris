"""Separate recorded attempts from checkpoint-proven, unreceipted historical work."""
import math
from pathlib import Path
import json


def attempt_accounting(unit, paths, migration=None):
    paths = sorted(set(map(str, paths)))
    attempts = [json.loads(Path(p).read_text()) for p in paths]
    if any((x['domain'], x['method']) != (unit['domain'], unit['method']) or
           not math.isfinite(x['seconds']) or x['seconds'] < 0 for x in attempts):
        raise ValueError('invalid or mismatched attempt cost')
    recorded = sum(x['seconds'] for x in attempts)
    extra = 0.
    # This migration contract only permits the interrupted CogPilot/Chronaris unit.
    affected = bool(migration and (unit['domain'], unit['method']) == ('cogpilot', 'chronaris'))
    if affected:
        parent_paths = set(migration['parent_attempt_costs'])
        if not parent_paths.intersection(paths):
            extra = float(migration['preserved_training_seconds'])
            if not math.isfinite(extra) or extra <= 0:
                raise ValueError('missing positive preserved training duration')
    return dict(observed_attempt_seconds=recorded, attempt_count=len(attempts), attempt_files=paths,
        unreceipted_checkpoint_training_seconds=extra, historical_seconds_lower_bound=recorded+extra,
        historical_cost_complete=not affected,
        cost_note=('checkpoint prefix is a lower bound; stopped parent setup and unsaved tail are not fully timed'
                   if affected else 'sum of recorded attempts; not an estimate of a fresh unit'))
