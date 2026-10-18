"""Shared paths and serialization; earlier frozen evidence helpers are reused."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'chapter5-readout-repair-20261018'))
from repair_io import REPO,PREP,REGISTRY,OLD_ASSETS,ASSETS as REVISED,feature_path,sha,save_json,save_csv,manifest
ASSETS=REPO/'artifacts/application_evaluation/2026-10-18-frozen-feasibility'
MATERIALS=REPO/'docs/artifacts/runs/2026-10-18_chapter5-materials/data'
MECHANISMS=Path('/mnt/e/chronaris-v4-results/2026-10-16-chapter5-chronos-recovery/mechanisms')


def snapshot():
    output=HERE/'protected_sources.json'
    if output.exists():
        raise FileExistsError('source snapshot already exists')
    files=set()
    for name in ('chapter5-controlled-readouts-20261018','chapter5-readout-repair-20261018',
                 'thesis-cross-assessment-20261018'):
        files.update(p for p in (HERE.parent/name).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    files.update(p for p in MATERIALS.rglob('*') if p.is_file())
    for domain in ('clare','cogpilot'):
        files.update(p for p in (REVISED/domain).rglob('*') if p.is_file())
        files.add(PREP/domain/'sample_manifest.jsonl');files.add(PREP/domain/'task_targets.pt')
        for method in ('chronaris','physiology_only','vehicle_only'):
            for route in ('self_supervised','task_guided'):
                p=feature_path(domain,method,route);files.update((p,p.with_suffix('.json')))
    files.add(REGISTRY);files.add(MECHANISMS/'mechanism_summary.json')
    save_json(output,{str(p):sha(p) for p in sorted(files)})


if __name__=='__main__':snapshot()
