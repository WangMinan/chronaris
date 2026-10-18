"""Reuse the preceding audit's immutable assets and serialization helpers."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'chapter5-controlled-readouts-20261018'
sys.path.insert(0,str(OLD))
from common import REPO,PREP,REGISTRY,ASSETS as OLD_ASSETS,sha,save_json,save_csv,manifest

ASSETS=REPO/'artifacts/application_evaluation/2026-10-18-readout-repair'
METHODS=('chronaris','physiology_only','vehicle_only','naive_time_sync','mult','contiformer')


def feature_path(domain,method,route):
    root=OLD_ASSETS/'exclusive_endpoint' if domain=='cogpilot' else OLD_ASSETS
    if method=='naive_time_sync':route='self_supervised'
    return root/f'{domain}__{method}__{route}.npz'
