"""Thin entrypoint for the staged Chronaris v4 workflow."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chronaris.evaluation.application_tasks.v4_public_data import prepare_public_development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("public-data", "native-profile"))
    parser.add_argument("--domain", choices=("cogpilot", "clare"), required=True)
    parser.add_argument("--registry", default="docs/requirements/thesis-v4-public-subjects.json")
    parser.add_argument("--output-root", default="artifacts/application_evaluation/2026-09-06_v4-public-development")
    parser.add_argument("--data-root", default="artifacts/application_evaluation/2026-09-06_v4-public-development")
    args = parser.parse_args()
    if args.stage == "public-data":
        summary = prepare_public_development(args.domain, registry_path=args.registry, output_root=args.output_root)
    else:
        from chronaris.evaluation.application_tasks.v4_native_performance import profile_native_recurrence
        summary = profile_native_recurrence(domain=args.domain, registry_path=args.registry,
            data_root=args.data_root, output_root=args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
