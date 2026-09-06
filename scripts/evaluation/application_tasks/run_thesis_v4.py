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
    parser.add_argument("stage", choices=("public-data", "native-profile", "smoke", "diagnostic", "development-conditions"))
    parser.add_argument("--domain", choices=("simulation", "cogpilot", "clare", "dingxin"), required=True)
    parser.add_argument("--registry", default="docs/requirements/thesis-v4-public-subjects.json")
    parser.add_argument("--output-root")
    parser.add_argument("--data-root", default="artifacts/application_evaluation/2026-09-06_v4-public-development")
    parser.add_argument("--task-mode", choices=("single", "all"), default="all")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--method", choices=("physiology_only", "vehicle_only", "mult", "contiformer", "chronaris"), default="chronaris")
    args = parser.parse_args()
    default_roots = {"public-data": "2026-09-06_v4-public-development", "native-profile": "2026-09-06_v4-native-recurrence",
                     "smoke": "2026-09-06_v4-real-domain-smoke", "diagnostic": "2026-09-06_v4-learning-curves",
                     "development-conditions": "2026-09-06_v4-development-conditions-repair"}
    output_root = args.output_root or str(Path("artifacts/application_evaluation") / default_roots[args.stage])
    if args.stage == "public-data":
        summary = prepare_public_development(args.domain, registry_path=args.registry, output_root=output_root)
    elif args.stage == "development-conditions":
        if args.domain != "simulation":
            raise ValueError("these development conditions require simulation")
        from chronaris.evaluation.application_tasks.v4_development_conditions import generate_development_conditions
        summary = generate_development_conditions(output_root=output_root,
            clean_root="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development",
            registry_path="docs/requirements/thesis-v4-simulation-manifest.json")
    elif args.stage == "native-profile":
        from chronaris.evaluation.application_tasks.v4_native_performance import profile_native_recurrence
        summary = profile_native_recurrence(domain=args.domain, registry_path=args.registry,
            data_root=args.data_root, output_root=output_root)
    elif args.stage == "smoke":
        from chronaris.evaluation.application_tasks.v4_smoke_run import run_v4_smoke
        summary = run_v4_smoke(domain=args.domain, registry_path=args.registry,
            data_root=args.data_root, output_root=output_root, task_mode=args.task_mode)
    else:
        from chronaris.evaluation.application_tasks.v4_diagnostic_run import run_development_diagnostic
        summary = run_development_diagnostic(domain=args.domain, method=args.method, output_root=output_root,
            fold_index=args.fold_index, data_root=args.data_root, registry_path=args.registry)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
