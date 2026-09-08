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
    parser.add_argument("stage", choices=("public-data", "native-profile", "smoke", "diagnostic", "candidate", "candidate-pressure", "development-conditions", "development-pressure", "expand-training"))
    from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES
    parser.add_argument("--candidate-name", choices=tuple(CANDIDATE_CHANGES), default="reference")
    parser.add_argument("--prefetch-cpu-consumers", action="store_true")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--domain", choices=("simulation", "cogpilot", "clare", "dingxin"), required=True)
    parser.add_argument("--registry", default="docs/requirements/thesis-v4-public-subjects.json")
    parser.add_argument("--output-root")
    parser.add_argument("--data-root", default="artifacts/application_evaluation/2026-09-06_v4-public-development")
    parser.add_argument("--task-mode", choices=("single", "all"), default="all")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--route", choices=("self_supervised", "task_guided"), default="self_supervised")
    parser.add_argument("--update", type=int, choices=(50, 200, 500), default=500)
    parser.add_argument("--diagnostic-root", default="artifacts/application_evaluation/2026-09-06_v4-learning-curves")
    parser.add_argument("--condition-root", default="artifacts/application_evaluation/2026-09-06_v4-development-conditions-repair")
    parser.add_argument("--simulation-root", default="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development")
    parser.add_argument("--method", choices=("physiology_only", "vehicle_only", "mult", "contiformer", "chronaris"), default="chronaris")
    args = parser.parse_args()
    default_roots = {"public-data": "2026-09-06_v4-public-development", "native-profile": "2026-09-06_v4-native-recurrence",
                     "smoke": "2026-09-06_v4-real-domain-smoke", "diagnostic": "2026-09-06_v4-learning-curves",
                     "development-conditions": "2026-09-06_v4-development-conditions-repair",
                     "development-pressure": "2026-09-06_v4-development-pressure",
                     "expand-training": "2026-09-07_thesis-v4-simulation-expanded",
                     "candidate": "2026-09-08_v4-single-factor-development",
                     "candidate-pressure": "2026-09-08_v4-candidate-pressure"}
    output_root = args.output_root or str(Path("artifacts/application_evaluation") / default_roots[args.stage])
    if args.stage == "public-data":
        summary = prepare_public_development(args.domain, registry_path=args.registry, output_root=output_root)
    elif args.stage == "candidate":
        from chronaris.evaluation.application_tasks.v4_candidates import run_candidate_development
        summary = run_candidate_development(domain=args.domain, method=args.method, candidate_name=args.candidate_name,
            output_root=output_root, fold_index=args.fold_index, data_root=args.data_root, registry_path=args.registry,
            prefetch_cpu_consumers=args.prefetch_cpu_consumers)
    elif args.stage == "expand-training":
        if args.domain != "simulation":
            raise ValueError("the approved training expansion only applies to simulation")
        from chronaris.evaluation.application_tasks.v4_simulation_extension import activate_simulation_extension
        summary = activate_simulation_extension(output_root=output_root, initial_root=args.simulation_root,
            diagnostic_root=args.diagnostic_root)
    elif args.stage == "development-conditions":
        if args.domain != "simulation":
            raise ValueError("these development conditions require simulation")
        from chronaris.evaluation.application_tasks.v4_development_conditions import generate_development_conditions
        summary = generate_development_conditions(output_root=output_root,
            clean_root="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development",
            registry_path="docs/requirements/thesis-v4-simulation-manifest.json")
    elif args.stage in {"development-pressure", "candidate-pressure"}:
        if args.domain != "simulation":
            raise ValueError("these development pressure conditions require simulation")
        from chronaris.evaluation.application_tasks.v4_pressure_run import run_development_pressure
        candidate_pressure = args.stage == "candidate-pressure"
        diagnostic_root = args.diagnostic_root
        if candidate_pressure and diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = "artifacts/application_evaluation/2026-09-08_v4-single-factor-development"
        summary = run_development_pressure(method=args.method, route=args.route,
            update=(300 if args.route == "self_supervised" else 200) if candidate_pressure else args.update,
            candidate_name=args.candidate_name if candidate_pressure else None,
            output_root=output_root, diagnostic_root=diagnostic_root, condition_root=args.condition_root,
            device=args.inference_device)
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
            fold_index=args.fold_index, data_root=args.data_root, registry_path=args.registry, simulation_root=args.simulation_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
