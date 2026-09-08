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
    parser.add_argument("stage", choices=("candidate-review-plan", "candidate-review-cohort", "public-screen-results", "dingxin-content-audit", "fixed-native-results", "public-screen-plan", "public-screen", "diagnostic-statistics", "naive-development", "diagnostic-figures", "public-data", "public-confirmation-data", "native-profile", "smoke", "diagnostic", "candidate", "candidate-review", "candidate-summary", "candidate-pressure", "development-conditions", "development-pressure", "expand-training"))
    from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES
    parser.add_argument("--candidate-name", choices=tuple(CANDIDATE_CHANGES), default="reference")
    parser.add_argument("--prefetch-cpu-consumers", action="store_true")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--routes", nargs="+", choices=("self_supervised", "task_guided"), default=("self_supervised", "task_guided"))
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--domain", choices=("simulation", "cogpilot", "clare", "dingxin"))
    parser.add_argument("--registry", default="docs/requirements/thesis-v4-public-subjects.json")
    parser.add_argument("--output-root")
    parser.add_argument("--dingxin-results-root")
    parser.add_argument("--screen-root", default="artifacts/application_evaluation/2026-09-08_v4-public-screen")
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
    public_screen_stage = args.stage in {"public-screen-plan", "public-screen"}
    review_stage = args.stage in {"candidate-review-plan", "candidate-review-cohort"}
    aggregate_stage = public_screen_stage or review_stage or args.stage in {"fixed-native-results", "public-screen-results"}
    if aggregate_stage and args.domain is not None:
        parser.error("this stage covers its fixed data domains; omit --domain")
    if not aggregate_stage and args.domain is None:
        parser.error("--domain is required for this stage")
    if args.seed != 17 and args.stage not in {"candidate-review", "naive-development"}:
        raise ValueError("seeds 29 and 43 require an approved seeded development stage")
    default_roots = {"public-screen-results": "2026-09-08_v4-public-screen", "dingxin-content-audit": "2026-09-08_v4-dingxin-content-audit", "fixed-native-results": "2026-09-08_v4-native-results", "public-screen-plan": "2026-09-08_v4-public-screen", "public-screen": "2026-09-08_v4-public-screen", "diagnostic-statistics": "2026-09-08_v4-grouped-statistics", "naive-development": "2026-09-08_v4-naive-development", "diagnostic-figures": "2026-09-08_v4-diagnostic-figures", "public-data": "2026-09-06_v4-public-development", "native-profile": "2026-09-06_v4-native-recurrence",
                     "public-confirmation-data": "2026-09-08_v4-public-confirmation-prepared",
                     "smoke": "2026-09-06_v4-real-domain-smoke", "diagnostic": "2026-09-06_v4-learning-curves",
                     "development-conditions": "2026-09-06_v4-development-conditions-repair",
                     "development-pressure": "2026-09-06_v4-development-pressure",
                     "expand-training": "2026-09-07_thesis-v4-simulation-expanded",
                     "candidate": "2026-09-08_v4-single-factor-development",
                     "candidate-pressure": "2026-09-08_v4-candidate-pressure"}
    default_roots["candidate-summary"] = "2026-09-08_v4-candidate-summary"
    default_roots["candidate-review"] = "2026-09-08_v4-candidate-review"
    for stage in ("candidate-review-plan", "candidate-review-cohort"):
        default_roots[stage] = "2026-09-08_v4-candidate-review"
    output_root = args.output_root or str(Path("artifacts/application_evaluation") / default_roots[args.stage])
    if review_stage:
        from chronaris.evaluation.application_tasks.v4_review_plan import build_review_plan, run_review_cohort
        kwargs = dict(screen_root=args.screen_root,data_root=args.data_root,registry_path=args.registry)
        if args.stage == "candidate-review-plan":
            plan = build_review_plan(**kwargs)
            root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
            (root / "readiness.json").write_text(json.dumps(plan,indent=2) + "\n")
            summary = {"status": plan["status"], "unit_count": len(plan["units"])}
        else:
            summary = run_review_cohort(output_root=output_root,**kwargs)
    elif args.stage == "public-screen-results":
        from chronaris.evaluation.application_tasks.v4_public_results import collect_public_screen_results
        result = collect_public_screen_results(output_root=output_root, data_root=args.data_root, registry_path=args.registry)
        root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
        (root / "results_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        summary = {"status": result["status"], "pending": result["pending"], "failed": result["failed"],
                   "ranked_method_routes": len(result["rankings"])}
    elif args.stage == "dingxin-content-audit":
        if args.domain != "dingxin":
            raise ValueError("Dingxin content audit requires its fixed data domain")
        from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development, audit_dingxin_vehicle_reuse
        result = audit_dingxin_vehicle_reuse(load_v4_dingxin_development())
        root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
        (root / "vehicle_content_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        summary = {key: value for key, value in result.items() if key != "context_rows"}
    elif args.stage == "fixed-native-results":
        from chronaris.evaluation.application_tasks.v4_native_result_audit import collect_fixed_native_development
        result = collect_fixed_native_development(run_root="artifacts/application_evaluation/2026-09-08_v4-naive-development",
            output_root=output_root, data_root=args.data_root, registry_path=args.registry,
            domain_run_roots={"dingxin": args.dingxin_results_root} if args.dingxin_results_root else None)
        summary = {key: result[key] for key in ("completed_units", "expected_units", "pending", "failed", "projection_fit_s", "consumer_fit_s")}
    elif public_screen_stage:
        from chronaris.evaluation.application_tasks.v4_public_screen import build_public_screen_plan, run_public_screen
        diagnostic_root = args.diagnostic_root
        if diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = "artifacts/application_evaluation/2026-09-08_v4-single-factor-development"
        kwargs = dict(diagnostic_root=diagnostic_root,
            pressure_root="artifacts/application_evaluation/2026-09-08_v4-candidate-pressure", registry_path=args.registry)
        if args.stage == "public-screen-plan":
            plan = build_public_screen_plan(**kwargs)
            root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
            (root / "readiness.json").write_text(json.dumps(plan, indent=2) + "\n")
            summary = {"status": plan["status"], "unit_count": len(plan["units"]), "pending": plan["pending"]}
        else:
            summary = run_public_screen(**kwargs, output_root=output_root, data_root=args.data_root)
    elif args.stage == "diagnostic-statistics":
        if args.domain != "simulation":
            raise ValueError("initial profile statistics require simulation development")
        from chronaris.evaluation.application_tasks.v4_grouped_statistics import summarize_initial_profile_statistics
        result = summarize_initial_profile_statistics(output_root=output_root)
        summary = {"output_root": output_root, "comparison_count": len(result["comparisons"]), "scope": result["scope"]}
    elif args.stage == "naive-development":
        from chronaris.evaluation.application_tasks.v4_naive_baseline import run_native_naive_development
        summary = run_native_naive_development(domain=args.domain, fold_index=args.fold_index, seed=args.seed,
            output_root=output_root, data_root=args.data_root, registry_path=args.registry)
    elif args.stage == "diagnostic-figures":
        if args.domain != "simulation":
            raise ValueError("initial diagnostic figures describe simulation development")
        from chronaris.evaluation.application_tasks.v4_diagnostic_figures import render_initial_diagnostic_figures
        result = render_initial_diagnostic_figures(output_root=output_root)
        summary = {"output_root": output_root, "figure_count": len(result["figures"]),
                   "verified_source_count": len(result["source_files"]), "scope": result["scope"]}
    elif args.stage in {"public-data", "public-confirmation-data"}:
        summary = prepare_public_development(args.domain, registry_path=args.registry, output_root=output_root,
            role="confirmation" if args.stage == "public-confirmation-data" else "development")
    elif args.stage == "candidate-summary":
        if args.domain != "simulation":
            raise ValueError("initial candidate summary requires simulation development")
        from chronaris.evaluation.application_tasks.v4_candidate_results import collect_simulation_screen
        diagnostic_root = args.diagnostic_root
        if diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = "artifacts/application_evaluation/2026-09-08_v4-single-factor-development"
        summary = collect_simulation_screen(diagnostic_root=diagnostic_root,
            pressure_root="artifacts/application_evaluation/2026-09-08_v4-candidate-pressure", route=args.route, method=args.method)
        root = Path(output_root) / args.method
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{args.route}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    elif args.stage in {"candidate", "candidate-review"}:
        from chronaris.evaluation.application_tasks.v4_candidates import run_candidate_development
        summary = run_candidate_development(domain=args.domain, method=args.method, candidate_name=args.candidate_name,
            output_root=output_root, fold_index=args.fold_index, data_root=args.data_root, registry_path=args.registry,
            prefetch_cpu_consumers=args.prefetch_cpu_consumers, seed=args.seed, routes=args.routes,
            phase="review" if args.stage == "candidate-review" else "screen")
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
