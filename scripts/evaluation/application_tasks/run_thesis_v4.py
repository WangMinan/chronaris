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
    parser.add_argument("stage", choices=("simulation-confirmation-train", "simulation-confirmation-evaluate", "simulation-confirmation-train-cohort", "simulation-confirmation-evaluate-cohort", "simulation-model-freeze", "simulation-confirmation-data", "dingxin-retained-data", "native-confirmation-plan", "native-confirmation-cohort", "naive-confirmation-unit", "configuration-cuda-validation", "freeze-configuration", "native-confirmation-unit", "review-pressure-plan", "review-pressure-cohort", "candidate-adoption", "simulation-review-results", "candidate-review-pressure", "public-review-results", "candidate-review-plan", "candidate-review-cohort", "public-screen-results", "dingxin-content-audit", "fixed-native-results", "public-screen-plan", "public-screen", "diagnostic-statistics", "naive-development", "diagnostic-figures", "public-data", "public-confirmation-data", "native-profile", "smoke", "diagnostic", "candidate", "candidate-review", "candidate-summary", "candidate-pressure", "development-conditions", "development-pressure", "expand-training"))
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
    parser.add_argument("--freeze-path",default="artifacts/application_evaluation/2026-09-08_v4-confirmation/frozen_configuration.json")
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--model-freeze-path")
    parser.add_argument("--model-freeze-sha256")
    parser.add_argument("--backend",choices=("neural","nonparametric"),default="neural")
    parser.add_argument("--validation-receipt",default="artifacts/application_evaluation/2026-09-08_v4-configuration-validation/validation_receipt.json")
    parser.add_argument("--screen-root", default="artifacts/application_evaluation/2026-09-08_v4-public-screen")
    parser.add_argument("--data-root", default="artifacts/application_evaluation/2026-09-06_v4-public-development")
    parser.add_argument("--task-mode", choices=("single", "all"), default="all")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--route", choices=("self_supervised", "task_guided"), default="self_supervised")
    parser.add_argument("--update", type=int, choices=(50, 200, 500), default=500)
    parser.add_argument("--diagnostic-root", default="artifacts/application_evaluation/2026-09-06_v4-learning-curves")
    parser.add_argument("--condition-root", default="artifacts/application_evaluation/2026-09-06_v4-development-conditions-repair")
    parser.add_argument("--simulation-root", default="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development")
    parser.add_argument("--method", choices=("physiology_only", "vehicle_only", "mult", "contiformer", "chronaris", "naive_time_sync"), default="chronaris")
    args = parser.parse_args()
    public_screen_stage = args.stage in {"public-screen-plan", "public-screen"}
    review_stage = args.stage in {"candidate-review-plan", "candidate-review-cohort"}
    aggregate_stage = public_screen_stage or review_stage or args.stage in {"fixed-native-results", "public-screen-results", "public-review-results", "candidate-adoption", "review-pressure-plan", "review-pressure-cohort", "configuration-cuda-validation", "freeze-configuration", "native-confirmation-plan", "native-confirmation-cohort"}
    if aggregate_stage and args.domain is not None:
        parser.error("this stage covers its fixed data domains; omit --domain")
    if not aggregate_stage and args.domain is None:
        parser.error("--domain is required for this stage")
    if args.seed != 17 and args.stage not in {"candidate-review", "candidate-review-pressure", "naive-development", "native-confirmation-unit", "naive-confirmation-unit", "simulation-confirmation-train", "simulation-confirmation-evaluate"}:
        raise ValueError("seeds 29 and 43 require an approved seeded development stage")
    default_roots = {"public-screen-results": "2026-09-08_v4-public-screen", "dingxin-content-audit": "2026-09-08_v4-dingxin-content-audit", "fixed-native-results": "2026-09-08_v4-native-results", "public-screen-plan": "2026-09-08_v4-public-screen", "public-screen": "2026-09-08_v4-public-screen", "diagnostic-statistics": "2026-09-08_v4-grouped-statistics", "naive-development": "2026-09-08_v4-naive-development", "diagnostic-figures": "2026-09-08_v4-diagnostic-figures", "public-data": "2026-09-06_v4-public-development", "native-profile": "2026-09-06_v4-native-recurrence",
                     "public-confirmation-data": "2026-09-08_v4-public-confirmation-prepared",
                     "smoke": "2026-09-06_v4-real-domain-smoke", "diagnostic": "2026-09-06_v4-learning-curves",
                     "development-conditions": "2026-09-06_v4-development-conditions-repair",
                     "development-pressure": "2026-09-06_v4-development-pressure",
                     "expand-training": "2026-09-07_thesis-v4-simulation-expanded",
                     "candidate": "2026-09-08_v4-single-factor-development",
                     "candidate-pressure": "2026-09-08_v4-candidate-pressure"}
    default_roots["dingxin-retained-data"] = "2026-09-08_v4-dingxin-deduplicated"
    default_roots["candidate-summary"] = "2026-09-08_v4-candidate-summary"
    default_roots["candidate-review"] = "2026-09-08_v4-candidate-review"
    for stage in ("candidate-review-plan", "candidate-review-cohort", "public-review-results", "simulation-review-results", "candidate-adoption"):
        default_roots[stage] = "2026-09-08_v4-candidate-review"
    default_roots["candidate-review-pressure"] = "2026-09-08_v4-review-pressure"
    default_roots.update({stage:"2026-09-08_v4-review-pressure" for stage in ("review-pressure-plan","review-pressure-cohort")})
    default_roots.update({"configuration-cuda-validation":"2026-09-08_v4-configuration-validation","freeze-configuration":"2026-09-08_v4-confirmation","native-confirmation-unit":"2026-09-08_v4-confirmation"})
    default_roots.update({stage:"2026-09-08_v4-confirmation" for stage in ("native-confirmation-plan","native-confirmation-cohort","naive-confirmation-unit")})
    for stage in ('simulation-confirmation-train','simulation-confirmation-evaluate','simulation-confirmation-train-cohort','simulation-confirmation-evaluate-cohort','simulation-model-freeze'):
        default_roots[stage]='2026-09-08_v4-confirmation'
    default_roots['simulation-confirmation-data']='2026-09-08_v4-simulation-confirmation'
    output_root = args.output_root or str(Path("artifacts/application_evaluation") / default_roots[args.stage])
    if args.stage.startswith('simulation-confirmation-') or args.stage=='simulation-model-freeze':
        if args.domain!='simulation' or not args.freeze_sha256:
            raise ValueError('formal simulation requires its domain and frozen configuration hash')
        from chronaris.evaluation.application_tasks.v4_simulation_confirmation import (
            train_simulation_confirmation,seal_simulation_models,run_simulation_confirmation_cohort,EXPANDED_SIMULATION_ROOT)
        simulation_root=EXPANDED_SIMULATION_ROOT if args.simulation_root.endswith('2026-09-06_thesis-v4-simulation-development') else args.simulation_root
        common=dict(freeze_path=args.freeze_path,freeze_sha256=args.freeze_sha256,output_root=output_root)
        if args.stage.endswith('-cohort'):
            result=run_simulation_confirmation_cohort(**common,backend=args.backend,
                stage='train' if args.stage=='simulation-confirmation-train-cohort' else 'evaluate')
        elif args.stage=='simulation-model-freeze':
            result=seal_simulation_models(**common,simulation_root=simulation_root)
        elif args.stage=='simulation-confirmation-train':
            result=train_simulation_confirmation(**common,method=args.method,candidate_name=args.candidate_name,
                seed=args.seed,simulation_root=simulation_root)
        else:
            if not args.model_freeze_path or not args.model_freeze_sha256:
                raise ValueError('confirmation generation/evaluation requires the frozen model inventory hash')
            common.update(model_freeze_path=args.model_freeze_path,model_freeze_sha256=args.model_freeze_sha256)
            if args.stage=='simulation-confirmation-data':
                from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import generate_simulation_confirmation
                result=generate_simulation_confirmation(**common)
            else:
                from chronaris.evaluation.application_tasks.v4_simulation_confirmation_evaluation import evaluate_simulation_confirmation
                data_root='artifacts/application_evaluation/2026-09-08_v4-simulation-confirmation' if args.data_root.endswith('2026-09-06_v4-public-development') else args.data_root
                result=evaluate_simulation_confirmation(**common,method=args.method,candidate_name=args.candidate_name,
                    seed=args.seed,confirmation_root=data_root)
        summary={key:result[key] for key in ('status','completed','evaluation_units','trajectory_count','scenario_count','pending') if key in result}
    elif args.stage in {"native-confirmation-plan","native-confirmation-cohort","naive-confirmation-unit"}:
        if not args.freeze_sha256:raise ValueError("formal confirmation requires --freeze-sha256")
        from chronaris.evaluation.application_tasks.v4_native_confirmation_cohort import build_native_confirmation_plan, run_native_confirmation_cohort
        data_root = args.data_root
        if data_root == "artifacts/application_evaluation/2026-09-06_v4-public-development":
            data_root = "artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared"
        if args.stage == "native-confirmation-plan":
            result = build_native_confirmation_plan(args.freeze_path,args.freeze_sha256)
            root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
            (root / "native_readiness.json").write_text(json.dumps(result,indent=2) + "\n")
            summary = {"units":len(result["units"]),"evaluation_units":result["enabled_evaluation_units"],"blocked_domains":result["blocked_domains"]}
        elif args.stage == "native-confirmation-cohort":
            summary = run_native_confirmation_cohort(freeze_path=args.freeze_path,freeze_sha256=args.freeze_sha256,
                output_root=output_root,backend=args.backend,data_root=data_root,registry_path=args.registry)
        else:
            from chronaris.evaluation.application_tasks.v4_confirmation_training import run_naive_native_confirmation
            result = run_naive_native_confirmation(freeze_path=args.freeze_path,freeze_sha256=args.freeze_sha256,
                domain=args.domain,fold_index=args.fold_index,seed=args.seed,output_root=output_root,data_root=data_root,registry_path=args.registry)
            summary = {"status":result["status"],"completed":result["completed"]}
    elif args.stage == "configuration-cuda-validation":
        from chronaris.evaluation.application_tasks.v4_configuration_freeze import validate_current_cuda
        summary = validate_current_cuda(output_root)
    elif args.stage == "freeze-configuration":
        from chronaris.evaluation.application_tasks.v4_configuration_freeze import freeze_reviewed_configuration
        diagnostic_root = args.diagnostic_root
        if diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = "artifacts/application_evaluation/2026-09-08_v4-candidate-review"
        result = freeze_reviewed_configuration(review_root=diagnostic_root,validation_receipt=args.validation_receipt,
            output_path=args.freeze_path,registry_path=args.registry)
        root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
        (root / "freeze_readiness.json").write_text(json.dumps(result,indent=2,allow_nan=False) + "\n")
        summary = {"status":result["status"],"configuration_frozen":result["configuration_frozen"]}
    elif args.stage == "native-confirmation-unit":
        from chronaris.evaluation.application_tasks.v4_confirmation_training import run_native_confirmation_training
        if not args.freeze_sha256:raise ValueError("native confirmation requires --freeze-sha256")
        data_root = args.data_root
        if data_root == "artifacts/application_evaluation/2026-09-06_v4-public-development":
            data_root = "artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared"
        result = run_native_confirmation_training(freeze_path=args.freeze_path,freeze_sha256=args.freeze_sha256,
            domain=args.domain,fold_index=args.fold_index,method=args.method,candidate_name=args.candidate_name,
            seed=args.seed,output_root=output_root,data_root=data_root,registry_path=args.registry,evaluate=True)
        summary = {"status":result["status"],"completed":result.get("completed",False)}
    elif args.stage in {"review-pressure-plan","review-pressure-cohort"}:
        from chronaris.evaluation.application_tasks.v4_review_pressure import build_review_pressure_plan, run_review_pressure_cohort
        diagnostic_root = args.diagnostic_root
        if diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = "artifacts/application_evaluation/2026-09-08_v4-candidate-review"
        kwargs = dict(diagnostic_root=diagnostic_root,condition_root=args.condition_root,
                      data_root=args.data_root,registry_path=args.registry)
        if args.stage == "review-pressure-plan":
            result = build_review_pressure_plan(**kwargs)
            root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
            (root / "readiness.json").write_text(json.dumps(result,indent=2) + "\n")
            summary = {"status":result["status"],"units":len(result["units"])}
        else:
            summary = run_review_pressure_cohort(output_root=output_root,**kwargs)
    elif args.stage == "candidate-adoption":
        from chronaris.evaluation.application_tasks.v4_adoption import collect_adoption_decisions
        result = collect_adoption_decisions(output_root=output_root,
            pressure_root="artifacts/application_evaluation/2026-09-08_v4-review-pressure",
            data_root=args.data_root,registry_path=args.registry)
        root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
        (root / "adoption_decisions.json").write_text(json.dumps(result,indent=2,allow_nan=False) + "\n")
        summary = {"status": result["status"], "method_routes": len(result["decisions"]),
                   "configuration_frozen": result["configuration_frozen"]}
    elif args.stage == "simulation-review-results":
        if args.domain != "simulation":raise ValueError("simulation review results require --domain simulation")
        from chronaris.evaluation.application_tasks.v4_simulation_review_results import collect_simulation_review_results
        result = collect_simulation_review_results(output_root=output_root,
            pressure_root="artifacts/application_evaluation/2026-09-08_v4-review-pressure",
            data_root=args.data_root,registry_path=args.registry)
        root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
        (root / "simulation_results_summary.json").write_text(json.dumps(result,indent=2) + "\n")
        summary = {"status": result["status"], "clean_units": len(result["completed"]),
                   "pending": result["pending"], "pressure_pending": result["pressure_pending"]}
    elif args.stage == "public-review-results":
        from chronaris.evaluation.application_tasks.v4_public_review_results import collect_public_review_results
        result = collect_public_review_results(output_root=output_root,data_root=args.data_root,registry_path=args.registry)
        root = Path(output_root); root.mkdir(parents=True,exist_ok=True)
        (root / "public_results_summary.json").write_text(json.dumps(result,indent=2) + "\n")
        summary = {"status": result["status"], "verified_units": len(result["units"]),
                   "pending": result["pending"], "failed": result["failed"]}
    elif review_stage:
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
    elif args.stage == "dingxin-retained-data":
        if args.domain != "dingxin":raise ValueError("retained-record preparation requires Dingxin")
        from chronaris.evaluation.application_tasks.v4_dingxin_deduplicated import prepare_deduplicated_dingxin
        result = prepare_deduplicated_dingxin(output_root)
        summary = {key:result[key] for key in ("status","encoder_context_counts","consumer_context_counts","inner_fields","outer_fields")}
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
    elif args.stage in {"development-pressure", "candidate-pressure", "candidate-review-pressure"}:
        if args.domain != "simulation":
            raise ValueError("these development pressure conditions require simulation")
        from chronaris.evaluation.application_tasks.v4_pressure_run import run_development_pressure
        reviewing = args.stage == "candidate-review-pressure"
        candidate_pressure = args.stage != "development-pressure"
        diagnostic_root = args.diagnostic_root
        if candidate_pressure and diagnostic_root == "artifacts/application_evaluation/2026-09-06_v4-learning-curves":
            diagnostic_root = ("artifacts/application_evaluation/2026-09-08_v4-candidate-review" if reviewing else
                               "artifacts/application_evaluation/2026-09-08_v4-single-factor-development")
        update = (1500 if reviewing else 300) if args.route == "self_supervised" else (500 if reviewing else 200)
        kwargs = dict(method=args.method,route=args.route,update=update if candidate_pressure else args.update,
            candidate_name=args.candidate_name if candidate_pressure else None,
            output_root=output_root,diagnostic_root=diagnostic_root,condition_root=args.condition_root,
            device=args.inference_device,phase="review" if reviewing else "screen",seed=args.seed)
        if reviewing:
            from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
            with development_gpu_lock() as acquired:
                summary=run_development_pressure(**kwargs) if acquired else {"status":"waiting_gpu", "seed":args.seed}
        else:
            summary=run_development_pressure(**kwargs)
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
