"""Approved single-factor development cohort; no combination search here."""

EXPANDED_SIMULATION_ROOT = "artifacts/application_evaluation/2026-09-07_thesis-v4-simulation-expanded"
CANDIDATE_CHANGES = {
    "reference": {},
    "cosine_temperature": {"attention_kind": "cosine_temperature"},
    "projected_attention": {"attention_kind": "projected_dot_product"},
    "no_same_time_alignment": {"continuous_alignment_weight": 0.},
    "independent_pairing": {"independent_pairing_enabled": True, "independent_pair_weight": .1},
    "single_stream_fidelity": {"single_stream_fidelity_weight": .2},
    "quality_gate": {"quality_gate_enabled": True},
    "missingness_mixture": {},
    "multihorizon": {"prediction_horizons_s": (.5, 2., 5.)},
    "capacity64": {},
}
BASELINE_CANDIDATES = ("reference", "capacity64", "missingness_mixture", "multihorizon")


def validate_candidate_training_budget(state, *, phase, route):
    """Validate recorded optimizer steps, independently of export directory labels."""
    if phase not in ('screen','review') or route not in ('self_supervised','task_guided'):
        raise ValueError('unknown candidate phase or route')
    count=state['self_supervised_training']['optimizer_updates']
    if type(count) is not int or (not 500<=count<=1500 if phase=='review' else count!=300):
        raise ValueError('candidate pretraining update counts differ from the frozen budget')
    if route=='task_guided':
        guided=state['task_guided_training'];joint=guided['joint_updates']
        if (type(joint) is not int or type(guided['optimizer_updates']) is not int
            or guided['head_warmup_updates']!=50 or guided['optimizer_updates']!=50+joint
            or (not 200<=joint<=500 if phase=='review' else joint!=200)):
            raise ValueError('candidate supervised update counts differ from the frozen budget')


def candidate_options(method, name):
    if name not in CANDIDATE_CHANGES:
        raise ValueError("candidate outside the approved single-factor cohort")
    if method not in ("chronaris", "physiology_only", "vehicle_only", "mult", "contiformer"):
        raise ValueError("candidate requires a trainable method")
    if method != "chronaris" and name not in BASELINE_CANDIDATES:
        raise ValueError("Chronaris structural candidate cannot be assigned to a baseline")
    return {"name": name, "hidden_dim": 64 if name == "capacity64" else 32,
            "training": dict(CANDIDATE_CHANGES[name]), "missingness_mixture": name == "missingness_mixture"}


def run_candidate_development(*, method, candidate_name, output_root, domain="simulation", fold_index=0,
                              data_root="artifacts/application_evaluation/2026-09-06_v4-public-development",
                              registry_path="docs/requirements/thesis-v4-public-subjects.json", prefetch_cpu_consumers=False,
                              phase="screen", seed=17, routes=("self_supervised", "task_guided")):
    from chronaris.evaluation.application_tasks.v4_diagnostic_run import run_development_diagnostic
    return run_development_diagnostic(domain=domain, method=method, output_root=output_root,
        fold_index=fold_index, data_root=data_root, registry_path=registry_path,
        simulation_root=EXPANDED_SIMULATION_ROOT, candidate_name=candidate_name,
        prefetch_cpu_consumers=prefetch_cpu_consumers, phase=phase, seed=seed, routes=routes)
