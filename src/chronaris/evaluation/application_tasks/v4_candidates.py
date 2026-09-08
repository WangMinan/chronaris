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
                              registry_path="docs/requirements/thesis-v4-public-subjects.json", prefetch_cpu_consumers=False):
    from chronaris.evaluation.application_tasks.v4_diagnostic_run import run_development_diagnostic
    return run_development_diagnostic(domain=domain, method=method, output_root=output_root,
        fold_index=fold_index, data_root=data_root, registry_path=registry_path,
        simulation_root=EXPANDED_SIMULATION_ROOT, candidate_name=candidate_name,
        prefetch_cpu_consumers=prefetch_cpu_consumers)
