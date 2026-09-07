"""Activate the one approved training expansion after the initial diagnostic matrix."""
from dataclasses import fields, replace
from pathlib import Path
import json
import math
import hashlib
import time

from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.simulation.aviation_dual_stream.benchmark import SimulationBenchmarkConfig, SimulationSplitSpec, generate_benchmark
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from chronaris.simulation.aviation_dual_stream.profiles import sample_pilot_profile


INITIAL_SIMULATION_ROOT = "artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development"
EXPANDED_SIMULATION_ROOT = "artifacts/application_evaluation/2026-09-07_thesis-v4-simulation-expanded"


def _verify_extension_decision(decision, diagnostic_root):
    if (decision["allowed_extension_count"] != 1 or decision["additional_profiles"] != 32
        or decision["additional_trajectories_per_profile"] != 8
        or decision["approved_training_trajectories_after_activation"] != 512 or decision["confirmation_feedback_used"]):
        raise ValueError("extension decision differs from the approved single expansion")
    evidence = {}
    for method in ("chronaris", "physiology_only", "vehicle_only", "mult", "contiformer"):
        path = Path(diagnostic_root) / "simulation" / method / "run_state.json"
        state = json.loads(path.read_text())
        expected = {f"{route}:{update}" for route in ("self_supervised", "task_guided") for update in (50, 200, 500)}
        if (not state.get("completed") or set(state["completed_consumers"]) != expected
            or state["seed"] != 17 or state["method"] != method
            or state["self_supervised_training"]["optimizer_updates"] != 500
            or state["task_guided_training"]["optimizer_updates"] != 550):
            raise ValueError("initial five-method two-route diagnostic matrix is incomplete")
        evidence[method] = sha256_file(path)
    state = json.loads((Path(diagnostic_root) / "simulation" / decision["trigger_method"] / "run_state.json").read_text())
    rows = {row["optimizer_updates"]: row for row in state[decision["route"] + "_training"]["epoch_rows"]}
    losses = {update: rows[update]["public_selection_loss"] for update in (200, 400, 500)}
    if (not all(math.isfinite(value) and value > 0 for value in losses.values())
        or losses != {int(key): value for key, value in decision["validation_losses"].items()}
        or 1 - losses[500] / losses[200] < .02 or 1 - losses[500] / losses[400] < .01
        or sha256_file(decision["training_checkpoint"]) != decision["checkpoint_sha256"]):
        raise ValueError("recorded extension trigger does not match completed learning curves")
    return evidence


def activate_simulation_extension(*, output_root=EXPANDED_SIMULATION_ROOT, initial_root=INITIAL_SIMULATION_ROOT,
    registry_path="docs/requirements/thesis-v4-simulation-manifest.json",
    decision_path="docs/requirements/thesis-v4-data-extension-decision.json",
    diagnostic_root="artifacts/application_evaluation/2026-09-06_v4-learning-curves"):
    root, initial = Path(output_root).resolve(), Path(initial_root).resolve()
    if root == initial:
        raise ValueError("training expansion must preserve the initial data root")
    registry = json.loads(Path(registry_path).read_text())
    decision = json.loads(Path(decision_path).read_text())
    evidence = _verify_extension_decision(decision, diagnostic_root)
    initial_audit = json.loads((initial / "v4_generation_audit.json").read_text())
    if initial_audit["registry_sha256"] != sha256_file(registry_path) or initial_audit["roles"] != {"train": 256, "validation": 64}:
        raise ValueError("initial simulation data/registry changed")
    specs = [SimulationSplitSpec(**{field.name: row[field.name] for field in fields(SimulationSplitSpec)})
             for row in registry["split_specs"] if row["role"] in {"train", "validation"}]
    specs = tuple(replace(spec, profile_count=64) if spec.split_id == "v4_train" else spec for spec in specs)
    if {spec.split_id: spec.profile_count for spec in specs} != {"v4_train": 64, "v4_development": 8}:
        raise ValueError("expanded data must contain 64 training and 8 development profiles")
    planned = [row for row in registry["trajectories"] if row["role"] in {"train", "validation"}]
    profiles = {row["profile_id"]: row for row in registry["profiles"] if row["role"] in {"train", "validation"}}
    for spec in specs:
        for index in range(spec.profile_count):
            profile = sample_pilot_profile(split_id=spec.split_id, profile_index=index, seed=spec.profile_seed_base + index)
            if any(profiles[profile.profile_id][key] != value for key, value in profile.to_dict().items()):
                raise ValueError("expanded generator profiles do not reproduce the frozen registry")
    original_rows = json.loads((initial / "simulation_manifest.json").read_text())["scenario_rows"]
    if {row["trajectory_id"] for row in original_rows} != {row["trajectory_id"] for row in planned if row["activation"] == "initial"}:
        raise ValueError("initial trajectory inventory changed")
    for row in original_rows:
        if any(sha256_file(Path(row["scenario_manifest_path"]).with_name(filename)) != row[key] for filename, key in
               (("raw_dual_stream.npz", "raw_sha256"), ("ground_truth.npz", "ground_truth_sha256"))):
            raise ValueError("initial observations changed")
    source_digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted((Path(__file__).parents[2] / "simulation/aviation_dual_stream").glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    contract = {"registry_sha256": sha256_file(registry_path), "extension_decision_sha256": sha256_file(decision_path),
        "initial_generation_audit_sha256": sha256_file(initial / "v4_generation_audit.json"),
        "diagnostic_completion_sha256": evidence, "split_specs": [spec.to_dict() for spec in specs],
        "context_starts_s": registry["context_starts_s"], "context_duration_s": registry["context_duration_s"],
        "generator_source_sha256": source_digest.hexdigest()}
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "extension_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("existing expansion source/decision changed")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    for path in root.glob("*/*/*/clean_asynchronous/scenario_manifest.json"):
        item = json.loads(path.read_text())
        if any(sha256_file(item[kind + "_path"]) != item[hash_key] for kind, hash_key in
               (("raw_dual_stream", "raw_dual_stream_sha256"), ("ground_truth", "ground_truth_sha256"))):
            raise ValueError("existing expanded data files changed")
    audit_path = root / "v4_generation_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (audit["extension_contract_sha256"] != sha256_file(contract_path)
            or audit["simulation_manifest_sha256"] != sha256_file(root / "simulation_manifest.json")):
            raise ValueError("expanded simulation manifest changed")
        return audit
    started = time.perf_counter()
    with _periodic_training_heartbeat("v4_training_expansion", 30., root=root) as progress:
        result = generate_benchmark(SimulationBenchmarkConfig(run_id=root.name, output_root=str(root.parent),
            duration_s=registry["duration_s"], truth_rate_hz=registry["truth_rate_hz"], split_specs=specs,
            observation_scenarios=(ObservationScenarioConfig("clean_asynchronous"),), paired_observation_seed=True),
            progress_callback=lambda event, values: progress.update(phase=event, **values))
    generated = {row["trajectory_id"]: row for row in json.loads((root / "simulation_manifest.json").read_text())["scenario_rows"]}
    if set(generated) != {row["trajectory_id"] for row in planned} or len(generated) != 576:
        raise ValueError("expanded trajectory inventory differs from the frozen registry")
    for row in original_rows:
        if any(generated[row["trajectory_id"]][key] != row[key] for key in ("raw_sha256", "ground_truth_sha256", "observation_seed")):
            raise ValueError("training expansion did not retain the complete initial dataset")
    if not result.split_identity["disjoint"] or not all(row["vehicle_values_finite"] and row["physiology_values_finite"]
        and row["all_states_present"] and row["event_count"] >= 2 for row in result.validation_rows):
        raise ValueError("expanded simulation failed generation acceptance")
    audit = {"registry_sha256": contract["registry_sha256"], "extension_contract_sha256": sha256_file(contract_path),
        "simulation_manifest_sha256": sha256_file(root / "simulation_manifest.json"), "roles": {"train": 512, "validation": 64},
        "initial_subset_preserved": True, "initial_trajectory_hashes_verified": len(original_rows),
        "additional_training_profiles": 32, "additional_training_trajectories": 256, "activation_completed": True,
        "clean_only": True, "confirmation_generated": False, "model_metrics_opened": False,
        "elapsed_s": time.perf_counter() - started, "split_identity": result.split_identity}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit
