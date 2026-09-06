"""Eight paired development conditions; confirmation observations remain sealed."""
from dataclasses import fields, replace
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeData
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import collate_observation_samples, load_simulation_observed_context
from chronaris.simulation.aviation_dual_stream.benchmark import SimulationBenchmarkConfig, SimulationSplitSpec, generate_benchmark
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from chronaris.simulation.aviation_dual_stream.profiles import sample_pilot_profile


DEVELOPMENT_CONDITIONS = (
    "clean_asynchronous", "random_missing_30pct", "contiguous_gap_15s", "contiguous_gap_30s",
    "physiology_missing", "vehicle_missing", "physiology_clock_offset_plus_1s", "physiology_response_lag_plus_15s")
TIMING_SCENARIOS = (
    ObservationScenarioConfig("clean_asynchronous"),
    ObservationScenarioConfig("physiology_clock_offset_plus_1s", physiology_clock_offset_s=1.),
    ObservationScenarioConfig("physiology_response_lag_plus_15s", additional_physiology_lag_s=15.))


def development_condition_protocol():
    return {"format": "chronaris.v4_development_conditions.v1", "conditions": list(DEVELOPMENT_CONDITIONS),
        "timing_scenarios": [scenario.to_dict() for scenario in TIMING_SCENARIOS],
        "missingness": {"random_probability_per_native_point": .3, "modalities": ["physiology", "vehicle"],
            "block_placement": "centered_within_each_30_second_context", "block_intervals_s": [[7.5, 22.5], [0., 30.]],
            "whole_modality_missing": ["physiology", "vehicle"], "retained_values_and_timestamps_unchanged": True,
            "seed_namespace": "chronaris-v4-development-mask"},
        "context_duration_s": 30., "context_starts_s": [30., 60., 90., 120.],
        "roles": ["validation"], "confirmation_generation": False, "model_scores_produced": False,
        "final_35_scenarios": "unchanged_existing_generator_protocol"}


def _generation_source_hash():
    source = Path(__file__).parents[2] / "simulation/aviation_dual_stream"
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted(source.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_development_conditions(*, output_root, clean_root, registry_path):
    """Reuse the existing generator only for clean and two separate timing factors."""
    root, clean = Path(output_root).resolve(), Path(clean_root).resolve()
    if root == clean:
        raise ValueError("development conditions require a separate output root")
    registry = json.loads(Path(registry_path).read_text())
    clean_audit = json.loads((clean / "v4_generation_audit.json").read_text())
    if clean_audit["registry_sha256"] != sha256_file(registry_path):
        raise ValueError("clean simulation registry changed")
    spec = next(row for row in registry["split_specs"] if row["role"] == "validation")
    split = SimulationSplitSpec(**{field.name: spec[field.name] for field in fields(SimulationSplitSpec)})
    if (split.generator_family, split.profile_count, split.trajectories_per_profile) != ("g1_state_space", 8, 8):
        raise ValueError("development conditions require the approved 64 development trajectories")
    profiles = {row["profile_id"]: row for row in registry["profiles"] if row["role"] == "validation"}
    planned = [row for row in registry["trajectories"] if row["role"] == "validation"]
    if len(profiles) != 8 or len(planned) != 64 or registry["context_starts_s"] != [30., 60., 90., 120.]:
        raise ValueError("development identities/contexts differ from the approved registry")
    for index in range(split.profile_count):
        profile = sample_pilot_profile(split_id=split.split_id, profile_index=index, seed=split.profile_seed_base + index)
        if any(profiles[profile.profile_id][key] != value for key, value in profile.to_dict().items()):
            raise ValueError("generator profile parameters no longer reproduce the registry")
    clean_rows = {row["trajectory_id"]: row for row in json.loads((clean / "simulation_manifest.json").read_text())["scenario_rows"]
                  if row["split_id"] == split.split_id}
    if set(clean_rows) != {row["trajectory_id"] for row in planned}:
        raise ValueError("clean trajectory identities differ from registry")
    for row in planned:
        original = clean_rows[row["trajectory_id"]]
        if original["observation_seed"] != row["observation_seed"]:
            raise ValueError("development observation seeds changed")
        if sha256_file(Path(original["scenario_manifest_path"]).with_name("raw_dual_stream.npz")) != original["raw_sha256"]:
            raise ValueError("clean observations changed")
    protocol = development_condition_protocol() | {"registry_sha256": sha256_file(registry_path),
        "generation_source_sha256": _generation_source_hash(), "split_spec": split.to_dict()}
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "development_condition_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("development condition sources/config changed; use a new output root")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    # The legacy generator can repair files in place; v4 rejects changed evidence first.
    for path in root.glob("v4_development/*/*/*/scenario_manifest.json"):
        stored = json.loads(path.read_text())
        for kind, hash_key in (("raw_dual_stream", "raw_dual_stream_sha256"), ("ground_truth", "ground_truth_sha256")):
            if sha256_file(Path(stored[kind + "_path"])) != stored[hash_key]:
                raise ValueError("existing development condition evidence changed")
    started = time.perf_counter()
    with _periodic_training_heartbeat("v4_development_conditions", 30., root=root) as progress:
        result = generate_benchmark(SimulationBenchmarkConfig(run_id=root.name, output_root=str(root.parent),
            duration_s=registry["duration_s"], truth_rate_hz=registry["truth_rate_hz"], split_specs=(split,),
            observation_scenarios=TIMING_SCENARIOS, paired_observation_seed=True, resume=True),
            progress_callback=lambda event, values: progress.update(phase=event, **values))
    generated = json.loads(Path(result.simulation_manifest_path).read_text())["scenario_rows"]
    restored = [row for row in generated if row["scenario_id"] == "clean_asynchronous"]
    if len(restored) != 64 or any(row["raw_sha256"] != clean_rows[row["trajectory_id"]]["raw_sha256"] for row in restored):
        raise ValueError("regenerated clean observations differ from the original development data")
    if not all(row["latent_hash_shared"] and row["trajectory_id_shared"] and row["scenario_ids_unique"] for row in result.paired_rows):
        raise ValueError("development timing factors changed latent truth")
    if not all(row["all_states_present"] and row["event_count"] >= 2
        and row["vehicle_values_finite"] and row["physiology_values_finite"]
        and row["vehicle_clock_mapping_max_error_s"] <= 1e-9 and row["physiology_clock_mapping_max_error_s"] <= 1e-9
        for row in result.validation_rows):
        raise ValueError("development timing generation failed finite values, event or clock checks")
    summary = {"status": "completed", "registry_sha256": protocol["registry_sha256"],
        "generation_source_sha256": protocol["generation_source_sha256"], "trajectory_count": 64,
        "stored_observation_count": len(generated), "condition_count": 8, "contexts_per_condition": 256,
        "clean_raw_hashes_identical": True, "latent_truth_shared": True,
        "confirmation_generated": False, "model_scores_produced": False, "elapsed_s": time.perf_counter() - started,
        "simulation_manifest_sha256": sha256_file(root / "simulation_manifest.json")}
    (root / "development_condition_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def apply_development_missingness(sample, condition):
    if condition not in DEVELOPMENT_CONDITIONS:
        raise ValueError("unknown development condition")
    if sample.context_duration_s != 30.:
        raise ValueError("development simulation contexts must be 30 seconds")
    updates = {}
    for stream in ("physiology", "vehicle"):
        mask = getattr(sample, stream + "_feature_mask").copy()
        time_s = getattr(sample, stream + "_timestamps_s")
        if condition == "random_missing_30pct":
            seed = hashlib.sha256(f"chronaris-v4-development-mask:{sample.sample_id}:{stream}".encode()).digest()
            rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint32))
            mask &= (rng.random(len(time_s)) >= .3)[:, None]
        elif condition.startswith("contiguous_gap_"):
            duration = 15. if condition == "contiguous_gap_15s" else 30.
            start = (30. - duration) / 2
            mask &= ~((time_s >= start) & (time_s < start + duration))[:, None]
        elif condition == stream + "_missing":
            mask[:] = False
        retained = mask.any(axis=1)
        updates[stream + "_feature_mask"] = mask[retained]
        updates[stream + "_values"] = np.where(mask, getattr(sample, stream + "_values"), 0)[retained]
        updates[stream + "_timestamps_s"] = time_s[retained]
    digest = hashlib.sha256(f"{sample.source_sample_hash}:{condition}:development_conditions.v1".encode()).hexdigest()
    return replace(sample, **updates, source_sample_hash=digest)


def load_development_condition(*, condition, output_root, registry_path):
    """Use canonical clean IDs for paired task targets; keep condition hashes separate."""
    if condition not in DEVELOPMENT_CONDITIONS:
        raise ValueError("unknown development condition")
    root = Path(output_root)
    audit = json.loads((root / "development_condition_audit.json").read_text())
    if (audit["registry_sha256"] != sha256_file(registry_path)
        or audit["simulation_manifest_sha256"] != sha256_file(root / "simulation_manifest.json")
        or audit["generation_source_sha256"] != _generation_source_hash()):
        raise ValueError("development condition evidence sources changed")
    registry = json.loads(Path(registry_path).read_text())
    scenario = condition if condition in {config.scenario_id for config in TIMING_SCENARIOS} else "clean_asynchronous"
    generated = {row["trajectory_id"]: row for row in json.loads((root / "simulation_manifest.json").read_text())["scenario_rows"]
                 if row["scenario_id"] == scenario}
    samples, rows = [], []
    for trajectory in (row for row in registry["trajectories"] if row["role"] == "validation"):
        item = generated[trajectory["trajectory_id"]]
        path = Path(item["scenario_manifest_path"]).with_name("raw_dual_stream.npz")
        if sha256_file(path) != item["raw_sha256"]:
            raise ValueError("development condition observations changed")
        for start, sample_id in zip(registry["context_starts_s"], trajectory["context_sample_ids"], strict=True):
            sample = load_simulation_observed_context(path, context_start_s=start)
            if sample.group_id != trajectory["profile_id"]:
                raise ValueError("development condition profile identity changed")
            sample = apply_development_missingness(replace(sample, sample_id=sample_id), condition)
            samples.append(sample)
            rows.append({"sample_id": sample_id, "group_id": sample.group_id, "profile_id": sample.group_id,
                "trajectory_id": trajectory["trajectory_id"], "role": "validation", "condition": condition,
                "context_start_s": start, "context_end_s": start + 30., "source_sample_hash": sample.source_sample_hash,
                "observed_path": str(path), "observed_sha256": item["raw_sha256"], "oracle_opened_for_representation": False,
                "valid_point_counts": {stream: int(getattr(sample, stream + "_feature_mask").any(axis=1).sum())
                                       for stream in ("physiology", "vehicle")}})
    batch = collate_observation_samples(samples)
    return ApplicationConsumerSmokeData(batch, samples[0].schema,
        {"train": (), "validation": batch.sample_ids, "held_out": ()}, tuple(rows))
