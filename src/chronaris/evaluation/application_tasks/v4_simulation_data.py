"""Load all approved v4 development contexts without opening confirmation data."""
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeData
from chronaris.representation import FoldLineage, collate_observation_samples, load_simulation_observed_context
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_v4_simulation_development(*, simulation_root, registry_path):
    root = Path(simulation_root)
    registry = json.loads(Path(registry_path).read_text())
    audit = json.loads((root / "v4_generation_audit.json").read_text())
    if audit["registry_sha256"] != sha256_file(registry_path):
        raise ValueError("simulation parameter/context registry changed")
    if audit["roles"] != {"train": 256, "validation": 64}:
        raise ValueError("initial simulation size differs from the approved registry")
    generated = json.loads((root / "simulation_manifest.json").read_text())
    observed = {row["trajectory_id"]: row for row in generated["scenario_rows"]}
    planned = [row for row in registry["trajectories"]
               if row["role"] in {"train", "validation"} and row["activation"] == "initial"]
    if set(observed) != {row["trajectory_id"] for row in planned}:
        raise ValueError("generated simulation trajectories differ from the approved development set")
    samples, rows = [], []
    role_ids = {"train": [], "validation": [], "held_out": [sample_id
        for row in registry["trajectories"] if row["role"] == "confirmation" for sample_id in row["context_sample_ids"]]}
    for trajectory in planned:
        item = observed[trajectory["trajectory_id"]]
        if item["scenario_id"] != "clean_asynchronous":
            raise ValueError("development initialization only loads clean observations")
        path = Path(item["scenario_manifest_path"]).with_name("raw_dual_stream.npz")
        if sha256_file(path) != item["raw_sha256"]:
            raise ValueError("generated development observations changed")
        for start, sample_id in zip(registry["context_starts_s"], trajectory["context_sample_ids"], strict=True):
            sample = load_simulation_observed_context(path, context_start_s=start)
            if sample.sample_id != sample_id or sample.group_id != trajectory["profile_id"]:
                raise ValueError("simulation context/profile identity differs from registry")
            samples.append(sample)
            role_ids[trajectory["role"]].append(sample_id)
            rows.append({"sample_id": sample_id, "group_id": sample.group_id, "role": trajectory["role"],
                "profile_id": trajectory["profile_id"], "trajectory_id": trajectory["trajectory_id"],
                "context_start_s": start, "context_end_s": start + registry["context_duration_s"],
                "original_support_start_s": start, "original_support_end_s": start + registry["context_duration_s"] + 5.,
                "observed_path": str(path), "observed_sha256": item["raw_sha256"],
                "source_sample_hash": sample.source_sample_hash,
                "task_valid_mask": {"classification": True, "regression": True, "segmentation": True},
                "oracle_opened_for_representation": False})
    roles = {role: tuple(ids) for role, ids in role_ids.items()}
    fold = FoldLineage(fold_id="v4_simulation_g1_development_g2_confirmation", train_sample_ids=roles["train"],
        validation_sample_ids=roles["validation"], held_out_sample_ids=roles["held_out"])
    data = ApplicationConsumerSmokeData(collate_observation_samples(samples), samples[0].schema, roles, tuple(rows))
    return data, fold


def simulation_sampling_hierarchy(data, fold):
    train = set(fold.train_sample_ids)
    return {row["sample_id"]: (row["profile_id"], row["trajectory_id"])
            for row in data.sample_manifest_rows if row["sample_id"] in train}
