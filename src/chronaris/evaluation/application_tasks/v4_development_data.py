"""Shared full-development inputs, with an explicit small engineering subset."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS, select_application_targets
from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development, simulation_sampling_hierarchy
from chronaris.representation import select_observation_batch
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.models.alignment.calibrated_physics import SIMULATION_RELATIONS, fit_physics_calibration
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _hash_prefix(ids, count):
    return tuple(sorted(ids, key=lambda value: hashlib.sha256(f"v4-engineering-smoke:{value}".encode()).hexdigest())[:count])


def v4_workflow_source_sha256():
    """Bind orchestration/evidence reuse to data, training, targets and consumer code."""
    root = Path(__file__).parents[4]
    paths = sorted((root / "src/chronaris").rglob("*.py"))
    paths.append(root / "scripts/evaluation/application_tasks/run_thesis_v4.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_development_inputs(domain, data_root, registry_path, *, smoke=False, fold_index=0):
    if fold_index < 0 or fold_index >= ({"simulation": 1, "dingxin": 2}.get(domain, 3)):
        raise ValueError("invalid development fold index")
    simulation = None
    if domain in {"cogpilot", "clare"}:
        data = load_prepared_public_development(domain, output_root=data_root, registry_path=registry_path)
        registry = json.loads(Path(registry_path).read_text())
        fold = data.fold(registry["domains"][domain]["folds"]["development"][fold_index])
        if smoke:
            fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke",
                train_sample_ids=_hash_prefix(fold.train_sample_ids, 32), validation_sample_ids=_hash_prefix(fold.validation_sample_ids, 8))
        provider, schema = data.dataset.batch_provider, data.dataset.schema
        hierarchy, digest = data.sampling_hierarchy(fold), data.prepared_manifest_sha256
        targets, definitions = data.targets, data.task_definitions
    elif domain == "dingxin":
        data = load_v4_dingxin_development()
        fold = data.folds[fold_index]
        provider, schema = data.development_provider(fold), data.index.plan.schema
        hierarchy, digest = data.sampling_by_fold[fold.fold_id], data.data_manifest_sha256
        targets, definitions = data.targets_by_fold[fold.fold_id], data.definitions_by_fold[fold.fold_id]
        if smoke:
            fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke")
    elif domain == "simulation":
        data, fold = load_v4_simulation_development(simulation_root="artifacts/application_evaluation/2026-09-06_thesis-v4-simulation-development",
            registry_path="docs/requirements/thesis-v4-simulation-manifest.json")
        if smoke:
            fold = replace(fold, fold_id=fold.fold_id + "__engineering_smoke",
                train_sample_ids=_hash_prefix(fold.train_sample_ids, 32), validation_sample_ids=_hash_prefix(fold.validation_sample_ids, 8))
        selected = fold.train_sample_ids + fold.validation_sample_ids
        simulation = replace(data, batch=select_observation_batch(data.batch, selected),
            role_sample_ids={role: getattr(fold, role + "_sample_ids") for role in ("train", "validation", "held_out")},
            sample_manifest_rows=tuple(next(row for row in data.sample_manifest_rows if row["sample_id"] == sample) for sample in selected))
        provider, schema = lambda ids: select_observation_batch(simulation.batch, ids), data.schema
        hierarchy, digest = simulation_sampling_hierarchy(data, fold), sha256_file("docs/requirements/thesis-v4-simulation-manifest.json")
        targets, definitions = None, SIMULATION_TASKS
    else:
        raise ValueError("unknown v4 development domain")
    allowed = set(fold.train_sample_ids + fold.validation_sample_ids)
    if targets is not None:
        ids = fold.train_sample_ids + fold.validation_sample_ids
        targets = replace(targets, sample_ids=ids, **select_application_targets(targets, ids, "cpu"))
    def guarded(ids):
        if not set(ids) <= allowed:
            raise ValueError("development cannot open confirmation observations")
        return provider(ids)
    return guarded, schema, fold, hierarchy, digest, targets, definitions, simulation if domain == "simulation" else data


def development_normalization(domain, provider, schema, fold, digest, *,
                              cache_root="artifacts/application_evaluation/2026-09-06_v4-development-normalizers-repair"):
    path = Path(cache_root) / domain / fold.fold_id / "normalization.json"
    normalization_source = Path(__file__).parents[2] / "representation/normalization.py"
    physics_source = Path(__file__).parents[2] / "models/alignment/calibrated_physics.py"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["data_manifest_sha256"] != digest or saved["fold"] != fold.to_dict()
            or saved["normalization_source_sha256"] != sha256_file(normalization_source)
            or saved["physics_source_sha256"] != sha256_file(physics_source)):
            raise ValueError("development normalization sources or data roles changed")
        return TrainOnlyRobustNormalizer.from_manifest(saved["normalizer"]), saved["physics_calibration"]
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
        train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=4)
    calibration = fit_physics_calibration(normalizer, provider, train_sample_ids=fold.train_sample_ids,
        vehicle_feature_names=schema.vehicle_feature_names, relations=SIMULATION_RELATIONS if domain == "simulation" else ())
    return normalizer, calibration
