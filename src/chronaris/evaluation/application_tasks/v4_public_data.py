"""Fixed subject roles and complete native windows for the v4 public tasks."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from chronaris.dataset.clare_native import build_clare_native_dataset, CENTRAL_COLUMNS, CLARE_SCHEMA
from chronaris.dataset.cogpilot_native import (build_cogpilot_difficulty_dataset,
    build_cogpilot_event_response_dataset, _load_native_sample, _recording_bounds, COGPILOT_SCHEMA)
from chronaris.dataset.lazy_observed import LazyObservedDataset, NativeSampleRecord
from chronaris.dataset.native_table_cache import read_native_table
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from chronaris.evaluation.application_tasks.thesis_native_data import COGPILOT_ROOT, CLARE_ROOT
from chronaris.representation import FoldLineage
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


DEVELOPMENT_SUBJECTS = {
    "cogpilot": tuple(f"sub-cp{value:03d}" for value in (3, 4, 8, 11, 19, 22, 25)),
    "clare": ("1390", "1624", "1629", "1674"),
}
PUBLIC_TASKS = {
    "cogpilot": (ApplicationTaskDefinition("difficulty", "classification", 4),
                 ApplicationTaskDefinition("event_response", "regression", 1)),
    "clare": (ApplicationTaskDefinition("workload_classification", "classification", 2),
              ApplicationTaskDefinition("workload_regression", "regression", 1)),
}


def _hash_order(values, salt):
    return tuple(sorted(values, key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()))


def public_subject_registry(*, cogpilot_root=COGPILOT_ROOT, clare_root=CLARE_ROOT):
    domains = {
        "cogpilot": tuple(sorted(path.name for path in Path(cogpilot_root).glob("sub-cp*") if path.is_dir())),
        "clare": tuple(sorted(path.name for path in (Path(clare_root) / "EEG").glob("[0-9]*") if path.is_dir())),
    }
    result = {"format": "chronaris.v4_public_subject_roles.v1", "training_seed_changes_roles": False, "domains": {}}
    for domain, subjects in domains.items():
        expected = 35 if domain == "cogpilot" else 19
        development = DEVELOPMENT_SUBJECTS[domain]
        if len(subjects) != expected or not set(development) <= set(subjects):
            raise ValueError(f"{domain} fixed subject inventory does not match the approved plan")
        confirmation = tuple(value for value in subjects if value not in development)
        folds = {}
        for role, groups, count in (("development", development, 3), ("confirmation", confirmation, 5)):
            ordered = _hash_order(groups, f"chronaris-v4:{domain}:{role}")
            rows = []
            for index in range(count):
                held = ordered[index::count]
                pool = tuple(value for value in ordered if value not in held)
                if role == "development":
                    train, validation, held_out = pool, held, ()
                else:
                    inner_order = _hash_order(pool, f"chronaris-v4:{domain}:inner:{index}")
                    validation = inner_order[:max(1, round(len(inner_order) * .2))]
                    train, held_out = tuple(value for value in inner_order if value not in validation), held
                rows.append({"fold_id": f"v4_{domain}_{role}_fold{index + 1:02d}",
                    "train_subjects": list(train), "validation_subjects": list(validation), "held_out_subjects": list(held_out)})
            folds[role] = rows
        result["domains"][domain] = {"development_subjects": list(development),
            "confirmation_subjects": list(confirmation), "folds": folds,
            "evidence_scope": "frozen_grouped_evaluation", "historical_exposure_inventory": "pending"}
    return result


def freeze_public_subject_registry(path, **roots):
    path = Path(path)
    payload = public_subject_registry(**roots)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("frozen public subject roles changed")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


@dataclass(frozen=True)
class V4PublicData:
    domain: str
    dataset: LazyObservedDataset
    targets: ApplicationTaskTargets
    task_definitions: tuple[ApplicationTaskDefinition, ...]
    sample_manifest: tuple[dict, ...]
    prepared_manifest_sha256: str | None = None

    def sampling_hierarchy(self, fold):
        train = set(fold.train_sample_ids)
        return {row["sample_id"]: (row["subject_id"], row["record_id"],
            "+".join(name for name, valid in row["task_valid_mask"].items() if valid))
            for row in self.sample_manifest if row["sample_id"] in train}

    def fold(self, subject_fold):
        role_ids = {role: tuple(record.sample_id for record in self.dataset.records
            if record.group_id in subject_fold[f"{role}_subjects"]) for role in ("train", "validation", "held_out")}
        return FoldLineage(fold_id=subject_fold["fold_id"], train_sample_ids=role_ids["train"],
            validation_sample_ids=role_ids["validation"], held_out_sample_ids=role_ids["held_out"],
            development_only=not bool(subject_fold["held_out_subjects"]))


def load_v4_public_development(domain, *, registry, cache_root, cogpilot_root=COGPILOT_ROOT, clare_root=CLARE_ROOT):
    subjects = tuple(registry["domains"][domain]["development_subjects"])
    return _load_public_subjects(domain, subjects, cache_root=cache_root,
        cogpilot_root=cogpilot_root, clare_root=clare_root)


def _load_public_subjects(domain, subjects, *, cache_root, cogpilot_root, clare_root):
    cache_root = Path(cache_root)
    if domain == "cogpilot":
        difficulty = build_cogpilot_difficulty_dataset(cogpilot_root, subject_ids=subjects,
            all_legal_windows=True, cache_root=cache_root)
        response = build_cogpilot_event_response_dataset(cogpilot_root, subject_ids=subjects,
            max_events_per_run=None, cache_root=cache_root)
        records = difficulty.records + response.records
        dataset = LazyObservedDataset(records, schema=difficulty.schema, loader=_load_native_sample, cache_root=cache_root)
        definitions = PUBLIC_TASKS[domain]
        split = len(difficulty.records)
        values = {"difficulty": torch.zeros(len(records), dtype=torch.long), "event_response": torch.zeros(len(records))}
        masks = {name: torch.zeros(len(records), dtype=torch.bool) for name in values}
        values["difficulty"][:split] = torch.tensor(difficulty.labels, dtype=torch.long)
        values["event_response"][split:] = torch.tensor(response.labels)
        masks["difficulty"][:split], masks["event_response"][split:] = True, True
    elif domain == "clare":
        dataset = build_clare_native_dataset(clare_root, subject_ids=subjects,
            all_legal_windows=True, window_stride=1, cache_root=cache_root)
        records = dataset.records
        definitions = PUBLIC_TASKS[domain]
        scores = torch.tensor(dataset.labels, dtype=torch.float32)
        values = {"workload_classification": (scores >= 7).long(), "workload_regression": scores}
        masks = {name: torch.ones(len(records), dtype=torch.bool) for name in values}
    else:
        raise ValueError(f"unsupported v4 public domain: {domain}")
    rows = []
    for index, record in enumerate(records):
        if domain == "cogpilot":
            origin = _recording_bounds(record.vehicle_path)[0]
            start = (record.window_start_native - origin) * record.time_scale
            source_paths = (*record.physiology_paths, record.ecg_path, record.vehicle_path)
            record_id = str(record.vehicle_path.parent)
            future_duration = 8. if masks["event_response"][index] else 0.
        else:
            origin = float(read_native_table(record.eeg_path, ("Timestamp", *CENTRAL_COLUMNS))["Timestamp"].min())
            start = record.window_start_s - origin
            source_paths = (record.eeg_path, record.eda_path, record.ecg_path)
            record_id = str(record.eeg_path)
            future_duration = 0.
        rows.append({"sample_id": record.sample_id, "subject_id": record.group_id, "record_id": record_id,
            "context_start_s": start, "context_end_s": start + record.context_duration_s,
            "original_support_start_s": start, "original_support_end_s": start + record.context_duration_s + future_duration,
            "record_time_origin_native": origin, "native_time_scale_s": record.time_scale if domain == "cogpilot" else 1.,
            "task_valid_mask": {name: bool(mask[index]) for name, mask in masks.items()},
            "source_sample_hash": record.source_sample_hash, "source_paths": [str(path) for path in source_paths]})
    targets = ApplicationTaskTargets(dataset.sample_ids, values, masks,
        {"domain": domain, "source_role": "fixed_development_subjects", "smoke_only": False})
    for subject in subjects:
        positions = [index for index, record in enumerate(records) if record.group_id == subject]
        if not positions or any(not mask[positions].any() for mask in masks.values()):
            raise ValueError(f"{domain} fixed development subject lacks task coverage: {subject}")
    return V4PublicData(domain, dataset, targets, definitions, tuple(rows))


def prepare_public_development(domain, *, registry_path, output_root):
    """Materialize native window caches, audit coverage and save reproducible folds."""
    from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
    output_root = Path(output_root) / domain
    registry = freeze_public_subject_registry(registry_path)
    started = time.perf_counter()
    with _periodic_training_heartbeat(f"data_{domain}", 30., root=output_root) as progress:
        progress["phase"] = "build_native_window_manifest"
        data = load_v4_public_development(domain, registry=registry, cache_root=output_root / "native_cache")
        subjects = registry["domains"][domain]["development_subjects"]
        coverage = {subject: {"window_count": 0, "physiology_empty_windows": 0, "vehicle_empty_windows": 0,
            "task_counts": {task.name: 0 for task in data.task_definitions}} for subject in subjects}
        point_counts = {"physiology": [], "vehicle": []}
        output_root.mkdir(parents=True, exist_ok=True)
        progress.update(phase="cache_and_audit_native_windows", total_samples=len(data.dataset.records))
        with (output_root / "sample_manifest.jsonl").open("w") as handle:
            for index, row in enumerate(data.sample_manifest):
                sample = data.dataset.load_sample(row["sample_id"])
                counts = {stream: int(getattr(sample, f"{stream}_feature_mask").any(axis=1).sum()) for stream in point_counts}
                group = coverage[row["subject_id"]]
                group["window_count"] += 1
                for stream, count in counts.items():
                    group[f"{stream}_empty_windows"] += int(count == 0)
                    point_counts[stream].append(count)
                for task, valid in row["task_valid_mask"].items():
                    group["task_counts"][task] += int(valid)
                handle.write(json.dumps(row | {"native_observation_point_counts": counts}, ensure_ascii=False) + "\n")
                progress["processed_samples"] = index + 1
        failures = [subject for subject, row in coverage.items()
            if any(row[f"{stream}_empty_windows"] == row["window_count"] for stream in point_counts)]
        folds = [data.fold(fold).to_dict() for fold in registry["domains"][domain]["folds"]["development"]]
        for task in data.task_definitions:
            if task.kind == "classification":
                for fold in folds:
                    positions = [i for i, sample in enumerate(data.targets.sample_ids) if sample in set(fold["train_sample_ids"])]
                    valid = data.targets.valid_masks[task.name][positions]
                    if len(torch.unique(data.targets.values[task.name][positions][valid])) < 2:
                        failures.append(f"{fold['fold_id']}:{task.name}:fewer_than_two_training_classes")
        summary = {"domain": domain, "status": "unavailable" if failures else "completed", "failures": failures,
            "subject_coverage": coverage, "sample_count": len(data.dataset.records),
            "task_counts": {name: int(mask.sum()) for name, mask in data.targets.valid_masks.items()},
            "native_point_counts": {name: {"minimum": min(values), "median": float(np.median(values)), "maximum": max(values)}
                                    for name, values in point_counts.items()},
            "registry_sha256": sha256_file(registry_path), "sample_manifest_sha256": sha256_file(output_root / "sample_manifest.jsonl"),
            "confirmation_observations_loaded": False, "model_scores_produced": False, "elapsed_s": time.perf_counter() - started}
        (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (output_root / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=2) + "\n")
        torch.save({"sample_ids": data.targets.sample_ids, "values": data.targets.values,
                    "valid_masks": data.targets.valid_masks, "manifest": data.targets.manifest}, output_root / "task_targets.pt")
        if failures:
            raise ValueError(f"fixed development coverage cannot support {domain}: {failures}")
        return summary


def load_prepared_public_development(domain, *, output_root, registry_path):
    """Read prepared windows without parsing their source recordings again."""
    root = Path(output_root) / domain
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "completed" or summary["registry_sha256"] != sha256_file(registry_path):
        raise ValueError("prepared development registry/status changed")
    if summary["sample_manifest_sha256"] != sha256_file(root / "sample_manifest.jsonl"):
        raise ValueError("prepared development sample manifest changed")
    rows = tuple(json.loads(line) for line in (root / "sample_manifest.jsonl").read_text().splitlines())
    targets = ApplicationTaskTargets(**torch.load(root / "task_targets.pt", map_location="cpu", weights_only=True))
    if targets.sample_ids != tuple(row["sample_id"] for row in rows):
        raise ValueError("prepared target sample order changed")
    schema = COGPILOT_SCHEMA if domain == "cogpilot" else CLARE_SCHEMA
    primary_tasks = [("workload_regression" if domain == "clare" else "difficulty" if row["task_valid_mask"].get("difficulty") else "event_response") for row in rows]
    records = tuple(NativeSampleRecord(sample_id=row["sample_id"], group_id=row["subject_id"],
        label=targets.values[primary_tasks[index]][index].item(),
        context_duration_s=(30. if row["task_valid_mask"].get("difficulty") else 12.) if domain == "cogpilot" else 10.,
        source_sample_hash=row["source_sample_hash"]) for index, row in enumerate(rows))
    index_path = root / "prepared_cache_index.json"
    schema_payload = json.loads(json.dumps(asdict(schema)))
    metadata = {"schema": schema_payload, "sample_manifest_sha256": summary["sample_manifest_sha256"],
                "targets_sha256": sha256_file(root / "task_targets.pt")}
    def missing(record):
        raise FileNotFoundError(f"native preparation must finish before training: {record.sample_id}")
    dataset = LazyObservedDataset(records, schema=schema, loader=missing, cache_root=root / "native_cache")
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if any(index.get(key) != value for key, value in metadata.items()):
            raise ValueError("prepared cache schema or targets changed")
    else:
        index = metadata | {"cache_file_sha256": {record.sample_id: sha256_file(dataset._cache_path(record)) for record in records}}
        index_path.write_text(json.dumps(index, indent=2) + "\n")
    dataset.cache_file_sha256 = index["cache_file_sha256"]
    if set(dataset.cache_file_sha256) != set(dataset.sample_ids):
        raise ValueError("prepared cache index does not cover the window manifest")
    return V4PublicData(domain, dataset, targets, PUBLIC_TASKS[domain], rows, sha256_file(index_path))
