"""Native Dingxin observations and targets fitted only on internal training blocks."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from chronaris.evaluation.application_tasks.dingxin_context_data import build_dingxin_lazy_context_index, dingxin_vehicle_field_labels
from chronaris.evaluation.application_tasks.dingxin_target_data import load_dingxin_target_source_data
from chronaris.evaluation.dingxin.simple_downstream_protocol import extract_simple_raw_targets, fit_simple_loso_targets
from chronaris.representation import DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY, FoldLineage


def dingxin_v4_inner_folds(contexts):
    """Last six vehicle blocks validate; overlapping complete supports are embargoed."""
    folds, embargo = [], {}
    for number, sortie in enumerate(sorted(contexts.sortie_id.unique()), 1):
        fold_id = f"leave_one_sortie_out__fold{number:02d}"
        pool = contexts[contexts.sortie_id != sortie]
        blocks = pool.drop_duplicates("vehicle_context_id").sort_values("target_start_offset_ms")
        validation = blocks.iloc[-6:]
        earliest_validation_input = int(validation.input_start_offset_ms.min())
        train = blocks[blocks.target_end_exclusive_ms <= earliest_validation_input]
        excluded = set(blocks.vehicle_context_id) - set(train.vehicle_context_id) - set(validation.vehicle_context_id)
        if (len(train), len(validation), len(excluded)) != (18, 6, 6):
            raise ValueError("Dingxin complete-support split differs from approved 18/6/6 blocks")
        folds.append(FoldLineage(fold_id, tuple(pool[pool.vehicle_context_id.isin(train.vehicle_context_id)].context_id),
            tuple(pool[pool.vehicle_context_id.isin(validation.vehicle_context_id)].context_id),
            tuple(contexts[contexts.sortie_id == sortie].context_id)))
        embargo[fold_id] = tuple(pool[pool.vehicle_context_id.isin(excluded)].context_id)
    return tuple(folds), embargo


@dataclass(frozen=True)
class V4DingxinData:
    index: object
    folds: tuple[FoldLineage, ...]
    embargo: dict
    raw_targets: object
    fitted_targets: object
    targets_by_fold: dict
    definitions_by_fold: dict
    sampling_by_fold: dict
    source_hashes: dict
    vehicle_field_labels: tuple
    data_manifest_sha256: str

    def development_provider(self, fold):
        allowed = set(fold.train_sample_ids + fold.validation_sample_ids)

        def load(ids):
            if not set(ids) <= allowed:
                raise ValueError("Dingxin development requested confirmation or embargo contexts")
            return self.index.load_batch(ids)
        return load


def load_v4_dingxin_development(*, snapshot_root="artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot",
                                audit_root="docs/artifacts/runs/2026-07-10_fixed-data-audit"):
    root = Path(audit_root)
    source = load_dingxin_target_source_data(fixed_audit_root=root, snapshot_root=snapshot_root)
    raw = extract_simple_raw_targets(source, snapshot_root=snapshot_root)
    folds, embargo = dingxin_v4_inner_folds(raw.contexts)
    fitted = fit_simple_loso_targets(raw, fit_context_ids_by_fold={fold.fold_id: fold.train_sample_ids for fold in folds})
    index = build_dingxin_lazy_context_index(snapshot_root=snapshot_root,
        field_role_manifest_path=root / "field_role_manifest.csv", context_manifest_path=root / "context_sample_manifest.jsonl",
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY)
    targets, definitions, sampling = {}, {}, {}
    for fold in folds:
        ids = fold.train_sample_ids + fold.validation_sample_ids
        maneuver = fitted.maneuver_targets[fitted.maneuver_targets.fold_id == fold.fold_id].set_index("context_id").loc[list(ids)]
        physiology = fitted.physiology_targets[(fitted.physiology_targets.fold_id == fold.fold_id) & fitted.physiology_targets.selected]
        fields = tuple(sorted(physiology.field_name.unique()))
        field_values = physiology.pivot(index="context_id", columns="field_name", values="future_value").loc[list(ids), list(fields)].to_numpy(dtype=np.float32)
        values = {"maneuver_regression": torch.tensor(maneuver.future_maneuver_score.to_numpy(dtype=np.float32)),
                  "maneuver_classification": torch.tensor(maneuver.future_maneuver_class.to_numpy(dtype=np.int64)),
                  "physiology_regression": torch.from_numpy(field_values)}
        targets[fold.fold_id] = ApplicationTaskTargets(ids, values,
            {name: torch.isfinite(value) for name, value in values.items()},
            {"domain": "dingxin", "fit_scope": "inner_training", "fit_sample_ids": list(fold.train_sample_ids),
             "physiology_field_order": list(fields), "future_duration_s": 5.},
            torch.tensor(maneuver.sample_weight.to_numpy(dtype=np.float32)))
        definitions[fold.fold_id] = (ApplicationTaskDefinition("maneuver_regression", "regression", 1),
            ApplicationTaskDefinition("maneuver_classification", "classification", 3),
            ApplicationTaskDefinition("physiology_regression", "regression", len(fields)))
        sampling[fold.fold_id] = {sample: (str(maneuver.loc[sample, "vehicle_context_id"]), str(maneuver.loc[sample, "view_id"]))
                                 for sample in fold.train_sample_ids}
    manifest = dict(source_hashes=source.source_hashes, folds=[fold.to_dict() for fold in folds], embargo=embargo,
        schema_sha256=index.plan.schema.schema_sha256, maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
        input_temporal_contract="all_native_points", target_fit="inner_training_only")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return V4DingxinData(index, folds, embargo, raw, fitted, targets, definitions, sampling, source.source_hashes,
        dingxin_vehicle_field_labels(index, field_role_manifest_path=root / "field_role_manifest.csv"), digest)
