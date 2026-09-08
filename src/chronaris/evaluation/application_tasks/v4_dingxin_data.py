"""Native Dingxin observations and targets fitted only on internal training blocks."""
from dataclasses import dataclass, replace
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
        targets[fold.fold_id], definitions[fold.fold_id], sampling[fold.fold_id] = _task_targets(fitted, fold, ids, "inner_training")
    manifest = dict(source_hashes=source.source_hashes, folds=[fold.to_dict() for fold in folds], embargo=embargo,
        schema_sha256=index.plan.schema.schema_sha256, maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY,
        input_temporal_contract="all_native_points", target_fit="inner_training_only")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return V4DingxinData(index, folds, embargo, raw, fitted, targets, definitions, sampling, source.source_hashes,
        dingxin_vehicle_field_labels(index, field_role_manifest_path=root / "field_role_manifest.csv"), digest)


def _task_targets(fitted, fold, ids, scope):
    maneuver = fitted.maneuver_targets[fitted.maneuver_targets.fold_id == fold.fold_id].set_index("context_id").loc[list(ids)]
    physiology = fitted.physiology_targets[(fitted.physiology_targets.fold_id == fold.fold_id) & fitted.physiology_targets.selected]
    fields = tuple(sorted(physiology.field_name.unique()))
    field_values = physiology.pivot(index="context_id", columns="field_name", values="future_value").loc[list(ids), list(fields)].to_numpy(dtype=np.float32)
    values = {"maneuver_regression": torch.tensor(maneuver.future_maneuver_score.to_numpy(dtype=np.float32)),
              "maneuver_classification": torch.tensor(maneuver.future_maneuver_class.fillna(-1).to_numpy(dtype=np.int64)),
              "physiology_regression": torch.from_numpy(field_values)}
    masks = {name: torch.isfinite(value) for name, value in values.items()}
    masks["maneuver_classification"] &= masks["maneuver_regression"] & (values["maneuver_classification"] >= 0)
    targets = ApplicationTaskTargets(ids, values, masks,
        {"domain": "dingxin", "fit_scope": scope, "fit_sample_ids": list(fold.train_sample_ids),
         "physiology_field_order": list(fields), "future_duration_s": 5.},
        torch.tensor(maneuver.sample_weight.to_numpy(dtype=np.float32)))
    definitions = (ApplicationTaskDefinition("maneuver_regression", "regression", 1),
        ApplicationTaskDefinition("maneuver_classification", "classification", 3),
        ApplicationTaskDefinition("physiology_regression", "regression", len(fields)))
    sampling = {sample: (str(maneuver.loc[sample, "vehicle_context_id"]), str(maneuver.loc[sample, "view_id"]))
                for sample in fold.train_sample_ids}
    return targets, definitions, sampling


def build_dingxin_outer_consumer_inputs(data, fold_id):
    """Data-only target contract; actual consumer fitting follows encoder freeze."""
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
    inner = next(fold for fold in data.folds if fold.fold_id == fold_id)
    held = set(inner.held_out_sample_ids)
    train = tuple(str(sample) for sample in data.raw_targets.contexts.context_id if sample not in held)
    if set(train) != set(inner.train_sample_ids + inner.validation_sample_ids + data.embargo[fold_id]):
        raise ValueError("Dingxin outer training membership differs from the complete sortie")
    outer = FoldLineage(fold_id, train, (), inner.held_out_sample_ids)
    fitted = fit_simple_loso_targets(data.raw_targets)
    targets, definitions, _ = _task_targets(fitted, outer, train + outer.held_out_sample_ids,
                                           "outer_training_after_encoder_freeze")
    context_data = replace(data, fitted_targets=fitted, targets_by_fold={fold_id: targets})
    return {"fold": outer, "targets": targets, "definitions": definitions,
            "context": native_consumer_context("dingxin", context_data, outer, include_held_out=True),
            "encoder_fold": inner, "data_manifest_sha256": data.data_manifest_sha256}


def audit_dingxin_vehicle_reuse(data):
    """Content identity ignores sortie names and padding, retaining actual vehicle history."""
    contexts=data.raw_targets.contexts.drop_duplicates('vehicle_context_id')
    rows=[]
    for context in contexts.itertuples(index=False):
        batch=data.index.load_batch((str(context.context_id),))
        valid=batch.vehicle_point_mask[0]
        mask=batch.vehicle_feature_mask[0,valid]
        values=batch.vehicle_values[0,valid].masked_fill(~mask,0)
        digest=hashlib.sha256(data.data_manifest_sha256.encode())
        for tensor in (batch.vehicle_timestamps_s[0,valid],values,mask,batch.query_timestamps_s[0]):
            array=tensor.cpu().numpy();digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
        rows.append(dict(vehicle_context_id=str(context.vehicle_context_id),sortie_id=str(context.sortie_id),
                         content_sha256=digest.hexdigest(),target_start_offset_ms=int(context.target_start_offset_ms)))
    signature={row['vehicle_context_id']:row['content_sha256'] for row in rows}
    by_sample={str(row.context_id):signature[str(row.vehicle_context_id)] for row in data.raw_targets.contexts.itertuples(index=False)}
    folds=[]
    for fold in data.folds:
        held={by_sample[s] for s in fold.held_out_sample_ids}
        train={by_sample[s] for s in fold.train_sample_ids}
        validation={by_sample[s] for s in fold.validation_sample_ids}
        outer={by_sample[s] for s in fold.train_sample_ids+fold.validation_sample_ids+data.embargo[fold.fold_id]}
        folds.append(dict(fold_id=fold.fold_id,inner_training_held_out_shared_contents=len(train&held),
            validation_held_out_shared_contents=len(validation&held),outer_training_held_out_shared_contents=len(outer&held)))
    return dict(format='chronaris.v4_dingxin_vehicle_content_audit.v1',data_manifest_sha256=data.data_manifest_sha256,
        declared_vehicle_contexts=len(rows),unique_vehicle_contents=len(set(signature.values())),
        context_rows=rows,folds=folds,outer_roles_disjoint=all(row['outer_training_held_out_shared_contents']==0 for row in folds))
