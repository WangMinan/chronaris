"""Equal-budget, label-closed matched-clean six-method representation training."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch

from chronaris.evaluation.application_tasks.clean_input_contract import (
    build_clean_input_contract,
    clean_guarded_provider,
)
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    build_dingxin_lazy_context_index,
    dingxin_vehicle_field_labels,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
)
from chronaris.evaluation.application_tasks.target_reconstruction_contracts import (
    MATCHED_METHODS,
)
from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    LockedChronarisTrainingConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_locked_chronaris,
    train_pretext_candidate,
)
from chronaris.representation import (
    AugmentationPolicy,
    FoldLineage,
    ResumableOOFExporter,
    TrainOnlyRobustNormalizer,
    build_checkpoint_record,
    coalesce_observation_batch,
    load_fusion_stream_batch,
    validate_fusion_method_alignment,
)


@dataclass(frozen=True, slots=True)
class MatchedCleanTrainingConfig:
    seed: int = 17
    max_epochs: int = 12
    batch_size: int = 16
    patience: int = 4
    export_batch_size: int = 8
    device: str = "cuda"
    resume: bool = True


def train_matched_clean_representations(
    *,
    config: MatchedCleanTrainingConfig,
    plans: Sequence[Mapping[str, object]],
    heavy_root: str | Path,
    snapshot_root: str | Path,
    fixed_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train all methods with identical clean inputs before opening task targets."""

    heavy = Path(heavy_root)
    heavy.mkdir(parents=True, exist_ok=True)
    fixed = Path(fixed_root)
    role_path = fixed / "field_role_manifest.csv"
    context_path = fixed / "context_sample_manifest.jsonl"
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot_root,
        field_role_manifest_path=role_path,
        context_manifest_path=context_path,
    )
    removed, input_contract = build_clean_input_contract(
        role_path=str(role_path),
        vehicle_raw_to_index=index.plan.vehicle_raw_to_index,
        vehicle_channel_count=len(index.plan.schema.vehicle_feature_names),
    )
    def base_provider(ids):
        return coalesce_observation_batch(
            index.load_batch(ids), bin_width_s=DINGXIN_MODEL_INPUT_BIN_WIDTH_S
        )
    candidate = ENCODER_SCREEN_CANDIDATES[0]
    policy = AugmentationPolicy()
    training_rows = []
    export_rows = []
    access_rows = []
    for plan in plans:
        fold = FoldLineage(
            fold_id=str(plan["fold_id"]),
            train_sample_ids=tuple(str(value) for value in plan["train_sample_ids"]),
            validation_sample_ids=tuple(str(value) for value in plan["validation_sample_ids"]),
            held_out_sample_ids=(),
        )
        provider, access = clean_guarded_provider(
            base_provider,
            allowed_sample_ids=fold.train_sample_ids + fold.validation_sample_ids,
            removed_indices=removed,
        )
        normalizer_path = heavy / "normalizers" / fold.fold_id / "normalizer.json"
        normalizer = _load_or_fit_normalizer(
            normalizer_path,
            provider=provider,
            fold=fold,
            batch_size=config.batch_size,
        )
        checkpoint_root = heavy / "checkpoints" / f"seed_{config.seed}" / fold.fold_id
        adapters = {}
        records = {}
        for method in TRAINABLE_FUSION_METHODS:
            started = time.perf_counter()
            if method == "chronaris":
                result = train_locked_chronaris(
                    batch=None,
                    batch_provider=provider,
                    fold=fold,
                    physiology_feature_names=index.plan.schema.physiology_feature_names,
                    vehicle_feature_names=index.plan.schema.vehicle_feature_names,
                    vehicle_field_labels=dingxin_vehicle_field_labels(
                        index, field_role_manifest_path=role_path
                    ),
                    normalizer=normalizer,
                    output_root=checkpoint_root,
                    config=LockedChronarisTrainingConfig(
                        max_epochs=config.max_epochs,
                        batch_size=config.batch_size,
                        patience=config.patience,
                        seed=config.seed,
                        device=config.device,
                    ),
                    augmentation_policy=policy,
                    candidate_config=candidate,
                    resume=config.resume,
                )
            else:
                result = train_pretext_candidate(
                    method,
                    candidate=candidate,
                    batch=None,
                    batch_provider=provider,
                    fold=fold,
                    physiology_feature_names=index.plan.schema.physiology_feature_names,
                    vehicle_feature_names=index.plan.schema.vehicle_feature_names,
                    vehicle_field_labels=dingxin_vehicle_field_labels(
                        index, field_role_manifest_path=role_path
                    ),
                    normalizer=normalizer,
                    output_root=checkpoint_root,
                    config=CandidateScreenConfig(
                        max_epochs=config.max_epochs,
                        batch_size=config.batch_size,
                        patience=config.patience,
                        seed=config.seed,
                        device=config.device,
                    ),
                    augmentation_policy=policy,
                    resume=config.resume,
                )
            record = build_checkpoint_record(
                method_name=method,
                fold=fold,
                checkpoint_path=result.best_checkpoint_path,
                seed=config.seed,
            )
            encoder, _heads, loaded_normalizer, _payload = load_common_pretraining_checkpoint(
                result.best_checkpoint_path, device=config.device
            )
            adapters[method] = TrainedFusionAdapter(
                encoder=encoder,
                normalizer=loaded_normalizer,
                fold_id=fold.fold_id,
                checkpoint_sha256=record.checkpoint_sha256,
            )
            records[method] = record
            training_rows.append(
                {
                    "split_id": fold.fold_id,
                    "outer_pool_id": str(plan["outer_pool_id"]),
                    "main_selection": bool(plan["main_selection"]),
                    "seed": config.seed,
                    "method": method,
                    "candidate_id": candidate.candidate_id,
                    "status": result.status,
                    "best_epoch": result.best_epoch,
                    "completed_epochs": result.completed_epochs,
                    "stopped_early": result.stopped_early,
                    "best_public_selection_loss": result.best_public_selection_loss,
                    "training_elapsed_s": time.perf_counter() - started,
                    "checkpoint_path": result.best_checkpoint_path,
                    "checkpoint_sha256": record.checkpoint_sha256,
                    "task_targets_opened": False,
                    "outer_test_opened": False,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        naive_path = checkpoint_root / "naive_time_sync" / "best.pt"
        if not naive_path.is_file() or not config.resume:
            naive = NaiveTimeSyncEncoder().fit_from_batch_provider(
                provider,
                train_sample_ids=fold.train_sample_ids,
                held_out_sample_ids=fold.validation_sample_ids,
                normalizer=normalizer,
                batch_size=config.batch_size,
            )
            save_naive_time_sync_checkpoint(naive_path, encoder=naive)
            naive_status = "completed"
        else:
            naive_status = "resumed"
        naive_record = build_checkpoint_record(
            method_name="naive_time_sync",
            fold=fold,
            checkpoint_path=naive_path,
            seed=config.seed,
        )
        adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
            encoder=load_naive_time_sync_checkpoint(naive_path),
            fold_id=fold.fold_id,
            checkpoint_sha256=naive_record.checkpoint_sha256,
        )
        records["naive_time_sync"] = naive_record
        training_rows.append(
            {
                "split_id": fold.fold_id,
                "outer_pool_id": str(plan["outer_pool_id"]),
                "main_selection": bool(plan["main_selection"]),
                "seed": config.seed,
                "method": "naive_time_sync",
                "candidate_id": "train_only_pca",
                "status": naive_status,
                "best_epoch": 0,
                "completed_epochs": 0,
                "stopped_early": False,
                "best_public_selection_loss": None,
                "training_elapsed_s": 0.0,
                "checkpoint_path": str(naive_path),
                "checkpoint_sha256": naive_record.checkpoint_sha256,
                "task_targets_opened": False,
                "outer_test_opened": False,
            }
        )
        exporter = ResumableOOFExporter(
            heavy / "representations" / f"seed_{config.seed}" / fold.fold_id,
            resume=config.resume,
        )
        outputs = {method: {} for method in MATCHED_METHODS}
        for method in MATCHED_METHODS:
            for role in ("train", "validation"):
                result = exporter.export_from_batch_provider(
                    encoder=adapters[method],
                    batch_provider=provider,
                    checkpoint=records[method],
                    export_role=role,
                    batch_size=config.export_batch_size,
                )
                outputs[method][role] = load_fusion_stream_batch(result.output_root)
                export_rows.append(
                    {
                        "split_id": fold.fold_id,
                        "outer_pool_id": str(plan["outer_pool_id"]),
                        "main_selection": bool(plan["main_selection"]),
                        "seed": config.seed,
                        **result.to_dict(),
                        "task_targets_opened": False,
                        "outer_test_opened": False,
                    }
                )
        for role in ("train", "validation"):
            validate_fusion_method_alignment([outputs[method][role] for method in MATCHED_METHODS])
        access_rows.append(
            {
                "split_id": fold.fold_id,
                "outer_pool_id": str(plan["outer_pool_id"]),
                "main_selection": bool(plan["main_selection"]),
                **access,
                "outer_test_opened": False,
            }
        )
        del adapters, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    protocol = {
        "format": "chronaris.dingxin_matched_clean_pretraining.v1",
        "training_config": asdict(config),
        "candidate": asdict(candidate),
        "augmentation_policy": asdict(policy),
        "model_input_bin_width_s": DINGXIN_MODEL_INPUT_BIN_WIDTH_S,
        "input_feature_contract": input_contract,
        "method_names": list(MATCHED_METHODS),
        "split_count": len(plans),
        "task_targets_opened": False,
        "outer_test_opened": False,
    }
    return pd.DataFrame(training_rows), pd.DataFrame(export_rows), pd.DataFrame(access_rows), protocol


def _load_or_fit_normalizer(path, *, provider, fold, batch_size):
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalizer = TrainOnlyRobustNormalizer.from_manifest(payload)
        if normalizer.fit_sample_ids != tuple(sorted(fold.train_sample_ids)):
            raise ValueError("matched-clean normalizer lineage changed")
        return normalizer
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids,
        batch_size=batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalizer.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalizer
