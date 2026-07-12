"""Task-independent observation and fusion representation infrastructure."""

from chronaris.representation.augmentation import (
    AugmentationPolicy,
    AugmentationRealization,
    build_augmentation_realization,
    build_batch_augmentation_realizations,
)
from chronaris.representation.augmentation_apply import (
    AppliedAugmentationBatch,
    AugmentationAuditRow,
    StreamAugmentationProvenance,
    apply_augmentation_realizations,
    augmentation_executor_accepts_method_name,
)
from chronaris.representation.collation import (
    ObservedDualStreamSample,
    collate_observation_samples,
    select_observation_batch,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    QUERY_POINT_COUNT,
    DualStreamObservationBatch,
    FusionStreamBatch,
    masked_mean_pool,
    shared_causal_query_valid_mask,
    FusionStreamEncoder,
    ObservationSchema,
    validate_fusion_method_alignment,
)
from chronaris.representation.contract_probe import (
    SIX_METHOD_NAMES,
    ContractProbeEncoder,
)
from chronaris.representation.loaders import (
    DingxinObservationSchemaPlan,
    DingxinSnapshotPointCache,
    build_dingxin_observation_schema_plan,
    build_dingxin_snapshot_point_cache,
    load_dingxin_observed_context,
    load_simulation_observed_context,
)
from chronaris.representation.lineage import (
    CheckpointRecord,
    CheckpointRegistry,
    FoldLineage,
    build_checkpoint_record,
    verify_fit_sample_isolation,
)
from chronaris.representation.normalization import (
    TrainOnlyPCAProjector,
    TrainOnlyRobustNormalizer,
)
from chronaris.representation.oof_export import (
    OOFExportResult,
    ResumableOOFExporter,
    load_fusion_stream_batch,
    validate_oof_coverage,
    write_fusion_stream_batch,
)
from chronaris.representation.pretext_targets import (
    LAG_DISCRIMINATION_SHIFTS_S,
    CommonPretextTargets,
    LagDiscriminationInputs,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    map_augmented_query_sources_to_original,
    move_common_pretext_targets,
)
from chronaris.representation.temporal_coalescing import (
    coalesce_observation_batch,
)

__all__ = [
    "AugmentationPolicy",
    "AugmentationRealization",
    "AppliedAugmentationBatch",
    "AugmentationAuditRow",
    "FUSION_OUTPUT_DIM",
    "QUERY_POINT_COUNT",
    "DualStreamObservationBatch",
    "FusionStreamBatch",
    "masked_mean_pool",
    "shared_causal_query_valid_mask",
    "FusionStreamEncoder",
    "ObservationSchema",
    "ObservedDualStreamSample",
    "TrainOnlyPCAProjector",
    "TrainOnlyRobustNormalizer",
    "StreamAugmentationProvenance",
    "DingxinObservationSchemaPlan",
    "DingxinSnapshotPointCache",
    "CheckpointRecord",
    "CheckpointRegistry",
    "CommonPretextTargets",
    "ContractProbeEncoder",
    "FoldLineage",
    "OOFExportResult",
    "LAG_DISCRIMINATION_SHIFTS_S",
    "LagDiscriminationInputs",
    "ResumableOOFExporter",
    "SIX_METHOD_NAMES",
    "build_dingxin_observation_schema_plan",
    "build_dingxin_snapshot_point_cache",
    "build_checkpoint_record",
    "build_common_pretext_targets",
    "build_lag_discrimination_inputs",
    "build_augmentation_realization",
    "build_batch_augmentation_realizations",
    "apply_augmentation_realizations",
    "augmentation_executor_accepts_method_name",
    "collate_observation_samples",
    "coalesce_observation_batch",
    "select_observation_batch",
    "load_dingxin_observed_context",
    "load_fusion_stream_batch",
    "load_simulation_observed_context",
    "map_augmented_query_sources_to_original",
    "move_common_pretext_targets",
    "validate_fusion_method_alignment",
    "validate_oof_coverage",
    "verify_fit_sample_isolation",
    "write_fusion_stream_batch",
]
