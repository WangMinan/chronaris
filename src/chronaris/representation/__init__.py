"""Task-independent observation and fusion representation infrastructure."""

from chronaris.representation.augmentation import (
    AugmentationPolicy,
    AugmentationRealization,
    build_augmentation_realization,
    build_batch_augmentation_realizations,
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
    build_dingxin_observation_schema_plan,
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

__all__ = [
    "AugmentationPolicy",
    "AugmentationRealization",
    "FUSION_OUTPUT_DIM",
    "QUERY_POINT_COUNT",
    "DualStreamObservationBatch",
    "FusionStreamBatch",
    "FusionStreamEncoder",
    "ObservationSchema",
    "ObservedDualStreamSample",
    "TrainOnlyPCAProjector",
    "TrainOnlyRobustNormalizer",
    "DingxinObservationSchemaPlan",
    "CheckpointRecord",
    "CheckpointRegistry",
    "ContractProbeEncoder",
    "FoldLineage",
    "OOFExportResult",
    "ResumableOOFExporter",
    "SIX_METHOD_NAMES",
    "build_dingxin_observation_schema_plan",
    "build_checkpoint_record",
    "build_augmentation_realization",
    "build_batch_augmentation_realizations",
    "collate_observation_samples",
    "select_observation_batch",
    "load_dingxin_observed_context",
    "load_fusion_stream_batch",
    "load_simulation_observed_context",
    "validate_fusion_method_alignment",
    "validate_oof_coverage",
    "verify_fit_sample_isolation",
    "write_fusion_stream_batch",
]
