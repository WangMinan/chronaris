"""Task-independent production adapters for unified fusion stream export."""

from chronaris.modeling.fusion_encoders.causal_query import (
    CausalQueryStream,
    causal_query_stream,
)
from chronaris.modeling.fusion_encoders.alignment_bridge import (
    build_alignment_batch_from_observations,
)
from chronaris.modeling.fusion_encoders.deep_baselines import (
    CausalContiFormerFusionEncoder,
    CausalMulTFusionEncoder,
    DeepBaselineEncoderConfig,
    DeepBaselineFusionAdapter,
    load_deep_baseline_checkpoint,
    save_deep_baseline_checkpoint,
)
from chronaris.modeling.fusion_encoders.chronaris_continuous import (
    ABLATION_TARGET_FIELDS,
    CHRONARIS_VARIANTS,
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousEncoding,
    ChronarisContinuousFusionAdapter,
    ChronarisContinuousFusionEncoder,
    build_chronaris_ablation_configs,
    chronaris_ablation_diff,
    load_chronaris_continuous_checkpoint,
    save_chronaris_continuous_checkpoint,
    validate_chronaris_ablation_diff,
)
from chronaris.modeling.fusion_encoders.chronaris_physics import (
    ChronarisPhysicsAudit,
    PhysicsComponentStatus,
    build_chronaris_physics_audit,
    physics_audit_to_rows,
)
from chronaris.modeling.fusion_encoders.chronaris_v2 import (
    V2_SUBSPACE_SLICES,
    ChronarisV2EncoderConfig,
    ChronarisV2Encoding,
    ChronarisV2FusionEncoder,
    load_chronaris_v2_checkpoint,
    save_chronaris_v2_checkpoint,
)
from chronaris.modeling.fusion_encoders.learned_causal import (
    FIVE_LAG_RANGES_S,
    LearnedRelativeCausalFusion,
    LearnedRelativeCausalFusionConfig,
    LearnedRelativeCausalFusionInput,
    LearnedRelativeCausalFusionOutput,
)
from chronaris.modeling.fusion_encoders.semantic_groups import (
    VEHICLE_SEMANTIC_GROUPS,
    SemanticGroupedObservationEncoder,
    VehicleSemanticGroupMap,
    build_vehicle_semantic_group_map,
)
from chronaris.modeling.fusion_encoders.single_stream import (
    ContinuousTimeSingleStreamEncoder,
    SingleStreamEncoderConfig,
    SingleStreamFusionAdapter,
    load_single_stream_checkpoint,
    save_single_stream_checkpoint,
)
from chronaris.modeling.fusion_encoders.naive_sync import (
    NaiveTimeSyncConfig,
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.fusion_encoders.multiscale_causal import (
    DEFAULT_LAG_RANGES_S,
    MultiScaleCausalFusionConfig,
    MultiScaleCausalFusionInput,
    MultiScaleCausalFusionOutput,
    MultiScaleCausalLagFusion,
    build_seconds_lag_mask,
)

__all__ = [
    "CausalQueryStream",
    "ABLATION_TARGET_FIELDS",
    "CHRONARIS_VARIANTS",
    "CausalContiFormerFusionEncoder",
    "CausalMulTFusionEncoder",
    "ContinuousTimeSingleStreamEncoder",
    "ChronarisContinuousEncoderConfig",
    "ChronarisContinuousEncoding",
    "ChronarisContinuousFusionAdapter",
    "ChronarisContinuousFusionEncoder",
    "ChronarisPhysicsAudit",
    "ChronarisV2EncoderConfig",
    "ChronarisV2Encoding",
    "ChronarisV2FusionEncoder",
    "DeepBaselineEncoderConfig",
    "DeepBaselineFusionAdapter",
    "DEFAULT_LAG_RANGES_S",
    "FIVE_LAG_RANGES_S",
    "LearnedRelativeCausalFusion",
    "LearnedRelativeCausalFusionConfig",
    "LearnedRelativeCausalFusionInput",
    "LearnedRelativeCausalFusionOutput",
    "MultiScaleCausalFusionConfig",
    "MultiScaleCausalFusionInput",
    "MultiScaleCausalFusionOutput",
    "MultiScaleCausalLagFusion",
    "NaiveTimeSyncConfig",
    "NaiveTimeSyncEncoder",
    "NaiveTimeSyncFusionAdapter",
    "PhysicsComponentStatus",
    "SemanticGroupedObservationEncoder",
    "SingleStreamEncoderConfig",
    "SingleStreamFusionAdapter",
    "V2_SUBSPACE_SLICES",
    "VEHICLE_SEMANTIC_GROUPS",
    "VehicleSemanticGroupMap",
    "causal_query_stream",
    "build_seconds_lag_mask",
    "build_alignment_batch_from_observations",
    "build_chronaris_ablation_configs",
    "build_chronaris_physics_audit",
    "build_vehicle_semantic_group_map",
    "chronaris_ablation_diff",
    "load_single_stream_checkpoint",
    "load_deep_baseline_checkpoint",
    "load_naive_time_sync_checkpoint",
    "load_chronaris_continuous_checkpoint",
    "load_chronaris_v2_checkpoint",
    "physics_audit_to_rows",
    "save_naive_time_sync_checkpoint",
    "save_deep_baseline_checkpoint",
    "save_single_stream_checkpoint",
    "save_chronaris_continuous_checkpoint",
    "save_chronaris_v2_checkpoint",
    "validate_chronaris_ablation_diff",
]
