"""Task-independent production adapters for unified fusion stream export."""

from chronaris.modeling.fusion_encoders.causal_query import (
    CausalQueryStream,
    causal_query_stream,
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

__all__ = [
    "CausalQueryStream",
    "ContinuousTimeSingleStreamEncoder",
    "NaiveTimeSyncConfig",
    "NaiveTimeSyncEncoder",
    "NaiveTimeSyncFusionAdapter",
    "SingleStreamEncoderConfig",
    "SingleStreamFusionAdapter",
    "causal_query_stream",
    "load_single_stream_checkpoint",
    "load_naive_time_sync_checkpoint",
    "save_naive_time_sync_checkpoint",
    "save_single_stream_checkpoint",
]
