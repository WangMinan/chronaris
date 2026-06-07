"""Dataset construction and windowing utilities."""

from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.dataset.nasa_csm_stage_i import NASACSMPreparedTaskSet, build_nasa_csm_task_entries
from chronaris.dataset.stage_i_contracts import (
    StageIDatasetSummary,
    StageITaskEntry,
    dump_stage_i_summary,
    dump_stage_i_task_entries,
    isoformat_utc,
    load_stage_i_task_entries,
)
from chronaris.dataset.stage_i_private_contracts import (
    StageIPrivateTaskEntry,
    dump_stage_i_private_task_entries,
    load_stage_i_private_task_entries,
)
from chronaris.dataset.stage_i_real_task_builders import (
    TASK_EVENT_REPLAY_TAG,
    TASK_RISK_PROXY,
    TASK_WORKLOAD_PROXY,
    THESIS_WEAK_LABEL_BENCHMARK_ROLE,
    THESIS_WEAK_LABEL_BOUNDARY,
    THESIS_WEAK_LABEL_ROLE,
    build_stage_i_real_task_payload,
)
from chronaris.dataset.stage_i_sequence_contracts import (
    StageISequenceBundle,
    StageISequenceDatasetSummary,
    StageISequenceEntry,
    dump_stage_i_sequence_entries,
    dump_stage_i_sequence_summary,
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    load_stage_i_sequence_summary,
    save_stage_i_sequence_bundle,
)
from chronaris.dataset.streaming_windows import (
    StreamingPointEvent,
    StreamingWindowBuffer,
    StreamingWindowBufferDiagnostics,
    iter_aligned_sortie_events,
)
from chronaris.dataset.uab_stage_i import UABPreparedTaskSet, build_uab_task_entries
from chronaris.dataset.timebase import ReferenceStrategy, TimebaseError, TimebasePolicy, align_sortie_bundle
from chronaris.dataset.windows import build_sample_windows

__all__ = [
    "NASACSMPreparedTaskSet",
    "ReferenceStrategy",
    "StageIDatasetSummary",
    "StageIPrivateTaskEntry",
    "StageISequenceBundle",
    "StageISequenceDatasetSummary",
    "StageISequenceEntry",
    "StageITaskEntry",
    "StreamingPointEvent",
    "StreamingWindowBuffer",
    "StreamingWindowBufferDiagnostics",
    "TASK_EVENT_REPLAY_TAG",
    "TASK_RISK_PROXY",
    "TASK_WORKLOAD_PROXY",
    "THESIS_WEAK_LABEL_BENCHMARK_ROLE",
    "THESIS_WEAK_LABEL_BOUNDARY",
    "THESIS_WEAK_LABEL_ROLE",
    "SortieDatasetBuilder",
    "TimebaseError",
    "TimebasePolicy",
    "UABPreparedTaskSet",
    "align_sortie_bundle",
    "build_sample_windows",
    "build_nasa_csm_task_entries",
    "build_stage_i_real_task_payload",
    "build_uab_task_entries",
    "dump_stage_i_sequence_entries",
    "dump_stage_i_sequence_summary",
    "dump_stage_i_private_task_entries",
    "dump_stage_i_summary",
    "dump_stage_i_task_entries",
    "isoformat_utc",
    "iter_aligned_sortie_events",
    "load_stage_i_sequence_bundle",
    "load_stage_i_sequence_entries",
    "load_stage_i_sequence_summary",
    "load_stage_i_private_task_entries",
    "load_stage_i_task_entries",
    "save_stage_i_sequence_bundle",
]
