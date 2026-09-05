"""Dataset construction and windowing utilities."""

from typing import TYPE_CHECKING

from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import (
    build_cogpilot_difficulty_dataset,
    build_cogpilot_event_response_dataset,
)
from chronaris.dataset.lazy_observed import LazyObservedDataset, NativeSampleRecord
from chronaris.dataset.nasa_csm_tasks import NASACSMPreparedTaskSet, build_nasa_csm_task_entries
from chronaris.dataset.task_contracts import (
    StageIDatasetSummary,
    StageITaskEntry,
    dump_task_eval_summary,
    dump_task_eval_task_entries,
    isoformat_utc,
    load_task_eval_task_entries,
)
from chronaris.dataset.dingxin_task_contracts import (
    StageIPrivateTaskEntry,
    dump_task_eval_private_task_entries,
    load_task_eval_private_task_entries,
)
from chronaris.dataset.public_sequence_contracts import (
    StageISequenceBundle,
    StageISequenceDatasetSummary,
    StageISequenceEntry,
    dump_task_eval_sequence_entries,
    dump_task_eval_sequence_summary,
    load_task_eval_sequence_bundle,
    load_task_eval_sequence_entries,
    load_task_eval_sequence_summary,
    save_task_eval_sequence_bundle,
)
from chronaris.dataset.streaming_windows import (
    StreamingPointEvent,
    StreamingWindowBuffer,
    StreamingWindowBufferDiagnostics,
    iter_aligned_sortie_events,
)
from chronaris.dataset.uab_workload_tasks import UABPreparedTaskSet, build_uab_task_entries
from chronaris.dataset.timebase import ReferenceStrategy, TimebaseError, TimebasePolicy, align_sortie_bundle
from chronaris.dataset.windows import build_sample_windows

if TYPE_CHECKING:  # pragma: no cover - typing-only compatibility exports
    from chronaris.dataset.dingxin_task_builders import (
        TASK_EVENT_REPLAY_TAG,
        TASK_RISK_PROXY,
        TASK_WORKLOAD_PROXY,
        THESIS_WEAK_LABEL_BENCHMARK_ROLE,
        THESIS_WEAK_LABEL_BOUNDARY,
        THESIS_WEAK_LABEL_ROLE,
        attach_llm_preprocessing_context_to_task_entries,
        build_task_eval_real_task_payload,
    )

__all__ = [
    "NASACSMPreparedTaskSet",
    "LazyObservedDataset",
    "NativeSampleRecord",
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
    "attach_llm_preprocessing_context_to_task_entries",
    "build_sample_windows",
    "build_clare_native_dataset",
    "build_cogpilot_difficulty_dataset",
    "build_cogpilot_event_response_dataset",
    "build_nasa_csm_task_entries",
    "build_task_eval_real_task_payload",
    "build_uab_task_entries",
    "dump_task_eval_sequence_entries",
    "dump_task_eval_sequence_summary",
    "dump_task_eval_private_task_entries",
    "dump_task_eval_summary",
    "dump_task_eval_task_entries",
    "isoformat_utc",
    "iter_aligned_sortie_events",
    "load_task_eval_sequence_bundle",
    "load_task_eval_sequence_entries",
    "load_task_eval_sequence_summary",
    "load_task_eval_private_task_entries",
    "load_task_eval_task_entries",
    "save_task_eval_sequence_bundle",
]

_DINGXIN_TASK_BUILDER_EXPORTS = {
    "TASK_EVENT_REPLAY_TAG",
    "TASK_RISK_PROXY",
    "TASK_WORKLOAD_PROXY",
    "THESIS_WEAK_LABEL_BENCHMARK_ROLE",
    "THESIS_WEAK_LABEL_BOUNDARY",
    "THESIS_WEAK_LABEL_ROLE",
    "attach_llm_preprocessing_context_to_task_entries",
    "build_task_eval_real_task_payload",
}


def __getattr__(name: str) -> object:
    if name in _DINGXIN_TASK_BUILDER_EXPORTS:
        from chronaris.dataset import dingxin_task_builders

        return getattr(dingxin_task_builders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
