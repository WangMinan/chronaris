from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from chronaris.evaluation.application_tasks.core_recovery_protocol import (
    CORE_RECOVERY_METHODS,
    OuterTestAccessGuard,
    CoreRecoveryProtocol,
    build_locked_candidate_registry,
    write_protocol_locks,
)
from chronaris.evaluation.application_tasks.core_recovery_splits import (
    ContextSupportInterval,
    build_task_decision_splits,
    purge_overlapping_training_contexts,
    select_dense_training_anchors,
)


def test_protocol_locks_exact_six_method_twelve_candidate_budget(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = CoreRecoveryProtocol(
        source_commit="1" * 40,
        branch_name="codex/chronaris-core-task-recovery-20260713",
        data_manifest_sha256=digest,
        outer_split_manifest_sha256=digest,
        task_definition_sha256=digest,
    )
    paths = write_protocol_locks(
        tmp_path / "locks",
        protocol=protocol,
        source_files={"source": source},
    )
    registry = json.loads(paths["candidate_registry"].read_text(encoding="utf-8"))
    candidates = build_locked_candidate_registry()

    assert protocol.methods == CORE_RECOVERY_METHODS
    assert len(candidates) == 5 * 12
    assert len(registry["candidates"]) == 60
    assert registry["failed_candidate_consumes_budget"] is True


def test_outer_test_guard_fails_closed_and_opens_once() -> None:
    guard = OuterTestAccessGuard(protocol_sha256="2" * 64)

    with pytest.raises(PermissionError, match="closed"):
        guard.require_authorized()
    with pytest.raises(PermissionError, match="development gate failed"):
        guard.authorize(
            locked_configuration_sha256="3" * 64,
            development_gate_passed=False,
        )
    authorization = guard.authorize(
        locked_configuration_sha256="3" * 64,
        development_gate_passed=True,
    )
    guard.require_authorized()
    assert authorization["authorized_open_count"] == 1
    with pytest.raises(PermissionError, match="already authorized"):
        guard.authorize(
            locked_configuration_sha256="3" * 64,
            development_gate_passed=True,
        )


def test_task_decision_split_purges_complete_input_and_future_support() -> None:
    frame = pd.DataFrame(
        [
            {
                "context_id": f"c{index:02d}",
                "sortie_id": "s1",
                "view_id": "v1",
                "start_offset_ms": index * 40_000,
                "end_offset_ms": index * 40_000 + 30_000,
            }
            for index in range(12)
        ]
    )
    splits = build_task_decision_splits(
        context_catalog=frame,
        eligible_context_ids=tuple(frame["context_id"]),
        minimum_train_count=4,
        minimum_evaluation_count=4,
        temporal_block_count=3,
    )

    assert len(splits) == 3
    assert all(len(split.evaluation_context_ids) == 4 for split in splits)
    assert all(set(split.train_context_ids).isdisjoint(split.evaluation_context_ids) for split in splits)


def test_overlap_and_dense_anchor_filters_use_full_support_interval() -> None:
    evaluation = ContextSupportInterval("eval", "s1", "v1", 40_000, 10_000, 45_000)
    overlapping = ContextSupportInterval("overlap", "s1", "v2", 70_000, 40_000, 75_000)
    isolated = ContextSupportInterval("isolated", "s1", "v2", 90_000, 60_000, 95_000)
    kept, purged = purge_overlapping_training_contexts(
        (overlapping, isolated),
        (evaluation,),
    )

    assert kept == (isolated,)
    assert purged == (overlapping,)
    anchors = select_dense_training_anchors(
        candidate_anchors_ms=(35_000, 70_000, 100_000),
        training_support_bounds=((0, 140_000),),
        protected_intervals=(evaluation,),
        sortie_id="s1",
    )
    assert anchors == (100_000,)
