from __future__ import annotations

import hashlib
from pathlib import Path

from chronaris.evaluation.application_tasks.application_consumer_representations import (
    APPLICATION_METHODS,
    export_application_context_representations,
)
from chronaris.representation import ContractProbeEncoder, collate_observation_samples
from tests.representation.test_contracts import _sample


def test_application_context_export_is_aligned_and_resumable(tmp_path: Path) -> None:
    batch = collate_observation_samples(
        [_sample("train"), _sample("validation"), _sample("held_out")]
    )
    adapters = {
        method: ContractProbeEncoder(
            method_name=method,
            fold_id="fold_a",
            checkpoint_sha256=hashlib.sha256(method.encode()).hexdigest(),
        )
        for method in APPLICATION_METHODS
    }
    roles = {
        "train": ("train",),
        "validation": ("validation",),
        "held_out": ("held_out",),
    }
    first, first_rows, alignment = export_application_context_representations(
        adapters=adapters,
        batch=batch,
        role_sample_ids=roles,
        output_root=tmp_path,
        resume=True,
    )
    second, second_rows, second_alignment = export_application_context_representations(
        adapters=adapters,
        batch=batch,
        role_sample_ids=roles,
        output_root=tmp_path,
        resume=True,
    )

    assert len(first_rows) == 18
    assert {row["status"] for row in first_rows} == {"completed"}
    assert {row["status"] for row in second_rows} == {"resumed"}
    assert alignment == second_alignment
    assert first["chronaris"]["held_out"].sequence_embedding.shape == (1, 96, 64)
    assert second["mult"]["validation"].sample_ids == ("validation",)
