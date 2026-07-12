from __future__ import annotations

import hashlib
import json

import pytest
import pandas as pd

from chronaris.evaluation.application_tasks.chronaris_v2_protocol import (
    PRIMARY_METRIC_THRESHOLDS,
    SealedConfirmationManifest,
    authorize_sealed_confirmation,
    audit_v2_promotion,
)
from chronaris.evaluation.application_tasks.chronaris_v2_hyperparameter_screen import (
    ChronarisV2HyperparameterScreenConfig,
    _assert_complete_v2_passed_structure_gate,
)
from chronaris.evaluation.application_tasks.chronaris_v2_structure_diagnostics_run import (
    _final_training_checkpoint,
)
from chronaris.evaluation.application_tasks.chronaris_v2_structure_merge import (
    ChronarisV2StructureMergeConfig,
    merge_chronaris_v2_structure_screens,
)
from chronaris.evaluation.application_tasks.simulation_stress_representation_run import (
    SimulationStressRepresentationConfig,
    _require_confirmation_access,
)
from chronaris.modeling.training import chronaris_v2_structure_candidates


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_sealed_confirmation_denies_development_access() -> None:
    manifest = SealedConfirmationManifest(
        family_id="confirmation_family_2026_07_12",
        generator_protocol_sha256=_sha("protocol"),
        payload_sha256=_sha("payload"),
        sealed_before_configuration_lock=True,
    )
    with pytest.raises(PermissionError, match="unavailable"):
        manifest.assert_access_allowed(
            phase="development",
            configuration_locked=False,
        )


def test_promotion_is_fail_closed_when_one_metric_misses_threshold() -> None:
    rows = []
    for metric, (direction, threshold) in PRIMARY_METRIC_THRESHOLDS.items():
        passing = threshold + 0.01 if direction == "higher" else threshold - 0.01
        if metric == "simulation_load_rmse":
            passing = threshold + 0.01
        for seed in (17, 29, 43):
            rows.append(
                {
                    "metric_name": metric,
                    "seed": seed,
                    "value": passing,
                    "rank_first": True,
                }
            )
    gates = {
        "single_stream_no_harm": True,
        "class_recall_gap_within_0_05": True,
        "time_mechanism_within_10_percent": True,
        "missingness_slopes_top_three": True,
        "causal_future_invariance": True,
        "invalid_queries_excluded_from_pooling": True,
    }
    audit = audit_v2_promotion(rows, additional_gates=gates)
    assert not audit["promoted"]
    assert audit["paper_main_model"] == "chronaris_v1"
    assert not audit["confirmed_v1_evidence_changed"]


def test_hyperparameter_screen_cannot_start_before_structure_gate(tmp_path) -> None:
    root = tmp_path / "structure"
    root.mkdir()
    (root / "evidence_manifest.json").write_text(
        json.dumps({"status": "gates_failed"}),
        encoding="utf-8",
    )
    (root / "task_independent_ranking.csv").write_text(
        "candidate_id,gate_passed\nstructure_08_complete_v2,False\n",
        encoding="utf-8",
    )
    config = ChronarisV2HyperparameterScreenConfig(
        compact_output_root=str(tmp_path),
        architecture_gate_run_id="structure",
        architecture_gate_candidate_id="structure_08_complete_v2",
        device="cpu",
        max_candidates=1,
        max_epochs=1,
    )

    with pytest.raises(PermissionError, match="closed"):
        _assert_complete_v2_passed_structure_gate(config)


def test_hyperparameter_screen_accepts_selected_task_independent_architecture(
    tmp_path,
) -> None:
    root = tmp_path / "architecture"
    root.mkdir()
    (root / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "status": "gates_passed",
                "selected_candidate_ids": ["direct"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "candidate_id": "direct",
                "gate_passed": True,
                "task_labels_opened": False,
                "outer_test_opened": False,
                "sealed_confirmation_opened": False,
            }
        ]
    ).to_csv(root / "task_independent_ranking.csv", index=False)
    config = ChronarisV2HyperparameterScreenConfig(
        compact_output_root=str(tmp_path),
        architecture_gate_run_id="architecture",
        architecture_gate_candidate_id="direct",
        device="cpu",
        max_candidates=1,
        max_epochs=1,
    )

    _assert_complete_v2_passed_structure_gate(config)


def test_confirmation_unlock_requires_a_task_independent_locked_config(tmp_path) -> None:
    sealed = SealedConfirmationManifest(
        family_id="family",
        generator_protocol_sha256=_sha("protocol"),
        payload_sha256=_sha("payload"),
        sealed_before_configuration_lock=True,
    )
    sealed_path = tmp_path / "sealed.json"
    sealed_path.write_text(json.dumps(sealed.to_dict()), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": False,
                "selection_uses_downstream_labels": False,
                "outer_test_opened": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="locked configuration"):
        authorize_sealed_confirmation(
            sealed_manifest_path=sealed_path,
            locked_configuration_path=lock_path,
            output_path=tmp_path / "access.json",
        )


def test_sealed_representation_access_is_bound_to_locked_configuration(tmp_path) -> None:
    sealed = SealedConfirmationManifest(
        family_id="family",
        generator_protocol_sha256=_sha("protocol"),
        payload_sha256=_sha("payload"),
        sealed_before_configuration_lock=True,
    )
    sealed_path = tmp_path / "sealed.json"
    sealed_path.write_text(json.dumps(sealed.to_dict()), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": True,
                "selection_uses_downstream_labels": False,
                "outer_test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    access_path = tmp_path / "access.json"
    authorize_sealed_confirmation(
        sealed_manifest_path=sealed_path,
        locked_configuration_path=lock_path,
        output_path=access_path,
    )
    config = SimulationStressRepresentationConfig(
        sealed_manifest_path=str(sealed_path),
        confirmation_access_path=str(access_path),
        locked_configuration_path=str(lock_path),
    )

    _require_confirmation_access(config)
    _require_confirmation_access(
        SimulationStressRepresentationConfig(
            sealed_manifest_path=str(sealed_path),
            confirmation_access_path=str(access_path),
            confirmation_locked_configuration_path=str(lock_path),
        )
    )

    lock_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="does not match"):
        _require_confirmation_access(config)


def test_structure_gate_uses_final_training_state_not_public_loss_best(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    best = root / "best.pt"
    last = root / "last.pt"
    best.write_bytes(b"public-best")
    last.write_bytes(b"final-state")

    assert _final_training_checkpoint(best) == last


def test_device_partitioned_structure_rows_merge_without_checkpoint_copy(tmp_path) -> None:
    candidates = chronaris_v2_structure_candidates()
    common = {
        "status": "completed",
        "checkpoint_sha256": "a" * 64,
        "task_labels_opened": False,
        "simulation_oracle_opened": False,
        "locked_test_opened": False,
    }
    first = [
        {"candidate_id": value.candidate_id, "device_partition": "gpu", **common}
        for value in candidates[:6]
    ]
    first[0]["status"] = "immutable_reference"
    second = [
        {
            "candidate_id": candidates[0].candidate_id,
            "device_partition": "gpu",
            **common,
        }
    ] + [
        {"candidate_id": value.candidate_id, "device_partition": "cpu", **common}
        for value in candidates[4:]
    ]
    second[0]["status"] = "immutable_reference"
    for run_id, rows in (("gpu", first), ("cpu", second)):
        root = tmp_path / run_id
        root.mkdir()
        pd.DataFrame(rows).to_csv(root / "candidate_training.csv", index=False)

    output = merge_chronaris_v2_structure_screens(
        ChronarisV2StructureMergeConfig(
            run_id="combined",
            source_run_ids=("gpu", "cpu"),
            compact_output_root=str(tmp_path),
        )
    )

    merged = pd.read_csv(output / "candidate_training.csv")
    assert len(merged) == 8
    assert (
        merged.loc[
            merged["candidate_id"] == candidates[4].candidate_id,
            "device_partition",
        ].item()
        == "cpu"
    )
