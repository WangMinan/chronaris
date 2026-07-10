from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronaris.representation import (
    CheckpointRegistry,
    ContractProbeEncoder,
    FoldLineage,
    ResumableOOFExporter,
    build_checkpoint_record,
    collate_observation_samples,
    load_fusion_stream_batch,
    validate_oof_coverage,
)
from chronaris.representation.contracts import RepresentationContractError
from tests.representation.test_contracts import _sample


def _checkpoint(tmp_path: Path, *, method: str = "chronaris"):
    fold = FoldLineage(
        fold_id="fold_a",
        train_sample_ids=("train",),
        validation_sample_ids=(),
        held_out_sample_ids=("test",),
    )
    path = tmp_path / f"{method}.checkpoint"
    path.write_text(f"contract probe checkpoint for {method}\n", encoding="utf-8")
    return build_checkpoint_record(
        method_name=method,
        fold=fold,
        checkpoint_path=path,
        seed=17,
    )


def test_checkpoint_registry_round_trips_and_verifies_hash(tmp_path):
    record = _checkpoint(tmp_path)
    path = tmp_path / "checkpoint_registry.json"
    registry = CheckpointRegistry(path)
    registry.register(record)

    loaded = CheckpointRegistry(path).require("chronaris", "fold_a")

    assert loaded == record
    assert loaded.fit_sample_hash == record.fit_sample_hash
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["records"][0]["label_used_for_encoder_training"] is False


def test_checkpoint_registry_detects_mutated_checkpoint(tmp_path):
    record = _checkpoint(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry.json")
    registry.register(record)
    Path(record.checkpoint_path).write_text("mutated\n", encoding="utf-8")

    with pytest.raises(RepresentationContractError, match="hash mismatch"):
        registry.require("chronaris", "fold_a")


def test_resumable_oof_export_reuses_only_valid_complete_output(tmp_path):
    record = _checkpoint(tmp_path)
    batch = collate_observation_samples([_sample("test")])
    encoder = ContractProbeEncoder(
        method_name="chronaris",
        fold_id=record.fold.fold_id,
        checkpoint_sha256=record.checkpoint_sha256,
    )
    exporter = ResumableOOFExporter(tmp_path / "exports", resume=True)

    first = exporter.export(
        encoder=encoder,
        batch=batch,
        checkpoint=record,
        export_role="held_out",
    )
    second = exporter.export(
        encoder=encoder,
        batch=batch,
        checkpoint=record,
        export_role="held_out",
    )
    Path(first.representation_path).unlink()
    rebuilt = exporter.export(
        encoder=encoder,
        batch=batch,
        checkpoint=record,
        export_role="held_out",
    )

    assert first.status == "completed"
    assert second.status == "resumed"
    assert rebuilt.status == "completed"
    loaded = load_fusion_stream_batch(rebuilt.output_root)
    assert loaded.sample_ids == ("test",)
    assert all(isinstance(value, str) for value in loaded.sample_ids)
    assert loaded.sequence_embedding.shape == (1, 96, 64)


def test_oof_export_rejects_train_sample_on_held_out_checkpoint(tmp_path):
    record = _checkpoint(tmp_path)
    encoder = ContractProbeEncoder(
        method_name="chronaris",
        fold_id=record.fold.fold_id,
        checkpoint_sha256=record.checkpoint_sha256,
    )

    with pytest.raises(RepresentationContractError, match="sample order"):
        ResumableOOFExporter(tmp_path / "exports").export(
            encoder=encoder,
            batch=collate_observation_samples([_sample("train")]),
            checkpoint=record,
            export_role="held_out",
        )


def test_fusion_manifest_rejects_prediction_or_diagnostic_injection(tmp_path):
    record = _checkpoint(tmp_path)
    encoder = ContractProbeEncoder(
        method_name="chronaris",
        fold_id=record.fold.fold_id,
        checkpoint_sha256=record.checkpoint_sha256,
    )
    result = ResumableOOFExporter(tmp_path / "exports").export(
        encoder=encoder,
        batch=collate_observation_samples([_sample("test")]),
        checkpoint=record,
        export_role="held_out",
    )
    manifest_path = Path(result.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["logits"] = [0.5]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RepresentationContractError, match="forbidden fields"):
        load_fusion_stream_batch(result.output_root)


def test_oof_coverage_rejects_duplicates_or_omissions(tmp_path):
    first = _checkpoint(tmp_path, method="chronaris")
    batch = collate_observation_samples([_sample("test")])
    result = ResumableOOFExporter(tmp_path / "exports").export(
        encoder=ContractProbeEncoder(
            method_name="chronaris",
            fold_id=first.fold.fold_id,
            checkpoint_sha256=first.checkpoint_sha256,
        ),
        batch=batch,
        checkpoint=first,
        export_role="held_out",
    )

    assert len(
        validate_oof_coverage(
            [result],
            method_name="chronaris",
            expected_sample_ids=("test",),
        )
    ) == 64
    with pytest.raises(RepresentationContractError, match="duplicated"):
        validate_oof_coverage(
            [result, result],
            method_name="chronaris",
            expected_sample_ids=("test",),
        )
