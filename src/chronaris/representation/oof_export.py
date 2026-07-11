"""Strict fusion representation serialization and resumable OOF export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    FusionStreamBatch,
    FusionStreamEncoder,
    RepresentationContractError,
    validate_representation_mapping_fields,
)
from chronaris.representation.lineage import CheckpointRecord, verify_checkpoint_record
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    sha256_file,
    write_deterministic_npz,
)


FUSION_ARCHIVE_KEYS = frozenset(
    {
        "sample_ids",
        "timestamps_s",
        "sequence_embedding",
        "valid_mask",
        "pooled_embedding",
        "source_sample_hashes",
    }
)
FUSION_MANIFEST_KEYS = frozenset(
    {
        "format",
        "method_name",
        "fold_id",
        "checkpoint_sha256",
        "export_role",
        "sample_ids",
        "source_sample_hashes",
        "representation_archive",
        "representation_sha256",
        "label_used_for_encoder_training",
        "output_dim",
        "query_point_count",
    }
)


@dataclass(frozen=True, slots=True)
class OOFExportResult:
    method_name: str
    fold_id: str
    export_role: str
    status: str
    output_root: str
    representation_path: str
    manifest_path: str
    representation_sha256: str
    sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "fold_id": self.fold_id,
            "export_role": self.export_role,
            "status": self.status,
            "output_root": self.output_root,
            "representation_path": self.representation_path,
            "manifest_path": self.manifest_path,
            "representation_sha256": self.representation_sha256,
            "sample_ids": list(self.sample_ids),
        }


class ResumableOOFExporter:
    """Export one method/fold/role and verify completed artifacts before reuse."""

    def __init__(self, output_root: str | Path, *, resume: bool = True) -> None:
        self.output_root = Path(output_root)
        self.resume = bool(resume)

    def export(
        self,
        *,
        encoder: FusionStreamEncoder,
        batch: DualStreamObservationBatch,
        checkpoint: CheckpointRecord,
        export_role: str,
    ) -> OOFExportResult:
        verify_checkpoint_record(checkpoint)
        if encoder.method_name != checkpoint.method_name:
            raise RepresentationContractError(
                "encoder method does not match checkpoint lineage"
            )
        allowed_ids = checkpoint.fold.sample_ids_for_role(export_role)
        if tuple(batch.sample_ids) != tuple(allowed_ids):
            raise RepresentationContractError(
                f"{export_role} batch sample order does not match frozen fold lineage"
            )
        root = (
            self.output_root
            / checkpoint.method_name
            / checkpoint.fold.fold_id
            / export_role
        )
        representation_path = root / "fusion_stream.npz"
        manifest_path = root / "representation_manifest.json"
        if self.resume and representation_path.exists() and manifest_path.exists():
            try:
                existing = load_fusion_stream_batch(root)
            except (OSError, ValueError, RepresentationContractError):
                existing = None
            if existing is not None and (
                existing.method_name == checkpoint.method_name
                and existing.fold_id == checkpoint.fold.fold_id
                and existing.checkpoint_sha256 == checkpoint.checkpoint_sha256
                and existing.sample_ids == batch.sample_ids
            ):
                return _result_from_paths(
                    batch=existing,
                    export_role=export_role,
                    status="resumed",
                    root=root,
                    representation_path=representation_path,
                    manifest_path=manifest_path,
                )
        output = encoder(batch)
        if output.method_name != checkpoint.method_name:
            raise RepresentationContractError("encoder output method lineage mismatch")
        if output.fold_id != checkpoint.fold.fold_id:
            raise RepresentationContractError("encoder output fold lineage mismatch")
        if output.checkpoint_sha256 != checkpoint.checkpoint_sha256:
            raise RepresentationContractError("encoder output checkpoint lineage mismatch")
        if output.sample_ids != batch.sample_ids:
            raise RepresentationContractError("encoder changed sample order")
        write_fusion_stream_batch(output, root=root, export_role=export_role)
        return _result_from_paths(
            batch=output,
            export_role=export_role,
            status="completed",
            root=root,
            representation_path=representation_path,
            manifest_path=manifest_path,
        )

    def export_from_batch_provider(
        self,
        *,
        encoder: FusionStreamEncoder,
        batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch],
        checkpoint: CheckpointRecord,
        export_role: str,
        batch_size: int = 2,
    ) -> OOFExportResult:
        """Export a frozen role through bounded raw-input batches."""
        verify_checkpoint_record(checkpoint)
        if batch_size <= 0:
            raise ValueError("OOF provider batch size must be positive")
        if encoder.method_name != checkpoint.method_name:
            raise RepresentationContractError(
                "encoder method does not match checkpoint lineage"
            )
        allowed_ids = tuple(checkpoint.fold.sample_ids_for_role(export_role))
        root = (
            self.output_root
            / checkpoint.method_name
            / checkpoint.fold.fold_id
            / export_role
        )
        representation_path = root / "fusion_stream.npz"
        manifest_path = root / "representation_manifest.json"
        if self.resume and representation_path.exists() and manifest_path.exists():
            try:
                existing = load_fusion_stream_batch(root)
            except (OSError, ValueError, RepresentationContractError):
                existing = None
            if existing is not None and (
                existing.method_name == checkpoint.method_name
                and existing.fold_id == checkpoint.fold.fold_id
                and existing.checkpoint_sha256 == checkpoint.checkpoint_sha256
                and existing.sample_ids == allowed_ids
            ):
                return _result_from_paths(
                    batch=existing,
                    export_role=export_role,
                    status="resumed",
                    root=root,
                    representation_path=representation_path,
                    manifest_path=manifest_path,
                )
        outputs = []
        for offset in range(0, len(allowed_ids), batch_size):
            sample_ids = allowed_ids[offset : offset + batch_size]
            raw = batch_provider(sample_ids)
            if tuple(raw.sample_ids) != sample_ids:
                raise RepresentationContractError(
                    "OOF batch provider changed sample order"
                )
            output = encoder(raw)
            _validate_output_lineage(output, checkpoint, sample_ids)
            outputs.append(output)
        combined = _concatenate_fusion_batches(outputs)
        write_fusion_stream_batch(combined, root=root, export_role=export_role)
        return _result_from_paths(
            batch=combined,
            export_role=export_role,
            status="completed",
            root=root,
            representation_path=representation_path,
            manifest_path=manifest_path,
        )


def write_fusion_stream_batch(
    batch: FusionStreamBatch,
    *,
    root: str | Path,
    export_role: str,
) -> tuple[Path, Path]:
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    representation_path = output_root / "fusion_stream.npz"
    representation_hash = write_deterministic_npz(
        representation_path,
        {
            "sample_ids": np.asarray(batch.sample_ids),
            "timestamps_s": _numpy(batch.timestamps_s, np.float64),
            "sequence_embedding": _numpy(batch.sequence_embedding, np.float32),
            "valid_mask": _numpy(batch.valid_mask, bool),
            "pooled_embedding": _numpy(batch.pooled_embedding, np.float32),
            "source_sample_hashes": np.asarray(batch.source_sample_hashes),
        },
    )
    manifest = {
        "format": "chronaris.fusion_stream.v1",
        "method_name": batch.method_name,
        "fold_id": batch.fold_id,
        "checkpoint_sha256": batch.checkpoint_sha256,
        "export_role": export_role,
        "sample_ids": list(batch.sample_ids),
        "source_sample_hashes": list(batch.source_sample_hashes),
        "representation_archive": representation_path.name,
        "representation_sha256": representation_hash,
        "label_used_for_encoder_training": False,
        "output_dim": int(batch.sequence_embedding.shape[-1]),
        "query_point_count": int(batch.sequence_embedding.shape[1]),
    }
    validate_representation_mapping_fields(manifest)
    manifest_path = output_root / "representation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return representation_path, manifest_path


def load_fusion_stream_batch(root: str | Path) -> FusionStreamBatch:
    output_root = Path(root)
    manifest_path = output_root / "representation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_representation_mapping_fields(manifest)
    manifest_keys = frozenset(manifest)
    if manifest_keys != FUSION_MANIFEST_KEYS:
        extra = sorted(manifest_keys - FUSION_MANIFEST_KEYS)
        missing = sorted(FUSION_MANIFEST_KEYS - manifest_keys)
        raise RepresentationContractError(
            f"fusion manifest schema mismatch; extra={extra}, missing={missing}"
        )
    if manifest["format"] != "chronaris.fusion_stream.v1":
        raise RepresentationContractError("unsupported fusion stream format")
    path = output_root / str(manifest["representation_archive"])
    if sha256_file(path) != str(manifest["representation_sha256"]):
        raise RepresentationContractError("fusion representation archive hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        if keys != FUSION_ARCHIVE_KEYS:
            extra = sorted(keys - FUSION_ARCHIVE_KEYS)
            missing = sorted(FUSION_ARCHIVE_KEYS - keys)
            raise RepresentationContractError(
                f"fusion archive schema mismatch; extra={extra}, missing={missing}"
            )
        payload = {key: archive[key] for key in archive.files}
    archive_sample_ids = tuple(str(value) for value in payload["sample_ids"])
    archive_source_hashes = tuple(
        str(value) for value in payload["source_sample_hashes"]
    )
    if archive_sample_ids != tuple(manifest["sample_ids"]):
        raise RepresentationContractError("fusion sample IDs differ between archive and manifest")
    if archive_source_hashes != tuple(manifest["source_sample_hashes"]):
        raise RepresentationContractError("fusion source hashes differ from manifest")
    return FusionStreamBatch(
        sample_ids=archive_sample_ids,
        timestamps_s=torch.from_numpy(payload["timestamps_s"].astype(np.float64)),
        sequence_embedding=torch.from_numpy(
            payload["sequence_embedding"].astype(np.float32)
        ),
        valid_mask=torch.from_numpy(payload["valid_mask"].astype(bool)),
        pooled_embedding=torch.from_numpy(payload["pooled_embedding"].astype(np.float32)),
        method_name=str(manifest["method_name"]),
        fold_id=str(manifest["fold_id"]),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        source_sample_hashes=archive_source_hashes,
    )


def validate_oof_coverage(
    results: Sequence[OOFExportResult],
    *,
    method_name: str,
    expected_sample_ids: Sequence[str],
) -> str:
    selected = [
        result
        for result in results
        if result.method_name == method_name and result.export_role == "held_out"
    ]
    observed = [sample for result in selected for sample in result.sample_ids]
    if len(observed) != len(set(observed)):
        raise RepresentationContractError(
            f"OOF held-out samples are duplicated for {method_name}"
        )
    expected = set(expected_sample_ids)
    actual = set(observed)
    if actual != expected:
        raise RepresentationContractError(
            f"OOF coverage mismatch for {method_name}; "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    payload = json.dumps(sorted(observed), ensure_ascii=False).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _validate_output_lineage(
    output: FusionStreamBatch,
    checkpoint: CheckpointRecord,
    sample_ids: Sequence[str],
) -> None:
    if output.method_name != checkpoint.method_name:
        raise RepresentationContractError("encoder output method lineage mismatch")
    if output.fold_id != checkpoint.fold.fold_id:
        raise RepresentationContractError("encoder output fold lineage mismatch")
    if output.checkpoint_sha256 != checkpoint.checkpoint_sha256:
        raise RepresentationContractError("encoder output checkpoint lineage mismatch")
    if output.sample_ids != tuple(sample_ids):
        raise RepresentationContractError("encoder changed sample order")


def _concatenate_fusion_batches(
    batches: Sequence[FusionStreamBatch],
) -> FusionStreamBatch:
    if not batches:
        raise RepresentationContractError("OOF provider produced no batches")
    first = batches[0]
    for batch in batches[1:]:
        if (
            batch.method_name != first.method_name
            or batch.fold_id != first.fold_id
            or batch.checkpoint_sha256 != first.checkpoint_sha256
            or batch.sequence_embedding.shape[1:] != first.sequence_embedding.shape[1:]
        ):
            raise RepresentationContractError("OOF provider batch lineage changed")
    return FusionStreamBatch(
        sample_ids=tuple(
            sample_id for batch in batches for sample_id in batch.sample_ids
        ),
        timestamps_s=torch.cat(
            tuple(batch.timestamps_s.detach().cpu() for batch in batches), dim=0
        ),
        sequence_embedding=torch.cat(
            tuple(batch.sequence_embedding.detach().cpu() for batch in batches), dim=0
        ),
        valid_mask=torch.cat(
            tuple(batch.valid_mask.detach().cpu() for batch in batches), dim=0
        ),
        pooled_embedding=torch.cat(
            tuple(batch.pooled_embedding.detach().cpu() for batch in batches), dim=0
        ),
        method_name=first.method_name,
        fold_id=first.fold_id,
        checkpoint_sha256=first.checkpoint_sha256,
        source_sample_hashes=tuple(
            value for batch in batches for value in batch.source_sample_hashes
        ),
    )


def _result_from_paths(
    *,
    batch: FusionStreamBatch,
    export_role: str,
    status: str,
    root: Path,
    representation_path: Path,
    manifest_path: Path,
) -> OOFExportResult:
    return OOFExportResult(
        method_name=batch.method_name,
        fold_id=batch.fold_id,
        export_role=export_role,
        status=status,
        output_root=str(root),
        representation_path=str(representation_path),
        manifest_path=str(manifest_path),
        representation_sha256=sha256_file(representation_path),
        sample_ids=batch.sample_ids,
    )


def _numpy(tensor: torch.Tensor, dtype) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)
