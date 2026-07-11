"""Fold, transform and checkpoint lineage with explicit held-out isolation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.representation.contracts import RepresentationContractError
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


@dataclass(frozen=True, slots=True)
class FoldLineage:
    fold_id: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    held_out_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fold_id:
            raise RepresentationContractError("fold_id is required")
        train = set(self.train_sample_ids)
        validation = set(self.validation_sample_ids)
        held_out = set(self.held_out_sample_ids)
        if not train or not held_out:
            raise RepresentationContractError(
                "fold train and held-out sample lists must be non-empty"
            )
        if len(train) != len(self.train_sample_ids):
            raise RepresentationContractError("fold train sample IDs contain duplicates")
        if len(validation) != len(self.validation_sample_ids):
            raise RepresentationContractError(
                "fold validation sample IDs contain duplicates"
            )
        if len(held_out) != len(self.held_out_sample_ids):
            raise RepresentationContractError(
                "fold held-out sample IDs contain duplicates"
            )
        intersections = {
            "train_validation": train & validation,
            "train_held_out": train & held_out,
            "validation_held_out": validation & held_out,
        }
        overlap = {key: sorted(value) for key, value in intersections.items() if value}
        if overlap:
            raise RepresentationContractError(f"fold sample sets overlap: {overlap}")

    def sample_ids_for_role(self, export_role: str) -> tuple[str, ...]:
        mapping = {
            "train": self.train_sample_ids,
            "validation": self.validation_sample_ids,
            "held_out": self.held_out_sample_ids,
        }
        if export_role not in mapping:
            raise RepresentationContractError(f"unsupported export role: {export_role}")
        return mapping[export_role]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
            "held_out_sample_ids": list(self.held_out_sample_ids),
            "train_sample_hash": stable_sample_hash(self.train_sample_ids),
            "validation_sample_hash": stable_sample_hash(self.validation_sample_ids),
            "held_out_sample_hash": stable_sample_hash(self.held_out_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    method_name: str
    fold: FoldLineage
    checkpoint_path: str
    checkpoint_sha256: str
    fit_sample_hash: str
    seed: int
    status: str = "locked"
    label_used_for_encoder_training: bool = False

    def __post_init__(self) -> None:
        if not self.method_name or self.status not in {"locked", "complete"}:
            raise RepresentationContractError("checkpoint method/status is invalid")
        if len(self.checkpoint_sha256) != 64 or len(self.fit_sample_hash) != 64:
            raise RepresentationContractError("checkpoint and fit hashes must be SHA-256")
        expected_fit_hash = stable_sample_hash(self.fold.train_sample_ids)
        if self.fit_sample_hash != expected_fit_hash:
            raise RepresentationContractError(
                "checkpoint fit hash does not match fold train samples"
            )
        if self.label_used_for_encoder_training:
            raise RepresentationContractError(
                "frozen representation family cannot use downstream labels"
            )

    @property
    def registry_key(self) -> str:
        return f"{self.method_name}::{self.fold.fold_id}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fold"] = self.fold.to_dict()
        return payload


class CheckpointRegistry:
    """Atomic JSON registry used by resumable fold export."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, CheckpointRecord] = {}
        if self.path.exists():
            self._load()

    def register(
        self,
        record: CheckpointRecord,
        *,
        replace_existing: bool = False,
    ) -> None:
        existing = self.records.get(record.registry_key)
        if existing is not None and existing != record and not replace_existing:
            raise RepresentationContractError(
                f"checkpoint registry conflict for {record.registry_key}"
            )
        verify_checkpoint_record(record)
        self.records[record.registry_key] = record
        self._write()

    def require(self, method_name: str, fold_id: str) -> CheckpointRecord:
        key = f"{method_name}::{fold_id}"
        if key not in self.records:
            raise RepresentationContractError(f"checkpoint registry has no record for {key}")
        record = self.records[key]
        verify_checkpoint_record(record)
        return record

    def to_dict(self) -> Mapping[str, object]:
        return {
            "format": "chronaris.checkpoint_registry.v1",
            "record_count": len(self.records),
            "records": [
                self.records[key].to_dict() for key in sorted(self.records)
            ],
        }

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("format") != "chronaris.checkpoint_registry.v1":
            raise RepresentationContractError("unsupported checkpoint registry format")
        records: dict[str, CheckpointRecord] = {}
        for item in payload.get("records", []):
            fold_payload = dict(item["fold"])
            fold = FoldLineage(
                fold_id=str(fold_payload["fold_id"]),
                train_sample_ids=tuple(fold_payload["train_sample_ids"]),
                validation_sample_ids=tuple(fold_payload["validation_sample_ids"]),
                held_out_sample_ids=tuple(fold_payload["held_out_sample_ids"]),
            )
            record = CheckpointRecord(
                method_name=str(item["method_name"]),
                fold=fold,
                checkpoint_path=str(item["checkpoint_path"]),
                checkpoint_sha256=str(item["checkpoint_sha256"]),
                fit_sample_hash=str(item["fit_sample_hash"]),
                seed=int(item["seed"]),
                status=str(item["status"]),
                label_used_for_encoder_training=bool(
                    item["label_used_for_encoder_training"]
                ),
            )
            records[record.registry_key] = record
        self.records = records


def build_checkpoint_record(
    *,
    method_name: str,
    fold: FoldLineage,
    checkpoint_path: str | Path,
    seed: int,
) -> CheckpointRecord:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise RepresentationContractError(f"checkpoint does not exist: {path}")
    return CheckpointRecord(
        method_name=method_name,
        fold=fold,
        checkpoint_path=str(path),
        checkpoint_sha256=sha256_file(path),
        fit_sample_hash=stable_sample_hash(fold.train_sample_ids),
        seed=int(seed),
    )


def verify_checkpoint_record(record: CheckpointRecord) -> None:
    path = Path(record.checkpoint_path)
    if not path.is_file():
        raise RepresentationContractError(
            f"registered checkpoint is missing: {record.checkpoint_path}"
        )
    actual = sha256_file(path)
    if actual != record.checkpoint_sha256:
        raise RepresentationContractError(
            f"registered checkpoint hash mismatch for {record.registry_key}"
        )


def verify_fit_sample_isolation(
    *,
    fit_sample_ids: Sequence[str],
    held_out_sample_ids: Sequence[str],
) -> str:
    overlap = sorted(set(fit_sample_ids) & set(held_out_sample_ids))
    if overlap:
        raise RepresentationContractError(
            f"held-out samples appear in transform fit lineage: {overlap[:5]}"
        )
    if not fit_sample_ids:
        raise RepresentationContractError("transform fit lineage is empty")
    return stable_sample_hash(fit_sample_ids)
