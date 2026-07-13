"""Fail-closed protocol locks for the one-shot Chronaris core-task recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


CORE_RECOVERY_FORMAT = "chronaris.core_task_recovery_protocol.v1"
CORE_RECOVERY_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)
TRAINABLE_METHODS = tuple(
    value for value in CORE_RECOVERY_METHODS if value != "naive_time_sync"
)
DEFAULT_STRIDES_S = (5.0, 2.5, 1.0)
DEFAULT_BACKBONE_LR_RATIOS = (0.05, 0.1)
DEFAULT_HIGH_RESPONSE_WEIGHTS = (0.5, 1.0)
MAX_CANDIDATES_PER_TRAINABLE_METHOD = 12


@dataclass(frozen=True, slots=True)
class CoreRecoveryCandidate:
    candidate_id: str
    method_name: str
    train_stride_s: float
    backbone_lr_ratio: float
    high_response_weight: float

    def __post_init__(self) -> None:
        if self.method_name not in TRAINABLE_METHODS:
            raise ValueError("candidate method must be one of the five trainable methods")
        if self.train_stride_s not in DEFAULT_STRIDES_S:
            raise ValueError("candidate stride is outside the locked grid")
        if self.backbone_lr_ratio not in DEFAULT_BACKBONE_LR_RATIOS:
            raise ValueError("candidate backbone learning-rate ratio is outside the locked grid")
        if self.high_response_weight not in DEFAULT_HIGH_RESPONSE_WEIGHTS:
            raise ValueError("candidate high-response weight is outside the locked grid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoreRecoveryProtocol:
    source_commit: str
    branch_name: str
    data_manifest_sha256: str
    outer_split_manifest_sha256: str
    task_definition_sha256: str
    methods: tuple[str, ...] = CORE_RECOVERY_METHODS
    candidate_budget_per_trainable_method: int = MAX_CANDIDATES_PER_TRAINABLE_METHOD
    development_seed: int = 17
    confirmation_seeds: tuple[int, ...] = (17, 29, 43)
    input_history_s: float = 30.0
    target_horizon_s: float = 5.0
    outer_test_opened: bool = False

    def __post_init__(self) -> None:
        if len(self.source_commit) != 40:
            raise ValueError("source_commit must be a full Git commit hash")
        if not self.branch_name.startswith("codex/"):
            raise ValueError("core-recovery branch must use the codex/ prefix")
        if self.methods != CORE_RECOVERY_METHODS:
            raise ValueError("core-recovery method panel is immutable")
        if self.candidate_budget_per_trainable_method != 12:
            raise ValueError("core-recovery candidate budget must stay at 12")
        if self.outer_test_opened:
            raise ValueError("a newly created protocol must keep outer-test closed")
        for value in (
            self.data_manifest_sha256,
            self.outer_split_manifest_sha256,
            self.task_definition_sha256,
        ):
            if len(value) != 64:
                raise ValueError("protocol lineage values must be SHA-256 digests")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_locked_candidate_registry() -> tuple[CoreRecoveryCandidate, ...]:
    rows = []
    for method_name in TRAINABLE_METHODS:
        index = 0
        for stride_s in DEFAULT_STRIDES_S:
            for lr_ratio in DEFAULT_BACKBONE_LR_RATIOS:
                for response_weight in DEFAULT_HIGH_RESPONSE_WEIGHTS:
                    index += 1
                    rows.append(
                        CoreRecoveryCandidate(
                            candidate_id=f"{method_name}-c{index:02d}",
                            method_name=method_name,
                            train_stride_s=stride_s,
                            backbone_lr_ratio=lr_ratio,
                            high_response_weight=response_weight,
                        )
                    )
    validate_candidate_registry(rows)
    return tuple(rows)


def validate_candidate_registry(
    candidates: Sequence[CoreRecoveryCandidate],
) -> None:
    identifiers = [row.candidate_id for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate registry contains duplicate identifiers")
    for method_name in TRAINABLE_METHODS:
        method_rows = [row for row in candidates if row.method_name == method_name]
        if len(method_rows) != MAX_CANDIDATES_PER_TRAINABLE_METHOD:
            raise ValueError(
                f"{method_name} must have exactly {MAX_CANDIDATES_PER_TRAINABLE_METHOD} candidates"
            )
        grid = {
            (row.train_stride_s, row.backbone_lr_ratio, row.high_response_weight)
            for row in method_rows
        }
        if len(grid) != MAX_CANDIDATES_PER_TRAINABLE_METHOD:
            raise ValueError(f"{method_name} candidate grid is incomplete or duplicated")
    unexpected = sorted(set(row.method_name for row in candidates) - set(TRAINABLE_METHODS))
    if unexpected:
        raise ValueError(f"candidate registry contains unsupported methods: {unexpected}")


class OuterTestAccessGuard:
    """Authorize one outer-test opening only after a unique model lock exists."""

    def __init__(self, *, protocol_sha256: str) -> None:
        if len(protocol_sha256) != 64:
            raise ValueError("protocol_sha256 must be a SHA-256 digest")
        self.protocol_sha256 = protocol_sha256
        self._authorization: dict[str, object] | None = None

    def authorize(
        self,
        *,
        locked_configuration_sha256: str,
        development_gate_passed: bool,
    ) -> Mapping[str, object]:
        if self._authorization is not None:
            raise PermissionError("outer-test access was already authorized once")
        if not development_gate_passed:
            raise PermissionError("outer-test remains closed because the development gate failed")
        if len(locked_configuration_sha256) != 64:
            raise ValueError("locked configuration must have a SHA-256 digest")
        self._authorization = {
            "format": "chronaris.outer_test_access_authorization.v1",
            "protocol_sha256": self.protocol_sha256,
            "locked_configuration_sha256": locked_configuration_sha256,
            "authorized_open_count": 1,
        }
        return dict(self._authorization)

    def require_authorized(self) -> None:
        if self._authorization is None:
            raise PermissionError("outer-test access is closed")


def write_protocol_locks(
    output_root: str | Path,
    *,
    protocol: CoreRecoveryProtocol,
    source_files: Mapping[str, str | Path],
) -> Mapping[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256_file(path) for name, path in source_files.items()}
    protocol_payload = {
        "format": CORE_RECOVERY_FORMAT,
        **protocol.to_dict(),
        "source_file_sha256": source_hashes,
    }
    protocol_payload["protocol_sha256"] = stable_mapping_sha256(protocol_payload)
    registry_payload = {
        "format": "chronaris.core_task_recovery_candidates.v1",
        "failed_candidate_consumes_budget": True,
        "infrastructure_retry_limit_per_candidate": 1,
        "candidates": [row.to_dict() for row in build_locked_candidate_registry()],
    }
    registry_payload["registry_sha256"] = stable_mapping_sha256(registry_payload)
    protocol_path = root / "protocol_lock.json"
    registry_path = root / "candidate_registry.json"
    outer_path = root / "outer_test_access_lock.json"
    _write_json(protocol_path, protocol_payload)
    _write_json(registry_path, registry_payload)
    _write_json(
        outer_path,
        {
            "format": "chronaris.outer_test_access_lock.v1",
            "protocol_sha256": protocol_payload["protocol_sha256"],
            "authorized_open_count": 0,
            "outer_test_opened": False,
        },
    )
    return {
        "protocol_lock": protocol_path,
        "candidate_registry": registry_path,
        "outer_test_access_lock": outer_path,
    }


def stable_mapping_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
