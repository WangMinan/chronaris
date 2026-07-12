"""Sealed-confirmation and final-promotion guards for Chronaris v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


PRIMARY_METRIC_THRESHOLDS = {
    "dingxin_maneuver_macro_f1": ("higher", 0.9409316),
    "dingxin_high_response_auprc": ("higher", 0.8838787),
    "dingxin_response_rmse": ("lower", 0.2911710),
    "simulation_load_macro_f1": ("higher", 0.5191073),
    "simulation_load_rmse": ("lower", 0.1934353),
    "simulation_segmentation_macro_f1": ("higher", 0.4687076),
}


@dataclass(frozen=True, slots=True)
class SealedConfirmationManifest:
    family_id: str
    generator_protocol_sha256: str
    payload_sha256: str
    sealed_before_configuration_lock: bool
    method_count: int = 6
    unlocked: bool = False

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("confirmation family_id is required")
        if self.method_count != 6:
            raise ValueError("confirmation family must cover all six methods")
        for value in (self.generator_protocol_sha256, self.payload_sha256):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("confirmation hashes must be lowercase sha256")
        if not self.sealed_before_configuration_lock:
            raise ValueError("confirmation family was not sealed before model lock")

    def assert_access_allowed(self, *, phase: str, configuration_locked: bool) -> None:
        if phase != "locked_confirmation" or not configuration_locked or not self.unlocked:
            raise PermissionError(
                "sealed simulation confirmation is unavailable before locked confirmation"
            )

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


def write_sealed_confirmation_manifest(
    path: str | Path,
    *,
    family_id: str,
    generator_protocol: Mapping[str, object],
    payload_paths: Sequence[str | Path],
) -> SealedConfirmationManifest:
    """Write a still-locked manifest without reading model-selection evidence."""

    protocol_bytes = json.dumps(
        generator_protocol,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_hash = hashlib.sha256()
    for payload_path in sorted(Path(value) for value in payload_paths):
        payload_hash.update(payload_path.name.encode("utf-8"))
        with payload_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                payload_hash.update(chunk)
    manifest = SealedConfirmationManifest(
        family_id=family_id,
        generator_protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        payload_sha256=payload_hash.hexdigest(),
        sealed_before_configuration_lock=True,
        unlocked=False,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_sealed_confirmation_manifest(
    path: str | Path,
) -> SealedConfirmationManifest:
    return SealedConfirmationManifest(
        **json.loads(Path(path).read_text(encoding="utf-8"))
    )


def audit_v2_promotion(
    seed_metrics: Sequence[Mapping[str, object]],
    *,
    additional_gates: Mapping[str, bool],
) -> Mapping[str, object]:
    """Audit promotion without mutating v1 evidence or feeding results to selection."""

    required_seeds = {17, 29, 43}
    seen_seeds = {int(row["seed"]) for row in seed_metrics}
    if seen_seeds != required_seeds:
        raise ValueError("promotion audit requires exactly seeds 17, 29, and 43")
    metric_names = {str(row["metric_name"]) for row in seed_metrics}
    if metric_names != set(PRIMARY_METRIC_THRESHOLDS):
        raise ValueError("promotion audit requires exactly the six primary metrics")
    by_metric: dict[str, list[Mapping[str, object]]] = {
        name: [] for name in PRIMARY_METRIC_THRESHOLDS
    }
    for row in seed_metrics:
        by_metric[str(row["metric_name"])].append(row)
    metric_audits = []
    for name, (direction, threshold) in PRIMARY_METRIC_THRESHOLDS.items():
        rows = by_metric[name]
        if {int(row["seed"]) for row in rows} != required_seeds:
            raise ValueError(f"metric {name} does not contain one row per required seed")
        values = [float(row["value"]) for row in rows]
        mean = sum(values) / len(values)
        mean_passed = mean > threshold if direction == "higher" else mean < threshold
        first_count = sum(bool(row["rank_first"]) for row in rows)
        metric_audits.append(
            {
                "metric_name": name,
                "direction": direction,
                "threshold": threshold,
                "mean": mean,
                "mean_rank_first_threshold_passed": mean_passed,
                "rank_first_seed_count": first_count,
                "seed_gate_passed": first_count >= 2,
                "passed": mean_passed and first_count >= 2,
            }
        )
    required_additional = {
        "single_stream_no_harm",
        "class_recall_gap_within_0_05",
        "time_mechanism_within_10_percent",
        "missingness_slopes_top_three",
        "causal_future_invariance",
        "invalid_queries_excluded_from_pooling",
    }
    if set(additional_gates) != required_additional:
        raise ValueError("promotion audit additional gates are incomplete or unexpected")
    promoted = all(row["passed"] for row in metric_audits) and all(
        additional_gates.values()
    )
    return {
        "format": "chronaris.v2_promotion_audit.v1",
        "promoted": promoted,
        "paper_main_model": "chronaris_v2" if promoted else "chronaris_v1",
        "metric_audits": metric_audits,
        "additional_gates": dict(additional_gates),
        "confirmed_v1_evidence_changed": False,
        "results_may_return_to_same_development_round": False,
    }
