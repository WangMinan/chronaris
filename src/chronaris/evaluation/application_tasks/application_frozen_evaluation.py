"""Evaluation-only reuse of already fitted application consumer bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from chronaris.evaluation.application_tasks.application_consumer_runtime import (
    _evaluate_models,
    _load_model_components,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeTargets,
)
from chronaris.representation import FusionStreamBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    write_deterministic_npz,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class ApplicationFrozenEvaluationResult:
    method_name: str
    evaluation_id: str
    metric_rows: tuple[Mapping[str, object], ...]
    workload_prediction_rows: tuple[Mapping[str, object], ...]
    unit_score_rows: tuple[Mapping[str, object], ...]
    prediction_path: str
    prediction_sha256: str
    source_model_protocol_sha256: str


def evaluate_frozen_application_consumers(
    *,
    method_name: str,
    output: FusionStreamBatch,
    targets: ApplicationConsumerSmokeTargets,
    model_root: str | Path,
    output_root: str | Path,
    fold_id: str,
    evaluation_id: str,
    seed: int,
    evaluation_role: str = "held_out",
) -> ApplicationFrozenEvaluationResult:
    """Apply an already fitted clean-condition consumer set without refitting."""

    if evaluation_role not in {"validation", "held_out"} or output.method_name != method_name:
        raise ValueError("invalid frozen consumer role or method")
    root = Path(model_root).resolve() / method_name
    manifest_path = root / "consumer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = tuple(sha256_file(Path(item["path"])) for item in manifest["model_files"].values())
    if hashes != tuple(item["sha256"] for item in manifest["model_files"].values()):
        raise ValueError("frozen consumer files changed")
    loaded, protocol_hash = _load_frozen_components(str(Path(model_root).resolve()), method_name,
        sha256_file(manifest_path), hashes)
    if set(loaded) != {"linear", "minirocket", "tcn"}:
        raise ValueError("frozen application consumer set is incomplete")
    tcn_model, duration, _training_rows = loaded["tcn"]
    metric_rows, workload_rows, unit_rows, prediction_payload = _evaluate_models(
        method_name=method_name,
        outputs={evaluation_role: output},
        targets=targets,
        fold_id=fold_id,
        linear=loaded["linear"],
        minirocket=loaded["minirocket"],
        tcn_model=tcn_model,
        duration=duration,
        seed=seed,
    )
    destination = Path(output_root) / method_name / evaluation_id
    prediction_path = destination / "predictions.npz"
    prediction_hash = write_deterministic_npz(prediction_path, prediction_payload)
    return ApplicationFrozenEvaluationResult(
        method_name=method_name,
        evaluation_id=evaluation_id,
        metric_rows=tuple(metric_rows),
        workload_prediction_rows=tuple(workload_rows),
        unit_score_rows=tuple(unit_rows),
        prediction_path=str(prediction_path),
        prediction_sha256=prediction_hash,
        source_model_protocol_sha256=protocol_hash,
    )


@lru_cache(maxsize=64)
def _load_frozen_components(model_root: str, method_name: str, manifest_sha256: str, model_hashes: tuple[str, ...]):
    root = Path(model_root) / method_name
    manifest_path = root / "consumer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_hash = str(manifest["protocol_sha256"])
    loaded = _load_model_components(
        root=root,
        manifest_path=manifest_path,
        protocol_sha256=protocol_hash,
        resume=True,
    )
    return loaded, protocol_hash
