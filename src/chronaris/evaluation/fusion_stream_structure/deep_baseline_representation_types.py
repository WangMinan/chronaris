"""Types and constants for Dingxin deep baseline E3 representation export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.evaluation.dingxin.pipelines.benchmark_data import TASK_RESPONSE

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = "docs/artifacts/runs"
DEFAULT_EXPORT_RUN_ID = "2026-07-07_deep-baseline-representation-export"
DEFAULT_FOUR_METHOD_RUN_ID = "2026-07-07_fusion-stream-structure-dingxin-four-method-validation"
DEFAULT_E_MANIFEST = "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
DEFAULT_F_MANIFEST = "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
REPRESENTATION_FAMILY = "T2_response_lovo_seed17_pooled_embedding"
FORBIDDEN_REPRESENTATION_COLUMNS = (
    "logit",
    "prediction",
    "predicted",
    "rank",
    "embedding_norm",
    "attention_entropy",
    "top_event_concentration",
    "event_mask_interference",
    "loss",
    "label",
    "y_true",
    "y_pred",
)


@dataclass(frozen=True, slots=True)
class DeepBaselineRepresentationExportConfig:
    run_id: str = DEFAULT_EXPORT_RUN_ID
    output_root: str = DEFAULT_OUTPUT_ROOT
    e_run_manifest_path: str = DEFAULT_E_MANIFEST
    f_run_manifest_path: str = DEFAULT_F_MANIFEST
    models: tuple[str, ...] = ("mult", "contiformer")
    task_name: str = TASK_RESPONSE
    task_type: str = "regression"
    split_strategy: str = "leave_one_view_out"
    seed: int = 17
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    device: str = "auto"
    require_cuda: bool = True
    tensor_cache: str = "auto"
    max_cache_gb: float = 18.0
    pin_memory: bool = True
    non_blocking_copy: bool = True
    auto_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (24576, 16384, 8192, 4096, 2048, 1024, 512, 256, 128)
    amp: str = "bf16"
    grad_scaler: bool = True
    amp_eval: bool = True
    torch_compile: str = "off"
    profile_gpu: bool = True
    eval_batch_size: int | None = None
    num_workers: int = 24
    checkpoint_policy: str = "last"
    resume: bool = True
    skip_completed: bool = True
    max_folds: int | None = None
    representation_family: str = REPRESENTATION_FAMILY
    allow_partial: bool = True
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20


@dataclass(frozen=True, slots=True)
class DeepBaselineRepresentationExportResult:
    run_id: str
    run_root: str
    status: str
    training_invoked: bool
    confirmed_metrics_changed: bool
    representation_table_path: str | None
    checkpoint_manifest_path: str | None
    report_path: str
    evidence_manifest_path: str
    completed_fold_count: int
    expected_fold_count: int
    model_embedding_status: Mapping[str, bool]
    blockers_path: str | None = None

    @property
    def all_models_complete(self) -> bool:
        return bool(self.model_embedding_status) and all(self.model_embedding_status.values())
