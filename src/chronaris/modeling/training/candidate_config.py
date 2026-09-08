"""Configuration contracts for the shared candidate trainer."""
from dataclasses import dataclass
import math
from typing import Mapping

import torch
from chronaris.modeling.fusion_encoders.safe_lag_fusion import ATTENTION_KINDS


@dataclass(frozen=True, slots=True)
class CandidateScreenConfig:
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    minimum_delta: float = 0.0
    device: str = "cpu"
    deterministic: bool = True
    max_ode_step_s: float | None = None
    ode_method: str = "euler"
    semantic_event_enabled: bool = False
    learnable_semantic_queries: bool = False
    heartbeat_interval_s: float = 30.0
    physics_calibration: Mapping[str, object] | None = None
    physics_weight: float = 0.1
    max_updates: int | None = None
    effective_batch_size: int | None = None
    validation_interval: int = 100
    validation_updates: tuple[int, ...] = ()
    minimum_updates: int = 500
    checkpoint_interval: int = 25
    early_stopping: bool = True
    sampling_hierarchy: Mapping[str, tuple[str, ...]] | None = None
    retained_updates: tuple[int, ...] = ()
    data_manifest_sha256: str | None = None
    cuda_graph_recurrence: bool = False
    attention_kind: str = "legacy_cosine"
    continuous_alignment_weight: float = 0.2
    independent_pairing_enabled: bool = False
    independent_pair_weight: float = 0.0
    prediction_horizons_s: tuple[float, ...] = ()
    single_stream_fidelity_weight: float = 0.
    quality_gate_enabled: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.single_stream_fidelity_weight) or self.single_stream_fidelity_weight < 0:
            raise ValueError("single-stream fidelity weight must be finite and non-negative")
        if tuple(self.prediction_horizons_s) not in ((), (.5, 2., 5.)):
            raise ValueError("unsupported prediction horizons")
        if self.attention_kind not in ATTENTION_KINDS:
            raise ValueError("unsupported candidate attention kind")
        if not math.isfinite(self.continuous_alignment_weight) or self.continuous_alignment_weight < 0:
            raise ValueError("continuous alignment weight must be finite and non-negative")
        if not math.isfinite(self.independent_pair_weight) or self.independent_pair_weight < 0:
            raise ValueError("independent pairing weight must be finite and non-negative")
        if self.independent_pair_weight > 0 and not self.independent_pairing_enabled:
            raise ValueError("independent pairing loss requires independent history projections")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("candidate screen epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("candidate screen optimizer configuration is invalid")
        if self.physics_weight < 0:
            raise ValueError("candidate physics weight must be non-negative")
        if self.minimum_delta < 0:
            raise ValueError("candidate screen minimum delta must be non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("candidate screen device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("candidate screen requested unavailable CUDA device")
        if self.max_ode_step_s is not None and (
            not math.isfinite(self.max_ode_step_s) or self.max_ode_step_s <= 0
        ):
            raise ValueError("max_ode_step_s must be finite and positive when set")
        if self.ode_method not in {"euler", "midpoint", "rk4", "dopri5"}:
            raise ValueError("unsupported candidate Chronaris ODE method")
        if self.learnable_semantic_queries and not self.semantic_event_enabled:
            raise ValueError("learnable semantic queries require semantic_event_enabled")
        if not 0 < self.heartbeat_interval_s <= 60:
            raise ValueError("heartbeat_interval_s must be in (0,60]")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive")
        if self.effective_batch_size is not None and (
            self.max_updates is None or self.effective_batch_size < self.batch_size
            or self.effective_batch_size % self.batch_size
        ):
            raise ValueError("effective batch must be a multiple of actual batch in update mode")
        if min(self.validation_interval, self.checkpoint_interval) <= 0 or self.minimum_updates < 0:
            raise ValueError("update validation/checkpoint schedule is invalid")
        if any(update <= 0 for update in self.validation_updates):
            raise ValueError("explicit validation updates must be positive")
        if any(update <= 0 for update in self.retained_updates):
            raise ValueError("retained update positions must be positive")
        if self.data_manifest_sha256 is not None and (len(self.data_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.data_manifest_sha256)):
            raise ValueError("data manifest fingerprint must be SHA-256")
