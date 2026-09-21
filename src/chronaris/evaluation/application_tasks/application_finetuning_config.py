"""Shared application fine-tuning configuration and result contracts."""
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class EndToEndFineTuningConfig:
    learning_rate: float = 1e-4
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"
    max_updates: int | None = None
    effective_batch_size: int | None = None
    head_warmup_updates: int = 50
    head_learning_rate: float = 3e-4
    validation_interval: int = 50
    minimum_updates: int = 200
    checkpoint_interval: int = 25
    early_stopping: bool = True
    self_supervised_weight: float = .2
    sampling_hierarchy: Mapping[str, tuple[str, ...]] | None = None
    retained_updates: tuple[int, ...] = ()
    record_gradient_groups: bool = False
    data_manifest_sha256: str | None = None
    cache_head_encodings: bool = True
    checkpoint_selection: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.gradient_clip_norm) <= 0:
            raise ValueError("fine-tuning learning rate and gradient clip must be positive")
        if min(self.max_epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("fine-tuning epochs, patience, and batch size must be positive")
        if self.weight_decay < 0 or self.device not in {"cpu", "cuda"}:
            raise ValueError("fine-tuning optimizer or device configuration is invalid")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("fine-tuning requested unavailable CUDA device")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("fine-tuning max_updates must be positive")
        if self.effective_batch_size is not None and (self.max_updates is None
            or self.effective_batch_size < self.batch_size or self.effective_batch_size % self.batch_size):
            raise ValueError("fine-tuning effective batch must be a multiple of actual batch")
        if min(self.validation_interval, self.checkpoint_interval, self.head_learning_rate) <= 0:
            raise ValueError("invalid task-guided update schedule")
        if min(self.head_warmup_updates, self.minimum_updates, self.self_supervised_weight) < 0:
            raise ValueError("task-guided budgets/weights cannot be negative")
        if any(update <= 0 for update in self.retained_updates):
            raise ValueError("retained task-guided updates must be positive")


@dataclass(frozen=True, slots=True)
class EndToEndFineTuningResult:
    method_name: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    best_epoch: int
    completed_epochs: int
    stopped_early: bool
    training_elapsed_s: float
    encoder_update_mode: str
    training_device_history: tuple[str, ...]
    epoch_rows: tuple[Mapping[str, object], ...]
    optimizer_updates: int = 0
    head_warmup_updates: int = 0
    joint_updates: int = 0
    best_update: int = 0
