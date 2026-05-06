"""Shared torch runtime helpers for Chronaris pipelines."""

from __future__ import annotations

import torch

TORCH_DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_torch_device_name(requested: str | None = "auto") -> str:
    """Resolve `auto/cpu/cuda` into one concrete torch device string."""

    normalized = str(requested or "auto").strip().lower()
    if normalized not in TORCH_DEVICE_CHOICES:
        raise ValueError(
            f"unsupported torch device '{requested}'; expected one of {TORCH_DEVICE_CHOICES}"
        )
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested device='cuda' but torch.cuda.is_available() is False.")
    return normalized


def seed_torch(seed: int, *, device: str | None = None) -> None:
    """Seed torch for CPU and CUDA paths."""

    torch.manual_seed(seed)
    if resolve_torch_device_name(device or "auto") == "cuda":
        torch.cuda.manual_seed_all(seed)
