"""Internal torch helpers for Stage E alignment modules."""

from __future__ import annotations

import torch.nn as nn


def build_activation_module(name: str) -> nn.Module:
    """Return a torch activation module by name."""

    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name!r}.")
