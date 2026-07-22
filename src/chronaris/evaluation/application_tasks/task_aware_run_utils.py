"""Shared persistence helpers for task-aware Dingxin screening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import torch


def compatible_frozen_resume(checkpoint_path, *, candidate, split_id) -> bool:
    """Accept a local legacy unit after unrelated orchestration fields change."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return bool(
        payload.get("format") == "chronaris.dingxin_safe_residual_checkpoint.v1"
        and payload.get("candidate") == asdict(candidate)
        and payload.get("split_id") == split_id
        and payload.get("task_targets_opened") is True
        and payload.get("outer_test_opened") is False
    )


def resolved_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


def preserved_elapsed(path, fallback: float) -> float:
    """Keep the first completed runtime stable across resume-only reruns."""

    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "task_aware_safe_residual_complete":
            return float(payload["elapsed_s"])
    return float(fallback)


def stable_hash(payload) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
