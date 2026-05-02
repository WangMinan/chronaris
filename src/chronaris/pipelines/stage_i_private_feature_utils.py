"""Utility helpers shared by private Stage I benchmark feature builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def cosine_similarity_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_2d = np.asarray(left, dtype=np.float32)
    right_2d = np.asarray(right, dtype=np.float32)
    if left_2d.ndim == 1:
        left_2d = left_2d.reshape(1, -1)
    if right_2d.ndim == 1:
        right_2d = right_2d.reshape(1, -1)
    left_norm = left_2d / np.clip(np.linalg.norm(left_2d, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    right_norm = right_2d / np.clip(np.linalg.norm(right_2d, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    similarity = left_norm @ right_norm.T
    return similarity.reshape(-1) if left.ndim == 1 else np.diag(similarity)


def row_l2_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).mean())


def mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_norm = left / np.clip(np.linalg.norm(left, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    return float((left_norm * right_norm).sum(axis=-1).mean())


def bucketize_score(score: float, lower_q: float, upper_q: float) -> str:
    if score <= lower_q:
        return "low"
    if score <= upper_q:
        return "medium"
    return "high"


def safe_mean(values: tuple[float, ...] | list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def sanitize_feature_name(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name).strip("_")


def read_jsonl(path_like: object) -> tuple[Mapping[str, object], ...]:
    path = Path(path_like)
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
