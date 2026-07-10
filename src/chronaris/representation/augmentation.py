"""Method-invariant augmentation realization planning for shared training batches."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentationPolicy:
    point_dropout_probability: float = 0.10
    block_duration_min_s: float = 1.0
    block_duration_max_s: float = 5.0
    modality_dropout_probability: float = 0.05
    timestamp_jitter_sigma_s: float = 0.05
    clock_offset_limit_s: float = 0.25

    def __post_init__(self) -> None:
        for name in ("point_dropout_probability", "modality_dropout_probability"):
            value = float(getattr(self, name))
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if not 0.005 <= self.timestamp_jitter_sigma_s <= 0.100:
            raise ValueError("timestamp jitter sigma must be calibrated to 5-100 ms")
        if not 0 < self.block_duration_min_s <= self.block_duration_max_s:
            raise ValueError("block duration bounds are invalid")
        if self.clock_offset_limit_s < 0:
            raise ValueError("clock_offset_limit_s must be non-negative")


@dataclass(frozen=True, slots=True)
class AugmentationRealization:
    augmentation_id: str
    sample_id: str
    epoch: int
    global_seed: int
    physiology_block_start_s: float
    physiology_block_duration_s: float
    vehicle_block_start_s: float
    vehicle_block_duration_s: float
    dropped_modality: str | None
    physiology_clock_offset_s: float
    vehicle_clock_offset_s: float
    physiology_jitter_seed: int
    vehicle_jitter_seed: int
    physiology_point_dropout_seed: int
    vehicle_point_dropout_seed: int

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


def build_augmentation_realization(
    *,
    sample_id: str,
    epoch: int,
    global_seed: int,
    context_duration_s: float = 30.0,
    policy: AugmentationPolicy | None = None,
) -> AugmentationRealization:
    """Create one stable realization without accepting a method name."""

    resolved = policy or AugmentationPolicy()
    if not sample_id or epoch < 0 or context_duration_s <= 0:
        raise ValueError("augmentation sample/epoch/context is invalid")
    digest = hashlib.sha256(
        f"{global_seed}\0{epoch}\0{sample_id}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little")
    generator = np.random.default_rng(seed)
    physiology_duration = float(
        generator.uniform(
            resolved.block_duration_min_s,
            resolved.block_duration_max_s,
        )
    )
    vehicle_duration = float(
        generator.uniform(
            resolved.block_duration_min_s,
            resolved.block_duration_max_s,
        )
    )
    modality_draw = float(generator.random())
    if modality_draw < resolved.modality_dropout_probability:
        dropped_modality = "physiology"
    elif modality_draw < 2 * resolved.modality_dropout_probability:
        dropped_modality = "vehicle"
    else:
        dropped_modality = None
    seeds = generator.integers(0, np.iinfo(np.int32).max, size=4, dtype=np.int64)
    return AugmentationRealization(
        augmentation_id=hashlib.sha256(digest + b"augmentation.v1").hexdigest(),
        sample_id=sample_id,
        epoch=int(epoch),
        global_seed=int(global_seed),
        physiology_block_start_s=float(
            generator.uniform(0, max(context_duration_s - physiology_duration, 0))
        ),
        physiology_block_duration_s=physiology_duration,
        vehicle_block_start_s=float(
            generator.uniform(0, max(context_duration_s - vehicle_duration, 0))
        ),
        vehicle_block_duration_s=vehicle_duration,
        dropped_modality=dropped_modality,
        physiology_clock_offset_s=float(
            generator.uniform(-resolved.clock_offset_limit_s, resolved.clock_offset_limit_s)
        ),
        vehicle_clock_offset_s=float(
            generator.uniform(-resolved.clock_offset_limit_s, resolved.clock_offset_limit_s)
        ),
        physiology_jitter_seed=int(seeds[0]),
        vehicle_jitter_seed=int(seeds[1]),
        physiology_point_dropout_seed=int(seeds[2]),
        vehicle_point_dropout_seed=int(seeds[3]),
    )


def build_batch_augmentation_realizations(
    sample_ids: Sequence[str],
    *,
    epoch: int,
    global_seed: int,
    context_duration_s: float = 30.0,
    policy: AugmentationPolicy | None = None,
) -> tuple[AugmentationRealization, ...]:
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("augmentation sample IDs must be unique")
    return tuple(
        build_augmentation_realization(
            sample_id=sample_id,
            epoch=epoch,
            global_seed=global_seed,
            context_duration_s=context_duration_s,
            policy=policy,
        )
        for sample_id in sample_ids
    )
