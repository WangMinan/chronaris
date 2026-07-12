"""Label-free modality counterfactuals for raw dual-stream inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch

from chronaris.representation.contracts import DualStreamObservationBatch


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    stream_name: str
    operation: str
    mean_absolute_change: float
    mean_l2_change: float
    relative_l2_change: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_stream_counterfactual(
    batch: DualStreamObservationBatch,
    *,
    stream_name: str,
    operation: str,
    seed: int = 17,
    shift_s: float = 3.0,
    gap_fraction: float = 0.3,
) -> DualStreamObservationBatch:
    """Return a deterministic zero/shuffle/shift/gap stream intervention."""

    if stream_name not in {"physiology", "vehicle"}:
        raise ValueError("stream_name must be physiology or vehicle")
    if operation not in {"zero", "shuffle", "shift", "gap"}:
        raise ValueError("unsupported counterfactual operation")
    if shift_s <= 0 or not 0 < gap_fraction < 1:
        raise ValueError("counterfactual shift/gap configuration is invalid")
    values = getattr(batch, f"{stream_name}_values").clone()
    timestamps = getattr(batch, f"{stream_name}_timestamps_s").clone()
    point_mask = getattr(batch, f"{stream_name}_point_mask").clone()
    feature_mask = getattr(batch, f"{stream_name}_feature_mask").clone()
    observation_age = getattr(batch, f"{stream_name}_observation_age_s").clone()
    if operation == "zero":
        values.zero_()
        feature_mask.zero_()
        point_mask.zero_()
        observation_age.fill_(torch.inf)
    elif operation == "shuffle":
        generator = torch.Generator(device=values.device).manual_seed(seed)
        permutation = torch.randperm(values.shape[0], generator=generator, device=values.device)
        values = values.index_select(0, permutation)
        timestamps = timestamps.index_select(0, permutation)
        point_mask = point_mask.index_select(0, permutation)
        feature_mask = feature_mask.index_select(0, permutation)
        observation_age = observation_age.index_select(0, permutation)
    elif operation == "shift":
        timestamps = timestamps + shift_s
    else:
        point_count = values.shape[1]
        gap_size = max(1, int(round(point_count * gap_fraction)))
        start = max(0, (point_count - gap_size) // 2)
        stop = start + gap_size
        point_mask[:, start:stop] = False
        feature_mask[:, start:stop] = False
        values[:, start:stop] = 0
        observation_age[:, start:stop] = torch.inf
    updates = {
        f"{stream_name}_values": values,
        f"{stream_name}_timestamps_s": timestamps,
        f"{stream_name}_point_mask": point_mask,
        f"{stream_name}_feature_mask": feature_mask,
        f"{stream_name}_observation_age_s": observation_age,
    }
    return replace(batch, **updates)


def compare_representations(
    baseline: torch.Tensor,
    counterfactual: torch.Tensor,
    *,
    stream_name: str,
    operation: str,
) -> CounterfactualResult:
    if baseline.shape != counterfactual.shape:
        raise ValueError("counterfactual representation shape mismatch")
    difference = counterfactual - baseline
    baseline_norm = torch.linalg.vector_norm(baseline.reshape(baseline.shape[0], -1), dim=-1)
    difference_norm = torch.linalg.vector_norm(
        difference.reshape(difference.shape[0], -1),
        dim=-1,
    )
    return CounterfactualResult(
        stream_name=stream_name,
        operation=operation,
        mean_absolute_change=float(difference.abs().mean()),
        mean_l2_change=float(difference_norm.mean()),
        relative_l2_change=float(
            (difference_norm / baseline_norm.clamp_min(1e-12)).mean()
        ),
    )
