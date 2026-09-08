"""History-only observation quality at each query; no trajectory-wide calibration."""
import torch

from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream


def causal_observation_quality(batch):
    """Per modality: mean field age, observed field/second coverage, current gap.

    Coverage counts 1-second bins starting at context time zero, including only
    observations visible at the query within its current bin. Ages/gaps saturate
    at 30 seconds, the maximum input duration in the frozen v4 tasks.
    """
    queries = batch.query_timestamps_s
    if bool((queries < 0).any()):
        raise ValueError("observation quality requires non-negative context queries")
    query_bins = queries.floor().long()
    bin_count = int(query_bins.max()) + 1
    outputs = []
    for stream in ("physiology", "vehicle"):
        sampled = causal_query_stream(batch, stream_name=stream)
        ages = sampled.observation_age_s.clamp(max=30.) / 30.
        coverage = torch.zeros_like(queries, dtype=sampled.values.dtype)
        times = getattr(batch, f"{stream}_timestamps_s")
        masks = getattr(batch, f"{stream}_feature_mask")
        points = getattr(batch, f"{stream}_point_mask")
        features = masks.shape[-1]
        for b in range(len(batch.sample_ids)):
            valid = masks[b] & points[b, :, None] & (times[b, :, None] <= queries[b, -1])
            bins = times[b].floor().long().clamp(0, bin_count - 1)
            first = times.new_full((bin_count, features), torch.inf)
            first.scatter_reduce_(0, bins[:, None].expand(-1, features),
                times[b, :, None].expand(-1, features).masked_fill(~valid, torch.inf), reduce="amin")
            filled = torch.isfinite(first).sum(dim=-1)
            previous = filled.cumsum(0) - filled
            current = first[query_bins[b]] <= queries[b, :, None]
            coverage[b] = (previous[query_bins[b]] + current.sum(dim=-1)) / ((query_bins[b] + 1) * features)
        outputs.extend((ages.mean(dim=-1), coverage, ages.min(dim=-1).values))
    return torch.stack(outputs, dim=-1)
