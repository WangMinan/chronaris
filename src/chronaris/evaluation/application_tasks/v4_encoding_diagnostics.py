"""Unlabeled full-validation state, attention and observation-fit diagnostics."""
from collections import defaultdict
import time

import numpy as np
import torch

from chronaris.modeling.fusion_encoders.alignment_bridge import build_alignment_batch_from_observations
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation import select_observation_batch


def _distribution(values):
    values = np.concatenate(values) if values else np.empty(0)
    if not np.isfinite(values).all():
        raise ValueError("valid diagnostic values are non-finite")
    if not len(values):
        return {"status": "unavailable_no_valid_values", "count": 0}
    return {"status": "completed", "count": len(values), "mean": float(values.mean()),
        **{name: float(value) for name, value in zip(("median", "p95", "p99", "maximum"),
                                                   np.quantile(values, (.5, .95, .99, 1.)), strict=True)}}


def collect_encoding_diagnostics(*, encoder, normalizer, batch, batch_size=4):
    """Keep every valid state; fully unobserved queries do not become zero-norm evidence."""
    started = time.perf_counter()
    if encoder.method_name != "chronaris":
        return {"status": "not_applicable_to_architecture", "method": encoder.method_name}
    encoder.eval()
    device = next(encoder.parameters()).device
    values, squared_errors, point_counts, physical = defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(list)
    def append(name, tensor):
        if tensor.numel():
            if not torch.isfinite(tensor).all():
                raise ValueError("valid diagnostic states are non-finite")
            values[name].append(tensor.detach().float().cpu().numpy().reshape(-1))
    with torch.inference_mode():
        for start in range(0, len(batch.sample_ids), batch_size):
            raw = select_observation_batch(batch, batch.sample_ids[start:start + batch_size])
            normalized = move_observation_batch(normalizer.transform(raw), device=device)
            output = encoder(normalized, compute_chronaris_diagnostics=True)
            alignment = output.auxiliary["alignment_output"]
            aligned_input = build_alignment_batch_from_observations(normalized,
                physiology_feature_names=alignment.physiology.feature_names, vehicle_feature_names=alignment.vehicle.feature_names)
            for stream in ("physiology", "vehicle"):
                encoded, observed = getattr(alignment, stream), getattr(aligned_input, stream)
                valid = encoded.reference_valid_mask
                point_counts[stream + "_valid_queries"] += int(valid.sum())
                point_counts[stream + "_unobserved_windows"] += int((~valid.any(dim=1)).sum())
                append(stream + "_query_state_norm", encoded.reference_hidden_states[valid].norm(dim=-1))
                append(stream + "_observed_state_norm", encoded.updated_hidden_states[encoded.mask].norm(dim=-1))
                append(stream + "_pre_update_state_norm", encoded.evolved_hidden_states[encoded.mask].norm(dim=-1))
                append(stream + "_observation_gap_s", encoded.delta_t_s[encoded.mask])
                for field, name in enumerate(encoded.feature_names):
                    mask = observed.feature_valid_mask[:, :, field] & observed.mask
                    residual = encoded.reconstructions[:, :, field][mask] - observed.values[:, :, field][mask]
                    if not torch.isfinite(residual).all():
                        raise ValueError("valid observation decoder output is non-finite")
                    squared_errors[(stream, name)].extend(residual.square().cpu().tolist())
            fusion = output.auxiliary["fusion_output"]
            pairing = output.auxiliary.get("independent_pairing")
            if pairing is not None:
                for stream in ("physiology", "vehicle"):
                    append(stream + "_independent_pair_norm", getattr(pairing, stream)[getattr(pairing, stream + "_valid")].norm(dim=-1))
            for index, attention in enumerate(fusion.attention_weights):
                mask = fusion.lag_masks[index]
                available = mask.any(dim=-1)
                entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
                append(f"attention_scale_{index}_entropy", entropy[available])
                append(f"attention_scale_{index}_maximum_weight", attention.max(dim=-1).values[available])
                append(f"attention_scale_{index}_valid_key_count", mask.sum(dim=-1)[available])
            cross_valid = alignment.physiology.reference_valid_mask & fusion.scale_available_mask.any(dim=-1)
            append("cross_gate", fusion.cross_gate.squeeze(-1)[cross_valid])
            append("scale_gate_entropy", -(fusion.scale_gate_weights[cross_valid]
                * fusion.scale_gate_weights[cross_valid].clamp_min(1e-12).log()).sum(dim=-1))
            audit = output.auxiliary["physical_consistency"]
            for component in audit.components:
                if component.available and component.raw_value is not None:
                    physical[component.component_name].append((float(component.raw_value), component.count))
    distributions = {name: _distribution(items) for name, items in values.items()}
    for stream in ("physiology", "vehicle"):
        for name in ("query_state_norm", "observed_state_norm", "pre_update_state_norm", "observation_gap_s"):
            distributions.setdefault(stream + "_" + name, _distribution([]))
    reconstruction = []
    for (stream, field), errors in squared_errors.items():
        error = np.asarray(errors, dtype=float)
        reconstruction.append({"stream": stream, "field": field, "count": len(error),
            "standardized_rmse": float(np.sqrt(error.mean())) if len(error) else None,
            "status": "completed" if len(error) else "unavailable_no_observations"})
    return {"status": "completed", "method": encoder.method_name, "sample_count": len(batch.sample_ids),
        "validity_counts": dict(point_counts), "distributions": distributions, "observation_fit": reconstruction,
        "physical_components": [{"component": name, "count": sum(count for _, count in rows),
            "mean_normalized_huber_residual": sum(value * count for value, count in rows) / sum(count for _, count in rows)}
            for name, rows in physical.items()],
        "attention_kind": encoder.backbone.config.attention_kind,
        "attention_temperature": float(torch.as_tensor(encoder.backbone.causal_fusion.effective_attention_temperature).detach())
            if hasattr(encoder.backbone.causal_fusion, "effective_attention_temperature") else None,
        "independent_pairing_enabled": encoder.backbone.config.independent_pairing_enabled,
        "event_pairing": "independent_native_history_windows" if encoder.backbone.config.independent_pairing_enabled
            else "disabled_shared_bank_pairing_not_evidence_of_independent_pairs",
        "label_used": False, "elapsed_s": time.perf_counter() - started}
