"""Checkpoint-level v4 correctness checks; never select models or open task targets."""

from dataclasses import replace

import torch

from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import perturb_future_observations
from chronaris.modeling.training import TrainedFusionAdapter, load_common_pretraining_checkpoint
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def audit_checkpoint_causality(checkpoint_path, batch, *, device="cuda", cutoff_s=15.0):
    before = sha256_file(checkpoint_path)
    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    guided = "role_sample_ids" in stored
    encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(
        stored["source_checkpoint_path"] if guided else checkpoint_path,
        device=device, allow_legacy_implementation=True,
    )
    if guided:
        encoder.load_state_dict({name.removeprefix("encoder."): value
            for name, value in stored["model_state_dict"].items() if name.startswith("encoder.")}, strict=True)
        if stored.get("normalizer") != normalizer.to_manifest():
            raise ValueError("guided correctness checkpoint normalization differs from initialization")
        payload = stored
    allowed = set(payload["role_sample_ids"]["validation"] if guided else payload["fold"]["validation_sample_ids"])
    if not set(batch.sample_ids) <= allowed:
        raise ValueError("correctness audit requires the checkpoint's internal validation samples")
    adapter = TrainedFusionAdapter(
        encoder=encoder, normalizer=normalizer,
        fold_id=payload["fold_id"] if guided else payload["fold"]["fold_id"], checkpoint_sha256=before,
    )
    baseline = adapter(batch)
    past = baseline.timestamps_s <= cutoff_s
    alterations = {
        "repeat": batch,
        "future_values": perturb_future_observations(batch, cutoff_s=cutoff_s),
        "future_missing": remove_future_observations(batch, cutoff_s=cutoff_s),
    }
    deltas = {}
    for name, changed in alterations.items():
        output = adapter(changed)
        if not torch.equal(baseline.valid_mask[past], output.valid_mask[past]):
            raise AssertionError("future information changed historical availability")
        deltas[name] = float((baseline.sequence_embedding[past] - output.sequence_embedding[past]).abs().max())
    unchanged = sha256_file(checkpoint_path) == before
    assert unchanged
    return {
        "seed": payload["seed"], "device": device, "cutoff_s": cutoff_s,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": before,
        "weight_source_format": payload["format"],
        "scope": "weights_loaded_into_current_implementation_for_diagnosis_only",
        "sample_ids": list(batch.sample_ids), "historical_max_delta": deltas,
        "checkpoint_unchanged": unchanged, "passed": max(deltas.values()) <= 1e-6,
    }


def remove_future_observations(batch, *, cutoff_s):
    updates = {}
    for stream in ("physiology", "vehicle"):
        mask = getattr(batch, f"{stream}_feature_mask") & (
            getattr(batch, f"{stream}_timestamps_s") <= cutoff_s
        ).unsqueeze(-1)
        updates.update({
            f"{stream}_feature_mask": mask,
            f"{stream}_point_mask": mask.any(dim=-1),
            f"{stream}_values": getattr(batch, f"{stream}_values").masked_fill(~mask, 0),
            f"{stream}_observation_age_s": getattr(batch, f"{stream}_observation_age_s").masked_fill(~mask, torch.inf),
        })
    return replace(batch, **updates)
