"""Read-only checkpoint diagnosis; no candidate selection or model repair."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import perturb_future_observations
from chronaris.evaluation.application_tasks.thesis_simulation_gates import (
    PRETRAINING, SEEDS, SIMULATION_ROOT, _adapter,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import load_simulation_locked_pretraining_data
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.models.fusion.causal import compute_vehicle_event_scores
from chronaris.representation import select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def main():
    data = load_simulation_locked_pretraining_data(SIMULATION_ROOT)
    raw = select_observation_batch(data.batch, data.fold.validation_sample_ids[:4])
    perturbed = perturb_future_observations(raw, cutoff_s=15.0)
    rows = []
    for device, seeds in (("cuda", SEEDS), ("cpu", (17,))):
        for seed in seeds:
            checkpoint = PRETRAINING / f"checkpoints/seed_{seed}/chronaris/best.pt"
            checkpoint_hash = sha256_file(checkpoint)
            adapter = _adapter(checkpoint, data.fold.fold_id, device)
            captured = []
            hook = adapter.encoder.backbone.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
            baseline = adapter(raw)
            repeat = adapter(raw)
            changed = adapter(perturbed)
            original_encoding, _, changed_encoding = captured
            past = baseline.timestamps_s <= 15.0

            def delta(left, right):
                return float((left[past] - right[past]).abs().max())

            normal = move_observation_batch(adapter.normalizer.transform(raw), device=device)
            altered = move_observation_batch(adapter.normalizer.transform(perturbed), device=device)
            input_deltas = {}
            for stream in ("physiology", "vehicle"):
                visible = getattr(normal, f"{stream}_timestamps_s") <= 15.0
                input_deltas[stream] = float((
                    getattr(normal, f"{stream}_values")[visible]
                    - getattr(altered, f"{stream}_values")[visible]
                ).abs().max())
            alignment_deltas = {
                stream: delta(
                    getattr(original_encoding.alignment_output, stream).reference_hidden_states,
                    getattr(changed_encoding.alignment_output, stream).reference_hidden_states,
                )
                for stream in ("physiology", "vehicle")
            }
            original_vehicle = original_encoding.alignment_output.vehicle.reference_hidden_states
            changed_vehicle = changed_encoding.alignment_output.vehicle.reference_hidden_states
            original_scores = compute_vehicle_event_scores(original_vehicle)
            changed_scores = compute_vehicle_event_scores(changed_vehicle)
            # Diagnostic intervention only: hold the event-score tensor fixed,
            # not a replacement normalization rule or a proposed model candidate.
            with patch(
                "chronaris.modeling.fusion_encoders.chronaris_continuous.compute_vehicle_event_scores",
                return_value=original_scores,
            ):
                fixed_scores = adapter(perturbed)
            hook.remove()
            row = {
                "seed": seed,
                "device": device,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_unchanged": sha256_file(checkpoint) == checkpoint_hash,
                "historical_normalized_input_delta": input_deltas,
                "historical_alignment_delta": alignment_deltas,
                "repeat_baseline_delta": delta(baseline.sequence_embedding, repeat.sequence_embedding),
                "historical_sequence_delta": delta(baseline.sequence_embedding, changed.sequence_embedding),
                "historical_branch_delta": {
                    name: delta(baseline.sequence_embedding[..., start:end], changed.sequence_embedding[..., start:end])
                    for name, start, end in (("physiology_bypass", 0, 24), ("vehicle_bypass", 24, 48), ("cross_branch", 48, 64))
                },
                "historical_lag_attention_delta": delta(original_encoding.aggregated_lag_attention, changed_encoding.aggregated_lag_attention),
                "historical_event_score_delta": delta(original_scores, changed_scores),
                "historical_delta_with_reference_scores": delta(baseline.sequence_embedding, fixed_scores.sequence_embedding),
            }
            assert row["checkpoint_unchanged"]
            assert all(value == 0 for value in input_deltas.values())
            assert all(value == 0 for value in alignment_deltas.values())
            assert row["repeat_baseline_delta"] == 0
            assert row["historical_sequence_delta"] > 1e-6
            assert row["historical_event_score_delta"] > 0
            assert row["historical_lag_attention_delta"] == 0
            assert row["historical_branch_delta"]["physiology_bypass"] == 0
            assert row["historical_branch_delta"]["vehicle_bypass"] == 0
            assert row["historical_delta_with_reference_scores"] <= 1e-6
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "purpose": "read_only_failure_localization_not_model_selection",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip(),
        "sample_ids": list(raw.sample_ids),
        "cutoff_s": 15.0,
        "future_perturbation": "valid observations strictly after cutoff plus 100000",
        "threshold": 1e-6,
        "outer_results_authorized": False,
        "script_sha256": sha256_file(__file__),
        "rows": rows,
    }
    Path(__file__).with_name("future_information_diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
