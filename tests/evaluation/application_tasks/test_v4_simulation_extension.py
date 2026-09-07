import json
from copy import deepcopy

import pytest

from chronaris.evaluation.application_tasks.v4_simulation_extension import _verify_extension_decision
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def test_extension_requires_complete_paired_matrix_and_the_predeclared_loss_improvements(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed source checkpoint")
    decision = {"allowed_extension_count": 1, "additional_profiles": 32, "additional_trajectories_per_profile": 8,
        "approved_training_trajectories_after_activation": 512, "confirmation_feedback_used": False,
        "trigger_method": "chronaris", "route": "self_supervised", "training_checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "validation_losses": {"200": 1., "400": .9, "500": .8}}
    for method in ("chronaris", "physiology_only", "vehicle_only", "mult", "contiformer"):
        root = tmp_path / "simulation" / method
        root.mkdir(parents=True)
        state = {"completed": True, "seed": 17, "method": method,
            "completed_consumers": [f"{route}:{update}" for route in ("self_supervised", "task_guided") for update in (50, 200, 500)],
            "self_supervised_training": {"optimizer_updates": 500, "epoch_rows": [
                {"optimizer_updates": int(key), "public_selection_loss": value} for key, value in decision["validation_losses"].items()]},
            "task_guided_training": {"optimizer_updates": 550}}
        (root / "run_state.json").write_text(json.dumps(state))
    assert len(_verify_extension_decision(decision, tmp_path)) == 5
    changed = deepcopy(decision)
    changed["validation_losses"]["500"] = .95
    with pytest.raises(ValueError, match="trigger"):
        _verify_extension_decision(changed, tmp_path)
    path = tmp_path / "simulation/contiformer/run_state.json"
    state = json.loads(path.read_text()); state["task_guided_training"]["optimizer_updates"] = 549
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="incomplete"):
        _verify_extension_decision(decision, tmp_path)
