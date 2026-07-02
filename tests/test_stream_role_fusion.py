"""Tests for Stage I stream-role-aware fusion routing."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.models.fusion import (  # noqa: E402
    FusionRoute,
    RoleAwareCausalFusion,
    private_stream_metadata,
    public_stream_metadata,
)
from chronaris.pipelines.stage_i.common.deep_models import build_stage_i_deep_model  # noqa: E402
from chronaris.pipelines.stage_i.common.deep_role_aware import ChronarisRoleAwareFusionWrapper  # noqa: E402
from chronaris.pipelines.stage_i.public.fusion_ablation import _select_variants  # noqa: E402


class StreamRoleFusionTest(unittest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physiology = torch.randn(3, 4, 6)
        vehicle = torch.randn(3, 4, 6)
        offsets = torch.arange(4, dtype=torch.float32).view(1, -1).repeat(3, 1)
        return physiology, vehicle, offsets

    def test_private_real_vehicle_uses_causal_lagged_route(self) -> None:
        physiology, vehicle, offsets = self._inputs()
        model = RoleAwareCausalFusion(hidden_dim=6)
        output = model(
            physiology_states=physiology,
            second_stream_states=vehicle,
            physiology_offsets_s=offsets,
            second_stream_offsets_s=offsets,
            metadata=private_stream_metadata(),
        )
        self.assertEqual(output.route_decision.route, FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO)
        self.assertTrue(output.metadata.second_stream_is_real_vehicle)
        self.assertEqual(tuple(output.fused_states.shape), (3, 4, 18))

    def test_public_context_proxy_is_not_real_vehicle(self) -> None:
        metadata = public_stream_metadata("uab_workload_dataset")
        self.assertFalse(metadata.second_stream_is_real_vehicle)
        self.assertIn("context_proxy", metadata.causal_policy)

    def test_public_variants_run(self) -> None:
        physiology, context, offsets = self._inputs()
        model = RoleAwareCausalFusion(hidden_dim=6)
        for variant in (
            "chronaris_v3_stream_role",
            "chronaris_v3_stream_role_fusion",
            "v3_stream_role",
            "v3_fixed_causal_lag",
            "v3_force_private_causal",
            "v3_context_adapter_only",
            "p37_public_no_lag_prior_context085",
            "p37_public_context_adapter_only_cap2x_do0p2",
        ):
            output = model(
                physiology_states=physiology,
                second_stream_states=context,
                physiology_offsets_s=offsets,
                second_stream_offsets_s=offsets,
                metadata=public_stream_metadata("nasa_csm"),
                variant=variant,
            )
            self.assertEqual(tuple(output.fused_states.shape), (3, 4, 18))
            if variant.startswith("p37_public"):
                self.assertFalse(output.metadata.second_stream_is_real_vehicle)
                self.assertGreaterEqual(float(output.route_decision.context_gate.mean()), 0.75)

    def test_deep_factory_threads_private_stream_metadata(self) -> None:
        model = build_stage_i_deep_model(
            model_name="chronaris_v3_stream_role_fusion",
            ordered_modalities=("physiology", "vehicle"),
            modality_input_dims={"physiology": 5, "vehicle": 4},
            output_dim=3,
            dataset_id="private_stage_h",
        )
        self.assertIsInstance(model, ChronarisRoleAwareFusionWrapper)
        self.assertTrue(model.metadata.second_stream_is_real_vehicle)
        self.assertEqual(model.metadata.second_stream_role.value, "real_vehicle")

    def test_public_p35_variants_are_selectable(self) -> None:
        variants = _select_variants(
            (
                "v3_stream_role",
                "v3_no_role_gate",
                "v3_force_private_causal",
                "v3_context_adapter_only",
                "p37_public_no_lag_prior_context075",
                "p37_public_context_adapter_only_cap2x_do0p2",
            )
        )
        self.assertEqual(
            {variant.variant_id for variant in variants},
            {
                "v3_stream_role",
                "v3_no_role_gate",
                "v3_force_private_causal",
                "v3_context_adapter_only",
                "p37_public_no_lag_prior_context075",
                "p37_public_context_adapter_only_cap2x_do0p2",
            },
        )


if __name__ == "__main__":
    unittest.main()
