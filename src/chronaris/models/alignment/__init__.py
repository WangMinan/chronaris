"""Continuous-time alignment models."""

from typing import Any

from chronaris.models.alignment.config import AlignmentPrototypeConfig
from chronaris.models.alignment.splits import (
    ChronologicalSampleSplit,
    ChronologicalSplitConfig,
    split_e0_samples_chronologically,
)
from chronaris.models.alignment.reference_grid import (
    ReferenceGrid,
    ReferenceGridConfig,
    build_reference_grid,
    build_reference_grids,
)

_BATCHING_EXPORTS = {
    "AlignmentBatch",
    "AlignmentStreamBatch",
    "build_alignment_batch",
}
_TORCH_BATCH_EXPORTS = {
    "TorchAlignmentBatch",
    "TorchAlignmentStreamBatch",
    "build_torch_alignment_batch",
}
_PROTOTYPE_EXPORTS = {
    "DualStreamODERNNPrototype",
    "DualStreamPrototypeOutput",
    "SingleStreamODERNNPrototype",
    "StreamPrototypeOutput",
}
_TASK_HEAD_EXPORTS = {
    "ClassificationTaskHead",
    "RegressionTaskHead",
    "RetrievalTaskHead",
    "StageITaskHeadBatch",
    "StageITaskHeadOutput",
    "StageITaskHeadSet",
    "StageITaskHeadSpec",
}
_LOSS_EXPORTS = {
    "AlignmentLossBreakdown",
    "PhysicsLossBreakdown",
    "RigidBodyPhysicsDiagnostics",
    "RigidBodyStateMapping",
    "StageEObjectiveBreakdown",
    "TaskLossBreakdown",
    "build_rigid_body_state_mapping",
    "build_task_loss_breakdown",
    "build_stage_f_physics_losses",
    "build_stage_e_objective",
    "build_physiology_feature_groups",
    "build_vehicle_feature_groups",
    "dual_stream_alignment_loss",
    "vehicle_physics_consistency_loss",
    "physiology_physics_consistency_loss",
    "projection_alignment_loss",
    "ReconstructionLossBreakdown",
    "StageFPhysicsContext",
    "StageFPhysiologyFeatureGroups",
    "StageFVehicleFeatureGroups",
    "dual_stream_reconstruction_loss",
    "masked_mean_squared_error",
    "inspect_rigid_body_physics",
    "stream_reconstruction_loss",
}

__all__ = [
    "AlignmentPrototypeConfig",
    "ChronologicalSampleSplit",
    "ChronologicalSplitConfig",
    "ReferenceGrid",
    "ReferenceGridConfig",
    "build_reference_grid",
    "build_reference_grids",
    "split_e0_samples_chronologically",
] + sorted(_BATCHING_EXPORTS | _TORCH_BATCH_EXPORTS | _PROTOTYPE_EXPORTS | _TASK_HEAD_EXPORTS | _LOSS_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _BATCHING_EXPORTS:
        from chronaris.models.alignment.batching import AlignmentBatch, AlignmentStreamBatch, build_alignment_batch

        exports = {
            "AlignmentBatch": AlignmentBatch,
            "AlignmentStreamBatch": AlignmentStreamBatch,
            "build_alignment_batch": build_alignment_batch,
        }
        globals().update(exports)
        return exports[name]
    if name in _TORCH_BATCH_EXPORTS:
        from chronaris.models.alignment.torch_batch import (
            TorchAlignmentBatch,
            TorchAlignmentStreamBatch,
            build_torch_alignment_batch,
        )

        exports = {
            "TorchAlignmentBatch": TorchAlignmentBatch,
            "TorchAlignmentStreamBatch": TorchAlignmentStreamBatch,
            "build_torch_alignment_batch": build_torch_alignment_batch,
        }
        globals().update(exports)
        return exports[name]
    if name in _PROTOTYPE_EXPORTS:
        from chronaris.models.alignment.prototype import (
            DualStreamODERNNPrototype,
            DualStreamPrototypeOutput,
            SingleStreamODERNNPrototype,
            StreamPrototypeOutput,
        )

        exports = {
            "DualStreamODERNNPrototype": DualStreamODERNNPrototype,
            "DualStreamPrototypeOutput": DualStreamPrototypeOutput,
            "SingleStreamODERNNPrototype": SingleStreamODERNNPrototype,
            "StreamPrototypeOutput": StreamPrototypeOutput,
        }
        globals().update(exports)
        return exports[name]
    if name in _TASK_HEAD_EXPORTS:
        from chronaris.models.alignment.task_heads import (
            ClassificationTaskHead,
            RegressionTaskHead,
            RetrievalTaskHead,
            StageITaskHeadBatch,
            StageITaskHeadOutput,
            StageITaskHeadSet,
            StageITaskHeadSpec,
        )

        exports = {
            "ClassificationTaskHead": ClassificationTaskHead,
            "RegressionTaskHead": RegressionTaskHead,
            "RetrievalTaskHead": RetrievalTaskHead,
            "StageITaskHeadBatch": StageITaskHeadBatch,
            "StageITaskHeadOutput": StageITaskHeadOutput,
            "StageITaskHeadSet": StageITaskHeadSet,
            "StageITaskHeadSpec": StageITaskHeadSpec,
        }
        globals().update(exports)
        return exports[name]
    if name in _LOSS_EXPORTS:
        from chronaris.models.alignment.losses import (
            AlignmentLossBreakdown,
            PhysicsLossBreakdown,
            ReconstructionLossBreakdown,
            StageEObjectiveBreakdown,
            TaskLossBreakdown,
            build_stage_e_objective,
            build_stage_f_physics_losses,
            build_task_loss_breakdown,
            dual_stream_alignment_loss,
            dual_stream_reconstruction_loss,
            masked_mean_squared_error,
            physiology_physics_consistency_loss,
            projection_alignment_loss,
            stream_reconstruction_loss,
            vehicle_physics_consistency_loss,
        )
        from chronaris.models.alignment.physics_features import (
            StageFPhysicsContext,
            StageFPhysiologyFeatureGroups,
            StageFVehicleFeatureGroups,
            build_physiology_feature_groups,
            build_vehicle_feature_groups,
        )
        from chronaris.models.alignment.physics_state_mapping import (
            RigidBodyPhysicsDiagnostics,
            RigidBodyStateMapping,
            build_rigid_body_state_mapping,
            inspect_rigid_body_physics,
        )

        exports = {
            "AlignmentLossBreakdown": AlignmentLossBreakdown,
            "PhysicsLossBreakdown": PhysicsLossBreakdown,
            "ReconstructionLossBreakdown": ReconstructionLossBreakdown,
            "RigidBodyPhysicsDiagnostics": RigidBodyPhysicsDiagnostics,
            "RigidBodyStateMapping": RigidBodyStateMapping,
            "StageEObjectiveBreakdown": StageEObjectiveBreakdown,
            "TaskLossBreakdown": TaskLossBreakdown,
            "StageFPhysicsContext": StageFPhysicsContext,
            "StageFPhysiologyFeatureGroups": StageFPhysiologyFeatureGroups,
            "StageFVehicleFeatureGroups": StageFVehicleFeatureGroups,
            "build_rigid_body_state_mapping": build_rigid_body_state_mapping,
            "build_stage_e_objective": build_stage_e_objective,
            "build_stage_f_physics_losses": build_stage_f_physics_losses,
            "build_task_loss_breakdown": build_task_loss_breakdown,
            "build_physiology_feature_groups": build_physiology_feature_groups,
            "build_vehicle_feature_groups": build_vehicle_feature_groups,
            "dual_stream_alignment_loss": dual_stream_alignment_loss,
            "dual_stream_reconstruction_loss": dual_stream_reconstruction_loss,
            "inspect_rigid_body_physics": inspect_rigid_body_physics,
            "masked_mean_squared_error": masked_mean_squared_error,
            "physiology_physics_consistency_loss": physiology_physics_consistency_loss,
            "projection_alignment_loss": projection_alignment_loss,
            "stream_reconstruction_loss": stream_reconstruction_loss,
            "vehicle_physics_consistency_loss": vehicle_physics_consistency_loss,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
