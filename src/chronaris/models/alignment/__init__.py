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
    "ContrastiveRetrievalProjectionHead",
    "GateContributionSummary",
    "PhysiologyResponseResidualRegressionHead",
    "RegressionTaskHead",
    "RetrievalTaskHead",
    "StageITaskHeadBatch",
    "StageITaskHeadOutput",
    "StageITaskHeadSet",
    "StageITaskHeadSpec",
    "VehicleAuxClassificationOutput",
    "VehicleDominantAuxiliaryClassificationHead",
    "gate_regularization_loss",
}
_TASK_LOSS_V2_EXPORTS = {
    "TrainFoldTargetTransform",
    "class_balanced_cross_entropy",
    "focal_cross_entropy",
    "persistence_improvement_rate",
    "residual_regression_loss",
}
_CONTRASTIVE_EXPORTS = {
    "RetrievalCandidate",
    "build_positive_index_tensor",
    "info_nce_loss",
    "retrieval_metrics_from_scores",
    "validate_same_sortie_cross_pilot_policy",
}
_LOSS_EXPORTS = {
    "AlignmentLossBreakdown",
    "PhysicsLossBreakdown",
    "RigidBodyMappingDiagnostics",
    "RigidBodyPhysicsDiagnostics",
    "RigidBodyStateMapping",
    "build_rigid_body_mapping_diagnostics",
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
] + sorted(
    _BATCHING_EXPORTS
    | _TORCH_BATCH_EXPORTS
    | _PROTOTYPE_EXPORTS
    | _TASK_HEAD_EXPORTS
    | _TASK_LOSS_V2_EXPORTS
    | _CONTRASTIVE_EXPORTS
    | _LOSS_EXPORTS
)


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
        from chronaris.models.alignment.task_heads_v2 import (
            ContrastiveRetrievalProjectionHead,
            GateContributionSummary,
            PhysiologyResponseResidualRegressionHead,
            VehicleAuxClassificationOutput,
            VehicleDominantAuxiliaryClassificationHead,
            gate_regularization_loss,
        )

        exports = {
            "ClassificationTaskHead": ClassificationTaskHead,
            "ContrastiveRetrievalProjectionHead": ContrastiveRetrievalProjectionHead,
            "GateContributionSummary": GateContributionSummary,
            "PhysiologyResponseResidualRegressionHead": PhysiologyResponseResidualRegressionHead,
            "RegressionTaskHead": RegressionTaskHead,
            "RetrievalTaskHead": RetrievalTaskHead,
            "StageITaskHeadBatch": StageITaskHeadBatch,
            "StageITaskHeadOutput": StageITaskHeadOutput,
            "StageITaskHeadSet": StageITaskHeadSet,
            "StageITaskHeadSpec": StageITaskHeadSpec,
            "VehicleAuxClassificationOutput": VehicleAuxClassificationOutput,
            "VehicleDominantAuxiliaryClassificationHead": VehicleDominantAuxiliaryClassificationHead,
            "gate_regularization_loss": gate_regularization_loss,
        }
        globals().update(exports)
        return exports[name]
    if name in _TASK_LOSS_V2_EXPORTS:
        from chronaris.models.alignment.task_losses_v2 import (
            TrainFoldTargetTransform,
            class_balanced_cross_entropy,
            focal_cross_entropy,
            persistence_improvement_rate,
            residual_regression_loss,
        )

        exports = {
            "TrainFoldTargetTransform": TrainFoldTargetTransform,
            "class_balanced_cross_entropy": class_balanced_cross_entropy,
            "focal_cross_entropy": focal_cross_entropy,
            "persistence_improvement_rate": persistence_improvement_rate,
            "residual_regression_loss": residual_regression_loss,
        }
        globals().update(exports)
        return exports[name]
    if name in _CONTRASTIVE_EXPORTS:
        from chronaris.models.alignment.contrastive import (
            RetrievalCandidate,
            build_positive_index_tensor,
            info_nce_loss,
            retrieval_metrics_from_scores,
            validate_same_sortie_cross_pilot_policy,
        )

        exports = {
            "RetrievalCandidate": RetrievalCandidate,
            "build_positive_index_tensor": build_positive_index_tensor,
            "info_nce_loss": info_nce_loss,
            "retrieval_metrics_from_scores": retrieval_metrics_from_scores,
            "validate_same_sortie_cross_pilot_policy": validate_same_sortie_cross_pilot_policy,
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
            RigidBodyMappingDiagnostics,
            RigidBodyPhysicsDiagnostics,
            RigidBodyStateMapping,
            build_rigid_body_mapping_diagnostics,
            build_rigid_body_state_mapping,
            inspect_rigid_body_physics,
        )

        exports = {
            "AlignmentLossBreakdown": AlignmentLossBreakdown,
            "PhysicsLossBreakdown": PhysicsLossBreakdown,
            "ReconstructionLossBreakdown": ReconstructionLossBreakdown,
            "RigidBodyMappingDiagnostics": RigidBodyMappingDiagnostics,
            "RigidBodyPhysicsDiagnostics": RigidBodyPhysicsDiagnostics,
            "RigidBodyStateMapping": RigidBodyStateMapping,
            "build_rigid_body_mapping_diagnostics": build_rigid_body_mapping_diagnostics,
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
