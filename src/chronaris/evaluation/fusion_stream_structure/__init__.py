"""Fusion stream structure evaluation (E3)."""

from chronaris.evaluation.fusion_stream_structure.contracts import (
    ContractError,
    FusionStreamContract,
    FusionStreamRecord,
    FusionStreamRunConfig,
    detect_feature_columns,
    detect_forbidden_sidecar_columns,
    normalize_method_name,
    validate_input_frame,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import (
    DeepBaselineRepresentationExportConfig,
    DeepBaselineRepresentationExportResult,
)

__all__ = [
    "ContractError",
    "DeepBaselineRepresentationExportConfig",
    "DeepBaselineRepresentationExportResult",
    "FusionStreamContract",
    "FusionStreamRecord",
    "FusionStreamRunConfig",
    "detect_feature_columns",
    "detect_forbidden_sidecar_columns",
    "normalize_method_name",
    "validate_input_frame",
]
