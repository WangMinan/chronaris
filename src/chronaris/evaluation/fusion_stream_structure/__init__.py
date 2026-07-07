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

__all__ = [
    "ContractError",
    "FusionStreamContract",
    "FusionStreamRecord",
    "FusionStreamRunConfig",
    "detect_feature_columns",
    "detect_forbidden_sidecar_columns",
    "normalize_method_name",
    "validate_input_frame",
]
