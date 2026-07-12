"""Task-independent diagnostics for frozen fusion representations."""

from chronaris.evaluation.representation_diagnostics.counterfactual import (
    CounterfactualResult,
    apply_stream_counterfactual,
    compare_representations,
)
from chronaris.evaluation.representation_diagnostics.fidelity import (
    FeatureRecoveryProbeResult,
    FidelityProbeResult,
    fit_feature_recovery_probe,
    fit_fidelity_probe,
)
from chronaris.evaluation.representation_diagnostics.gradients import (
    gradient_conflict_rows,
)
from chronaris.evaluation.representation_diagnostics.health import (
    RepresentationHealth,
    representation_health,
)
from chronaris.evaluation.representation_diagnostics.internals import (
    fusion_internal_rows,
)

__all__ = [
    "CounterfactualResult",
    "FidelityProbeResult",
    "FeatureRecoveryProbeResult",
    "RepresentationHealth",
    "apply_stream_counterfactual",
    "compare_representations",
    "fit_fidelity_probe",
    "fit_feature_recovery_probe",
    "fusion_internal_rows",
    "gradient_conflict_rows",
    "representation_health",
]
