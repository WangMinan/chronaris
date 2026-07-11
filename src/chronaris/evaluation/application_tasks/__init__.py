"""Application-facing downstream task audits and benchmarks."""

from chronaris.evaluation.application_tasks.fixed_data_audit import (
    FixedDataAuditConfig,
    FixedDataAuditResult,
    run_fixed_data_audit,
)
from chronaris.evaluation.application_tasks.fixed_data_snapshot import (
    DEFAULT_SNAPSHOT_RUN_ID,
    FixedDataSnapshotConfig,
    run_fixed_data_snapshot,
)
from chronaris.evaluation.application_tasks.deep_baseline_adapter_smoke import (
    DeepBaselineAdapterSmokeConfig,
    DeepBaselineAdapterSmokeResult,
    run_deep_baseline_adapter_smoke,
)
from chronaris.evaluation.application_tasks.chronaris_continuous_adapter_smoke import (
    ChronarisContinuousAdapterSmokeConfig,
    ChronarisContinuousAdapterSmokeResult,
    run_chronaris_continuous_adapter_smoke,
)
from chronaris.evaluation.application_tasks.common_pretraining_loop_smoke import (
    CommonPretrainingLoopSmokeConfig,
    CommonPretrainingLoopSmokeResult,
    run_common_pretraining_loop_smoke,
)
from chronaris.evaluation.application_tasks.application_consumer_smoke import (
    ApplicationConsumerSmokeConfig,
    ApplicationConsumerSmokeResult,
    run_application_consumer_smoke,
)
from chronaris.evaluation.application_tasks.application_frozen_evaluation import (
    ApplicationFrozenEvaluationResult,
    evaluate_frozen_application_consumers,
)
from chronaris.evaluation.application_tasks.dingxin_target_archive import (
    DingxinTargetArchiveConfig,
    DingxinTargetArchiveResult,
    run_dingxin_target_archive,
)
from chronaris.evaluation.application_tasks.dingxin_context_binding import (
    DingxinContextBindingConfig,
    DingxinContextBindingResult,
    run_dingxin_context_binding_audit,
)
from chronaris.evaluation.application_tasks.dingxin_inner_split_run import (
    DingxinInnerSplitConfig,
    DingxinInnerSplitResult,
    run_dingxin_inner_split,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_run import (
    DingxinFoldPretrainingConfig,
    DingxinFoldPretrainingResult,
    run_dingxin_fold_pretraining_smoke,
)
from chronaris.evaluation.application_tasks.dingxin_pretraining_aggregate import (
    DingxinPretrainingAggregateConfig,
    DingxinPretrainingAggregateResult,
    run_dingxin_pretraining_aggregate,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_run import (
    DingxinConsumerSmokeConfig,
    DingxinConsumerSmokeResult,
    run_dingxin_consumer_smoke,
)
from chronaris.evaluation.application_tasks.dingxin_nested_target_run import (
    DingxinNestedTargetConfig,
    DingxinNestedTargetResult,
    run_dingxin_nested_targets,
)
from chronaris.evaluation.application_tasks.dingxin_nested_consumer_run import (
    DingxinNestedConsumerConfig,
    DingxinNestedConsumerResult,
    run_dingxin_nested_validation_consumers,
)
from chronaris.evaluation.application_tasks.snapshot_live_source import (
    InfluxSnapshotPointSource,
    SnapshotPointSource,
)
from chronaris.evaluation.application_tasks.simulation_audit import (
    SimulationAuditConfig,
    SimulationAuditResult,
    audit_existing_simulation,
    run_simulation_audit,
)
from chronaris.evaluation.application_tasks.representation_contract_smoke import (
    RepresentationContractSmokeConfig,
    RepresentationContractSmokeResult,
    run_representation_contract_smoke,
)
from chronaris.evaluation.application_tasks.shallow_adapter_smoke import (
    ShallowAdapterSmokeConfig,
    ShallowAdapterSmokeResult,
    run_shallow_adapter_smoke,
)
from chronaris.evaluation.application_tasks.candidate_screen_run import (
    EncoderCandidateScreenRunConfig,
    EncoderCandidateScreenRunResult,
    run_encoder_candidate_screen,
)
from chronaris.evaluation.application_tasks.dingxin_selected_screen_run import (
    DingxinSelectedScreenConfig,
    DingxinSelectedScreenResult,
    run_dingxin_selected_screen,
)
from chronaris.evaluation.application_tasks.dingxin_selected_representation_run import (
    DingxinSelectedRepresentationConfig,
    DingxinSelectedRepresentationResult,
    run_dingxin_selected_representations,
)
from chronaris.evaluation.application_tasks.dingxin_locked_pretraining_run import (
    DingxinLockedPretrainingConfig,
    DingxinLockedPretrainingResult,
    run_dingxin_locked_pretraining,
)
from chronaris.evaluation.application_tasks.dingxin_locked_representation_run import (
    DingxinLockedRepresentationConfig,
    DingxinLockedRepresentationResult,
    run_dingxin_locked_representations,
)
from chronaris.evaluation.application_tasks.dingxin_locked_consumer_run import (
    DingxinLockedConsumerConfig,
    DingxinLockedConsumerResult,
    run_dingxin_locked_consumers,
)
from chronaris.evaluation.application_tasks.simulation_locked_pretraining_run import (
    SimulationLockedPretrainingConfig,
    SimulationLockedPretrainingResult,
    run_simulation_locked_pretraining,
)
from chronaris.evaluation.application_tasks.simulation_locked_representation_run import (
    SimulationLockedRepresentationConfig,
    SimulationLockedRepresentationResult,
    run_simulation_locked_representations,
)
from chronaris.evaluation.application_tasks.simulation_locked_consumer_run import (
    SimulationLockedConsumerConfig,
    SimulationLockedConsumerResult,
    run_simulation_locked_consumers,
)
from chronaris.evaluation.application_tasks.simulation_stress_run import (
    SimulationStressGenerationConfig,
    SimulationStressGenerationResult,
    run_simulation_stress_generation,
)
from chronaris.evaluation.application_tasks.simulation_stress_representation_run import (
    SimulationStressRepresentationConfig,
    SimulationStressRepresentationResult,
    run_simulation_stress_representations,
)
from chronaris.evaluation.application_tasks.simulation_stress_consumer_run import (
    SimulationStressConsumerConfig,
    SimulationStressConsumerResult,
    run_simulation_stress_consumers,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_pretraining_run import (
    CHRONARIS_ABLATION_VARIANTS,
    SimulationChronarisAblationPretrainingConfig,
    SimulationChronarisAblationPretrainingResult,
    run_simulation_chronaris_ablation_pretraining,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_representation_run import (
    SimulationChronarisAblationRepresentationConfig,
    SimulationChronarisAblationRepresentationResult,
    run_simulation_chronaris_ablation_representations,
)
from chronaris.evaluation.application_tasks.simulation_chronaris_ablation_consumer_run import (
    SimulationChronarisAblationConsumerConfig,
    SimulationChronarisAblationConsumerResult,
    run_simulation_chronaris_ablation_consumers,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_representation_run import (
    MECHANISM_METHODS,
    SimulationMechanismRepresentationConfig,
    SimulationMechanismRepresentationResult,
    run_simulation_mechanism_representations,
)
from chronaris.evaluation.application_tasks.simulation_mechanism_consumer_run import (
    SimulationMechanismConsumerConfig,
    SimulationMechanismConsumerResult,
    run_simulation_mechanism_consumers,
)
from chronaris.evaluation.application_tasks.simulation_finetuning_run import (
    SimulationFineTuningConfig,
    SimulationFineTuningResult,
    run_simulation_end_to_end_finetuning,
)

__all__ = [
    "FixedDataAuditConfig",
    "EncoderCandidateScreenRunConfig",
    "EncoderCandidateScreenRunResult",
    "DingxinSelectedScreenConfig",
    "DingxinSelectedScreenResult",
    "DingxinSelectedRepresentationConfig",
    "DingxinSelectedRepresentationResult",
    "DingxinLockedPretrainingConfig",
    "DingxinLockedPretrainingResult",
    "DingxinLockedRepresentationConfig",
    "DingxinLockedRepresentationResult",
    "DingxinLockedConsumerConfig",
    "DingxinLockedConsumerResult",
    "SimulationLockedPretrainingConfig",
    "SimulationLockedPretrainingResult",
    "SimulationLockedRepresentationConfig",
    "SimulationLockedRepresentationResult",
    "SimulationLockedConsumerConfig",
    "SimulationLockedConsumerResult",
    "SimulationStressGenerationConfig",
    "SimulationStressGenerationResult",
    "SimulationStressRepresentationConfig",
    "SimulationStressRepresentationResult",
    "SimulationStressConsumerConfig",
    "SimulationStressConsumerResult",
    "CHRONARIS_ABLATION_VARIANTS",
    "SimulationChronarisAblationPretrainingConfig",
    "SimulationChronarisAblationPretrainingResult",
    "SimulationChronarisAblationRepresentationConfig",
    "SimulationChronarisAblationRepresentationResult",
    "SimulationChronarisAblationConsumerConfig",
    "SimulationChronarisAblationConsumerResult",
    "MECHANISM_METHODS",
    "SimulationMechanismRepresentationConfig",
    "SimulationMechanismRepresentationResult",
    "SimulationMechanismConsumerConfig",
    "SimulationMechanismConsumerResult",
    "SimulationFineTuningConfig",
    "SimulationFineTuningResult",
    "FixedDataAuditResult",
    "FixedDataSnapshotConfig",
    "ChronarisContinuousAdapterSmokeConfig",
    "ChronarisContinuousAdapterSmokeResult",
    "CommonPretrainingLoopSmokeConfig",
    "CommonPretrainingLoopSmokeResult",
    "ApplicationConsumerSmokeConfig",
    "ApplicationConsumerSmokeResult",
    "ApplicationFrozenEvaluationResult",
    "DingxinTargetArchiveConfig",
    "DingxinTargetArchiveResult",
    "DingxinContextBindingConfig",
    "DingxinContextBindingResult",
    "DingxinInnerSplitConfig",
    "DingxinInnerSplitResult",
    "DingxinFoldPretrainingConfig",
    "DingxinFoldPretrainingResult",
    "DingxinPretrainingAggregateConfig",
    "DingxinPretrainingAggregateResult",
    "DingxinConsumerSmokeConfig",
    "DingxinConsumerSmokeResult",
    "DingxinNestedTargetConfig",
    "DingxinNestedTargetResult",
    "DingxinNestedConsumerConfig",
    "DingxinNestedConsumerResult",
    "DeepBaselineAdapterSmokeConfig",
    "DeepBaselineAdapterSmokeResult",
    "InfluxSnapshotPointSource",
    "SnapshotPointSource",
    "SimulationAuditConfig",
    "SimulationAuditResult",
    "RepresentationContractSmokeConfig",
    "RepresentationContractSmokeResult",
    "ShallowAdapterSmokeConfig",
    "ShallowAdapterSmokeResult",
    "audit_existing_simulation",
    "DEFAULT_SNAPSHOT_RUN_ID",
    "run_fixed_data_audit",
    "run_encoder_candidate_screen",
    "run_dingxin_selected_screen",
    "run_dingxin_selected_representations",
    "run_dingxin_locked_pretraining",
    "run_dingxin_locked_representations",
    "run_dingxin_locked_consumers",
    "run_simulation_locked_pretraining",
    "run_simulation_locked_representations",
    "run_simulation_locked_consumers",
    "run_simulation_stress_generation",
    "run_simulation_stress_representations",
    "run_simulation_stress_consumers",
    "run_simulation_chronaris_ablation_pretraining",
    "run_simulation_chronaris_ablation_representations",
    "run_simulation_chronaris_ablation_consumers",
    "run_simulation_mechanism_representations",
    "run_simulation_mechanism_consumers",
    "run_simulation_end_to_end_finetuning",
    "run_fixed_data_snapshot",
    "run_chronaris_continuous_adapter_smoke",
    "run_common_pretraining_loop_smoke",
    "run_application_consumer_smoke",
    "evaluate_frozen_application_consumers",
    "run_dingxin_target_archive",
    "run_dingxin_context_binding_audit",
    "run_dingxin_inner_split",
    "run_dingxin_fold_pretraining_smoke",
    "run_dingxin_pretraining_aggregate",
    "run_dingxin_consumer_smoke",
    "run_dingxin_nested_targets",
    "run_dingxin_nested_validation_consumers",
    "run_deep_baseline_adapter_smoke",
    "run_simulation_audit",
    "run_representation_contract_smoke",
    "run_shallow_adapter_smoke",
]
