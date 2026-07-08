"""Specs for controlled Chronaris candidate optimization."""

from __future__ import annotations

from dataclasses import dataclass

from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import (
    DEFAULT_E_MANIFEST,
    DEFAULT_F_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
)

OLD_E3_REVIEW_ROOT = "docs/artifacts/runs/2026-07-08_e3-result-review"
OLD_FOUR_METHOD_ROOT = "docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation"
OLD_BASELINE_INPUT = f"{OLD_FOUR_METHOD_ROOT}/combined_four_method_e3_input_long.csv"
OLD_CONSISTENCY_TABLE = f"{OLD_E3_REVIEW_ROOT}/t1_t2_e3_consistency_table.csv"
OLD_E3_METHOD_SUMMARY = f"{OLD_E3_REVIEW_ROOT}/e3_method_metric_summary.csv"
DEEP_BASELINE_ROOT = "docs/artifacts/runs/2026-07-07_deep-baseline-representation-export"
THESIS_PROTOCOL_ROOT = "docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot"

DEV_RUN_ID = "2026-07-08_chronaris-controlled-optimization-dev"
CONFIRM_RUN_ID = "2026-07-08_chronaris-controlled-optimization-confirm"
REPRESENTATION_RUN_ID = "2026-07-08_chronaris-oof-representation-export"
REPRESENTATION_FAMILY = "Chronaris_T2_response_lovo_seed17_pooled_embedding"

E3_TARGET_METRICS = (
    "event_alignment_score",
    "discord_maneuver_overlap",
    "motif_event_consistency",
    "cross_view_segment_stability",
    "fragment_replay_recall",
    "nn_segment_cross_view_consistency",
)
LOWER_IS_BETTER = {"rmse", "mae", "nrmse"}


@dataclass(frozen=True, slots=True)
class ChronarisCandidateSpec:
    candidate_id: str
    model_name: str
    hidden_dim: int
    dropout: float
    weight_decay: float
    learning_rate: float
    layers: int = 2
    num_heads: int = 4
    representation_postprocess: str = "pooled"
    changed_knobs: str = ""
    expected_t2_effect: str = ""
    expected_e3_effect: str = ""
    risk: str = ""
    changes_representation_family: bool = True


@dataclass(frozen=True, slots=True)
class ControlledOptimizationConfig:
    dev_run_id: str = DEV_RUN_ID
    confirm_run_id: str = CONFIRM_RUN_ID
    representation_run_id: str = REPRESENTATION_RUN_ID
    output_root: str = DEFAULT_OUTPUT_ROOT
    e_run_manifest_path: str = DEFAULT_E_MANIFEST
    f_run_manifest_path: str = DEFAULT_F_MANIFEST
    old_baseline_input_path: str = OLD_BASELINE_INPUT
    old_consistency_table_path: str = OLD_CONSISTENCY_TABLE
    old_e3_method_summary_path: str = OLD_E3_METHOD_SUMMARY
    deep_baseline_root: str = DEEP_BASELINE_ROOT
    thesis_protocol_root: str = THESIS_PROTOCOL_ROOT
    seed: int = 17
    smoke_epochs: int = 1
    dev_epochs: int = 2
    confirm_epochs: int = 5
    smoke_max_folds: int = 1
    dev_max_folds: int | None = None
    device: str = "auto"
    require_cuda: bool = True
    tensor_cache: str = "auto"
    amp: str = "bf16"
    batch_size: int = 128
    eval_batch_size: int | None = None
    max_cache_gb: float = 18.0
    pin_memory: bool = True
    non_blocking_copy: bool = True
    auto_batch_size: bool = True
    min_T: int = 30
    run_confirmation: bool = True
    max_candidates: int | None = None


def default_candidate_specs() -> tuple[ChronarisCandidateSpec, ...]:
    """Return the fixed 8-candidate development registry."""

    return (
        ChronarisCandidateSpec(
            candidate_id="chr_v2_residual_h64",
            model_name="chronaris_v2_task_heads",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="pooled",
            changed_knobs="v2 residual T2 head; hidden=64; dropout=0.10; weight_decay=1e-5",
            expected_t2_effect="Residual physiology response head should reduce old feature-only T2 error.",
            expected_e3_effect="Causal fused pooled embedding may reduce representation fragmentation.",
            risk="Can still overfit the weak T2 target on the small leave-one-view-out sample.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v2_residual_norm_h64",
            model_name="chronaris_v2_task_heads",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="l2_normalized",
            changed_knobs="v2 residual T2 head; row-wise L2 normalized pooled embedding",
            expected_t2_effect="Same supervised T2 head as v2 residual; normalization affects only exported representation.",
            expected_e3_effect="Normalization may make STUMPY/ClaSP less dominated by embedding magnitude.",
            risk="T2 direct metrics unchanged while E3 may lose amplitude cues.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v2_residual_delta_h64",
            model_name="chronaris_v2_task_heads",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="temporal_delta_append",
            changed_knobs="v2 residual T2 head; append adjacent held-out embedding deltas",
            expected_t2_effect="Same supervised T2 head as v2 residual; temporal delta affects only exported representation.",
            expected_e3_effect="Delta summary may expose event boundaries and discord intervals.",
            risk="Higher dimension can amplify noise on 36-window streams.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v2_residual_h96_wd1e4",
            model_name="chronaris_v2_task_heads",
            hidden_dim=96,
            dropout=0.10,
            weight_decay=1e-4,
            learning_rate=8e-4,
            representation_postprocess="pooled",
            changed_knobs="v2 residual T2 head; hidden=96; weight_decay=1e-4; lr=8e-4",
            expected_t2_effect="More capacity with stronger decay may improve response fit without collapse.",
            expected_e3_effect="Smoother larger hidden states may give more separable stream structure.",
            risk="Higher capacity can still be unstable on small folds.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v2_residual_h64_do0p05",
            model_name="chronaris_v2_task_heads",
            hidden_dim=64,
            dropout=0.05,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="pooled",
            changed_knobs="v2 residual T2 head; lower dropout=0.05",
            expected_t2_effect="Lower dropout may improve underfit T2 residual fit.",
            expected_e3_effect="Less noisy pooled embedding may improve motif consistency.",
            risk="Lower regularization may degrade held-out fold stability.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v2_noresidual_h64",
            model_name="v2_no_residual_t2",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="pooled",
            changed_knobs="v2 causal fusion with plain pooled regression head; no residual T2 decomposition",
            expected_t2_effect="Tests whether the residual head is hurting weak T2 regression.",
            expected_e3_effect="Plain pooled head may preserve cleaner fused geometry.",
            risk="Removing persistence/excitation decomposition may undercut the thesis-facing T2 design.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v3_stream_role_h64",
            model_name="chronaris_v3_stream_role",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="pooled",
            changed_knobs="v3 role-aware private stream routing; hidden=64",
            expected_t2_effect="Explicit real-vehicle stream role may improve contribution balance.",
            expected_e3_effect="Role-aware fused states may improve event alignment under fixed E3.",
            risk="Role router may collapse to a similar causal path on private streams.",
        ),
        ChronarisCandidateSpec(
            candidate_id="chr_v3_fixed_causal_norm_h64",
            model_name="v3_fixed_causal_lag",
            hidden_dim=64,
            dropout=0.10,
            weight_decay=1e-5,
            learning_rate=1e-3,
            representation_postprocess="l2_normalized",
            changed_knobs="v3 fixed causal lag route; row-wise L2 normalized pooled embedding",
            expected_t2_effect="Fixed causal route keeps vehicle contribution explicit while training T2.",
            expected_e3_effect="Normalization plus causal route may improve cross-view structural comparability.",
            risk="Fixed route may be too rigid for physiology response regression.",
        ),
    )


__all__ = [
    "CONFIRM_RUN_ID",
    "ControlledOptimizationConfig",
    "ChronarisCandidateSpec",
    "DEV_RUN_ID",
    "E3_TARGET_METRICS",
    "LOWER_IS_BETTER",
    "OLD_E3_REVIEW_ROOT",
    "OLD_FOUR_METHOD_ROOT",
    "REPRESENTATION_FAMILY",
    "REPRESENTATION_RUN_ID",
    "default_candidate_specs",
]
