"""Fixed-data application evaluation dataset contracts and builders."""

from chronaris.dataset.application_evaluation.contexts import (
    build_application_contexts,
    contexts_to_frame,
)
from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    FieldRoleRecord,
    OuterFoldDefinition,
    stable_sample_hash,
)
from chronaris.dataset.application_evaluation.field_roles import (
    build_field_role_manifest,
    selected_maneuver_roles,
    selected_response_roles,
)
from chronaris.dataset.application_evaluation.labels import (
    LabelConstructionError,
    build_fold_task_labels,
)
from chronaris.dataset.application_evaluation.splits import build_outer_folds
from chronaris.dataset.application_evaluation.snapshot_contracts import (
    FIXED_DINGXIN_SORTIE_IDS,
    RAW_POINT_FORMAT,
    SnapshotFileRecord,
    SnapshotRunResult,
    SnapshotViewPlan,
)
from chronaris.dataset.application_evaluation.snapshot_io import (
    iter_raw_point_snapshot,
    sha256_file,
    write_raw_point_snapshot,
)

__all__ = [
    "ApplicationContextRecord",
    "FieldRoleRecord",
    "FIXED_DINGXIN_SORTIE_IDS",
    "LabelConstructionError",
    "OuterFoldDefinition",
    "RAW_POINT_FORMAT",
    "SnapshotFileRecord",
    "SnapshotRunResult",
    "SnapshotViewPlan",
    "build_application_contexts",
    "build_field_role_manifest",
    "build_fold_task_labels",
    "build_outer_folds",
    "contexts_to_frame",
    "selected_maneuver_roles",
    "selected_response_roles",
    "iter_raw_point_snapshot",
    "sha256_file",
    "stable_sample_hash",
    "write_raw_point_snapshot",
]
