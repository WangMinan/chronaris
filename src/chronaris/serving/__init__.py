"""Serving and near-real-time inference interfaces."""

from chronaris.serving.runtime_demo import (
    StageIRuntimeDemoConfig,
    StageIRuntimeDemoRunResult,
    render_stage_i_runtime_demo_report,
    run_stage_i_runtime_demo,
)
from chronaris.serving.runtime_inference import (
    StageIRuntimeInferenceConfig,
    StageIRuntimeInferenceRunResult,
    dump_runtime_predictions_jsonl,
    dump_runtime_samples_jsonl,
    load_runtime_samples_jsonl,
    render_stage_i_runtime_inference_report,
    run_stage_i_runtime_inference_batch,
    run_stage_i_runtime_inference_incremental,
    run_stage_i_runtime_inference,
)
from chronaris.serving.runtime_service_smoke import (
    StageIRuntimeSmokeConfig,
    StageIRuntimeSmokeRunResult,
    render_stage_i_runtime_smoke_report,
    run_stage_i_runtime_smoke,
)
from chronaris.serving.runtime_schema_contract import (
    RuntimeSchemaContractRunResult,
    build_runtime_schema_contract,
    canonicalize_runtime_samples,
)

__all__ = [
    "StageIRuntimeDemoConfig",
    "StageIRuntimeDemoRunResult",
    "StageIRuntimeInferenceConfig",
    "StageIRuntimeInferenceRunResult",
    "StageIRuntimeSmokeConfig",
    "StageIRuntimeSmokeRunResult",
    "RuntimeSchemaContractRunResult",
    "build_runtime_schema_contract",
    "canonicalize_runtime_samples",
    "dump_runtime_predictions_jsonl",
    "dump_runtime_samples_jsonl",
    "load_runtime_samples_jsonl",
    "render_stage_i_runtime_demo_report",
    "render_stage_i_runtime_inference_report",
    "render_stage_i_runtime_smoke_report",
    "run_stage_i_runtime_demo",
    "run_stage_i_runtime_inference_batch",
    "run_stage_i_runtime_inference_incremental",
    "run_stage_i_runtime_inference",
    "run_stage_i_runtime_smoke",
]
