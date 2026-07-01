"""Stage I pipeline subpackage.

The concrete Stage I implementation is grouped by responsibility:

- ``common``: shared helpers used by several evidence/public/private paths.
- ``training``: reusable backbone and multitask training pipelines.
- ``public``: public benchmark/adapter workflows.
- ``private``: private proxy benchmark workflows.
- ``evidence``: thesis evidence, support, audits, and midterm packaging.
- ``llm``: optional LLM preprocessing context generation.
- ``legacy``: historical public Stage I phase/baseline closures.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from importlib import import_module

__all__: list[str] = []

_COMPAT_MODULE_ALIASES = {
    f"{__name__}.stage_i_anchor": f"{__name__}.evidence.anchors",
    f"{__name__}.stage_i_backbone_train": f"{__name__}.training.backbone_train",
    f"{__name__}.stage_i_baseline": f"{__name__}.legacy.baseline",
    f"{__name__}.stage_i_baseline_models": f"{__name__}.common.baseline_models",
    f"{__name__}.stage_i_baseline_reporting": f"{__name__}.legacy.baseline_reporting",
    f"{__name__}.stage_i_case_study": f"{__name__}.evidence.case_study",
    f"{__name__}.stage_i_deep_baseline": f"{__name__}.public.deep_baseline",
    f"{__name__}.stage_i_deep_baseline_case": f"{__name__}.public.deep_baseline_case",
    f"{__name__}.stage_i_deep_baseline_reporting": f"{__name__}.public.deep_baseline_reporting",
    f"{__name__}.stage_i_deep_baseline_runtime": f"{__name__}.public.deep_baseline_runtime",
    f"{__name__}.stage_i_deep_models": f"{__name__}.common.deep_models",
    f"{__name__}.stage_i_evidence_runner": f"{__name__}.evidence.closure_runner",
    f"{__name__}.stage_i_llm_preprocessing": f"{__name__}.llm.preprocessing",
    f"{__name__}.stage_i_llm_preprocessing_harness": f"{__name__}.llm.harness",
    f"{__name__}.stage_i_llm_preprocessing_reporting": f"{__name__}.llm.reporting",
    f"{__name__}.stage_i_llm_preprocessing_slicing": f"{__name__}.llm.slicing",
    f"{__name__}.stage_i_midterm_evidence": f"{__name__}.evidence.midterm_pack",
    f"{__name__}.stage_i_multitask_sweep": f"{__name__}.evidence.weak_label_sweep",
    f"{__name__}.stage_i_multitask_train": f"{__name__}.training.multitask_train",
    f"{__name__}.stage_i_phase3": f"{__name__}.legacy.phase3",
    f"{__name__}.stage_i_phase3_assets": f"{__name__}.legacy.phase3_assets",
    f"{__name__}.stage_i_phase3_reporting": f"{__name__}.legacy.phase3_reporting",
    f"{__name__}.stage_i_private_benchmark": f"{__name__}.private.benchmark",
    f"{__name__}.stage_i_private_benchmark_data": f"{__name__}.private.benchmark_data",
    f"{__name__}.stage_i_private_benchmark_models": f"{__name__}.private.benchmark_models",
    f"{__name__}.stage_i_private_component_ablation": f"{__name__}.evidence.private_component_ablation",
    f"{__name__}.stage_i_private_feature_utils": f"{__name__}.private.feature_utils",
    f"{__name__}.stage_i_private_leakage_audit": f"{__name__}.private.leakage_audit",
    f"{__name__}.stage_i_private_leakage_safe_ablation": f"{__name__}.private.leakage_safe_ablation",
    f"{__name__}.stage_i_private_optimization": f"{__name__}.private.optimization",
    f"{__name__}.stage_i_private_optimized_package": f"{__name__}.private.optimized_package",
    f"{__name__}.stage_i_public_adapter_calibration": f"{__name__}.evidence.public_adapter_calibration",
    f"{__name__}.stage_i_public_fusion_refresh": f"{__name__}.public.fusion_refresh",
    f"{__name__}.stage_i_public_fusion_screen": f"{__name__}.public.fusion_screen",
    f"{__name__}.stage_i_public_mainline_report": f"{__name__}.public.mainline_report",
    f"{__name__}.stage_i_public_model_comparison": f"{__name__}.public.model_comparison",
    f"{__name__}.stage_i_public_opt": f"{__name__}.public.opt",
    f"{__name__}.stage_i_public_opt_data": f"{__name__}.public.opt_data",
    f"{__name__}.stage_i_public_opt_postprocess": f"{__name__}.public.opt_postprocess",
    f"{__name__}.stage_i_public_opt_reference": f"{__name__}.public.opt_reference",
    f"{__name__}.stage_i_public_opt_reporting": f"{__name__}.public.opt_reporting",
    f"{__name__}.stage_i_public_opt_shared": f"{__name__}.public.opt_shared",
    f"{__name__}.stage_i_public_opt_sklearn": f"{__name__}.public.opt_sklearn",
    f"{__name__}.stage_i_public_opt_torch": f"{__name__}.public.opt_torch",
    f"{__name__}.stage_i_public_opt_torch_catalog": f"{__name__}.public.opt_torch_catalog",
    f"{__name__}.stage_i_public_opt_torch_reporting": f"{__name__}.public.opt_torch_reporting",
    f"{__name__}.stage_i_public_opt_torch_supervision": f"{__name__}.public.opt_torch_supervision",
    f"{__name__}.stage_i_public_transfer_boundary": f"{__name__}.evidence.public_transfer_boundary",
    f"{__name__}.stage_i_rigid_body_rotation_audit": f"{__name__}.evidence.rigid_body_rotation_audit",
    f"{__name__}.stage_i_run_observer": f"{__name__}.common.run_observer",
    f"{__name__}.stage_i_sequence_preparation": f"{__name__}.public.sequence_preparation",
    f"{__name__}.stage_i_support": f"{__name__}.evidence.support",
    f"{__name__}.stage_i_support_builders": f"{__name__}.evidence.support_builders",
    f"{__name__}.stage_i_support_reporting": f"{__name__}.evidence.support_reporting",
    f"{__name__}.stage_i_thesis_materials": f"{__name__}.evidence.thesis_materials",
}


class _CompatStageIModuleLoader(importlib.abc.Loader):
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        target_module = import_module(_COMPAT_MODULE_ALIASES[self.fullname])
        sys.modules[self.fullname] = target_module
        return target_module

    def exec_module(self, module) -> None:  # type: ignore[no-untyped-def]
        return None


class _CompatStageIModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[no-untyped-def]
        if fullname not in _COMPAT_MODULE_ALIASES:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _CompatStageIModuleLoader(fullname),
            origin=_COMPAT_MODULE_ALIASES[fullname],
        )


if not any(isinstance(finder, _CompatStageIModuleFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _CompatStageIModuleFinder())
