"""Run native full-method and measured-case acceptance using existing completed models."""
from run_fusion_workbench import PROJECT

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import torch
    from chronaris.evaluation.application_tasks.chapter6_full_assets import prepare_full_cases
    from chronaris.evaluation.application_tasks.chapter6_physics_audit import audit_full_physics
    from chronaris.evaluation.application_tasks.chapter6_field_audit import audit_observed_fields
    from chronaris.evaluation.application_tasks.chapter6_acceptance import AcceptanceService
    from chronaris.evaluation.application_tasks.chapter6_gpu_acceptance import (
        run_gpu_acceptance, capture_gpu_cases, verify_gpu_recovery)
    parser=argparse.ArgumentParser(description='完整方法与GPU服务验收，不执行优化器更新')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();evidence=args.evidence.resolve()
    if not root.is_relative_to(PROJECT/'artifacts') or not evidence.is_relative_to(PROJECT):
        parser.error('验收输出须位于本工作区，运行资产须位于artifacts')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    prepare_full_cases(root)
    audit_full_physics(evidence);audit_observed_fields(evidence)
    service=AcceptanceService(PROJECT,root,device='cuda').start()
    try:
        run_gpu_acceptance(PROJECT,root,evidence,service.port)
        capture_gpu_cases(PROJECT,root,evidence,service.base)
        verify_gpu_recovery(PROJECT,root,evidence,service.port)
    finally:service.stop()
    print('完整方法与GPU验收通过：',evidence)
