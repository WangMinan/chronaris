# 阶段 1：导出一致性、失败终态与已有计算复用

本轮按用户授权修复现有导出与队列问题，并核验已有计算的保留范围。来源为已读取的任务 `01a084c8-627f-7fe2-a726-5a0482e85bfc` 与[当前方案](../../../../requirements/thesis-downstream-representation-plan-20260909.md)。

## 计划与判据

1. 统一自监督、任务引导、压力与正式导出的有效掩码池化；训练计算保持原实现。
2. 用真实失败的 256 个开发窗口重放，沿用绝对容差 `1e-6`、相对容差零；覆盖部分有效和全无观测情况。
3. 初筛压力异常同步记录失败单元、原因、尝试和终态；显式恢复重试失败单元，保留成功单元及失败历史。
4. 核验原 52 份所选检查点、表示与下游模型散列，分别判定原样复用、重新汇总与重新拟合范围。
5. 新证据写入独立目录，原失败目录与冻结工作树保持原位；本阶段不启动完整长队列。

## 执行记录

共享导出及初筛压力终态修复已完成，真实失败单元完成八条件显卡重放。52 份开发检查点和 104 份序列表示可保留；26 组自监督线性模型需在新汇总上重新拟合。完整结论和逐类证据见[阶段报告](../../../../artifacts/runs/2026-09-10_v4-stage1-repair/report.md)。

代码落点为 `representation/contracts.py` 的共享导出池化、`modeling/training/common_pretraining.py` 的已训练模型适配器、`application_finetuning_export.py` 的任务引导导出，以及 `v4_pipeline_steps.py`、`v4_pipeline.py` 的失败与恢复路径。新增 `v4_export_reuse.py` 仅做本次源码差异和旧证据核验，未增加训练系统或放开旧目录恢复。

## 可复现命令

在仓库根目录执行以下只读核验。输出目录必须与原运行和冻结工作树分开；文件中的来源散列可用于再次检查旧证据未变。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/audit_v4_export_reuse.py \
  --run-root /mnt/e/chronaris-v4-results/2026-09-08-v4-complete \
  --frozen-project /home/wangminan/projects/chronaris-runtime-v4-code-complete-20260908 \
  --output-root /mnt/e/chronaris-v4-results/2026-09-10-stage1-repair/audit-verified
```

真实失败重放继续调用现有压力入口，只运行这一条已失败的路线。以下命令不会拟合或更新模型。

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python - <<'PY'
from chronaris.evaluation.application_tasks.v4_pressure_run import run_development_pressure
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock

with development_gpu_lock() as acquired:
    assert acquired, 'GPU lock occupied'
    result = run_development_pressure(
        method='chronaris', candidate_name='independent_pairing', route='task_guided', update=200,
        output_root='/mnt/e/chronaris-v4-results/2026-09-10-stage1-repair/pressure_verified',
        diagnostic_root='/mnt/e/chronaris-v4-results/2026-09-08-v4-complete/initial',
        condition_root='artifacts/application_evaluation/2026-09-06_v4-development-conditions-repair',
        device='cuda')
    assert result['completed'] and len(result['conditions']) == 8
PY
```

上述路径已有完成收据时会执行已有入口的散列核验并复用结果。需要重新执行神经编码时换用新输出目录；源码变化后同样必须更换目录，不能删除旧收据绕过检查。

定向回归入口为 `tests/evaluation/application_tasks/test_export_pooling.py` 和 `test_v4_pipeline.py`。全量 CUDA（统一计算设备架构）验收使用已有 `v4_configuration_freeze.validate_current_cuda`，保存完整测试日志、XML 和源码绑定收据。

最终完整 CUDA 验收：654 项通过、8 项跳过，337.04 秒。变更文件的未定义名称、编译、差异空白、文档链接与术语检查通过。原 1,106 份文件最终散列不变；未在原失败目录续跑，未执行新的编码器训练。
