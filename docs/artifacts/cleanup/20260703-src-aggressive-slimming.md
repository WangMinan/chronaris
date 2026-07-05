# 2026-07-03 src aggressive slimming cleanup

## 范围

本次是 仓库收敛清理 之后的追加 `src/` 激进瘦身，只处理过度设计、过度封装和无活跃依赖的早期薄层。未启动 仿真压力测试/检索任务与公开路线优化/论文级消融统一/论文材料化 实验，未重跑或改写 公开模型对比与公开融合刷新/鼎新真实数据第三方模型对比、公开融合消融、跨证据矩阵、任务感知头优化、流角色融合、优化模型再评估与最终指标打磨/论文协议冻结 confirmed metrics。

## 删除与合并

- 删除 `src/chronaris/pipelines/dataset_v1.py`：该类只包装 `SortieLoader + SortieDatasetBuilder`，没有活跃脚本入口。
- 删除 `src/chronaris/access/contracts.py`：三个 Protocol 并入 `src/chronaris/access/loader.py`，避免单独维护接口文件。
- 删除 `src/chronaris/pipelines/stage_i/common/deep_role_aware.py` 与 `src/chronaris/pipelines/stage_i/common/deep_task_aware.py`：v2/v3 wrapper 并入 `deep_models.py`，复用已有 forward result、mask pooling 与 time-feature helper。
- 删除 `src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py`：最终指标打磨 nested-root/status helper 并回唯一调用方。
- 删除 `src/chronaris/pipelines/stage_i/llm/comparison_io.py`：LLM 预处理对比 JSON/CSV helper 并回唯一调用方。
- 压薄 barrel/compat 层：`chronaris.pipelines`、`chronaris.pipelines.stage_i`、`chronaris.models.alignment`、`chronaris.models.fusion`、`chronaris.serving` 不再维护大规模 re-export / meta-path 兼容表；活跃代码改为从真实实现模块直接导入。

## 规模变化

- `src/chronaris` Python 文件：`193 -> 187`。
- `src/chronaris` Python 行数：`65,319 -> 64,211`。
- `src/scripts/tests` Python 文件：`282 -> 276`。
- `src/scripts/tests` Python 行数：`86,323 -> 85,162`。
- 当前 Python 代码规模净删：`6` 个文件、`1,161` 行；提交级 diff 规模以 `git diff --stat` 为准，文档回写会增加少量记录行。

## 验证

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall -q src scripts tests`：通过。
- CLI `--help` smoke：`training/train_multitask.py`、`public/run_opt.py`、`private/run_benchmark.py`、`runtime/run_inference.py`、`evidence/build_support.py`、`evidence/build_midterm_evidence.py` 均通过。
- focused tests：
  - `tests.test_alignment_data tests.test_alignment_pipeline`：`14` tests OK，`4` skipped。
  - `tests.test_runtime_inference tests.test_runtime_service_smoke tests.test_stage_i_case_study`：`13` tests OK。
  - `tests.test_stream_role_fusion tests.test_stage_i_deep_pipeline tests.test_stage_i_llm_preprocessing tests.test_stage_i_llm_comparison`：`31` tests OK，`2` skipped。
  - `tests.test_stage_i_multitask_train tests.test_stage_i_support tests.test_stage_i_private_optimization tests.test_stage_i_public_opt tests.test_stage_i_public_opt_aggregation`：`37` tests OK。
- full discover：`PYTHONDONTWRITEBYTECODE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover tests`，`221` tests OK，`8` skipped。
- `git diff --check`：通过。
- `git lfs status`：仅显示本次未暂存代码/文档改动，无 LFS 异常。

## 后续边界

后续代码应继续直接导入真实模块路径，不再新增包级大 re-export 表或历史 `stage_i_*` import hook。若需要恢复历史 notebook，可在 notebook 中迁移 import，而不是把兼容层重新放回 `src`。
