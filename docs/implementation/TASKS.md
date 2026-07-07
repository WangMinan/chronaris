# Chronaris 当前任务

更新时间：2026-07-07

## 当前任务

命名迁移已完成（见 `docs/maintenance/2026-07-05_name-migration-map.md`）。本轮（2026-07-07）已执行 **fusion stream structure evaluation（E3 融合表示流结构评价）** 第一批编码、第二轮 evaluator validation 和第三方来源审计：新增独立评价层、CLI、测试、no-training dry run 与 MulT / ContiFormer source audit，不训练、不改 confirmed metrics。

- 计划入口：`docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/`（report / input_contract / metric_contract / implementation_plan / acceptance_checklist / evidence_manifest）。
- 执行入口：`src/chronaris/evaluation/fusion_stream_structure/` 与 `scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py`。
- 当前 validation 产物入口：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-evaluator-validation/` 与 `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-evaluator-validation/`。
- 第三方来源审计入口：`docs/artifacts/runs/2026-07-07_fusion-stream-thirdparty-source-audit/`，结论为 `C. no_reusable_sources`。
- 历史首轮 fallback 产物入口：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-execution/` 与 `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-dry-run/`。
- 当前环境已安装并验证 `claspy 0.2.8` / `stumpy 1.14.1`；ClaSP 与 STUMPY 真实 evaluator 已启用，CLaP 在短序列上保留结构化 `clap_unavailable`。

验收关注点（本轮 execution）：

- E3 long 表固定 `evidence_quadrant = fusion_stream_structure`，不并入既有 leaderboard。
- 小规模 Dingxin dry run 只使用既有 feature export 和可复用 `feature_values`；source audit 确认 `mult` / `contiformer` 当前无可复用融合表示流或可加载 checkpoint，已写 `method_unavailable`，未训练补齐。
- 未改 confirmed metrics、未回写 `result_matrix_long.csv` / `experiment_registry.csv` / `claim_boundary_table.csv`。
- 相关测试、`compileall`、`git diff --check` 需在本轮提交前保持通过。

## 当前代码结构

- `src/chronaris/feature_export/`：标准化融合特征导出、导出 manifest 读取和相关 profile。
- `src/chronaris/modeling/`：公共建模组件、GPU runtime helper、backbone 与 multitask training。
- `src/chronaris/evaluation/dingxin/`：鼎新真实数据弱监督任务、组件消融、第三方模型对比和任务头校准。
- `src/chronaris/evaluation/fusion_stream_structure/`：E3 融合表示流结构评价，包含合同、Dingxin feature frame 重组、预处理、ClaSP/CLaP gated wrapper、STUMPY Matrix Profile wrapper、结构指标和报告写出。
- `src/chronaris/evaluation/public_datasets/`：公开 UAB/NASA 数据适配、公开模型对比、公开融合校准和公开消融。
- `src/chronaris/evidence/`：证据矩阵、指标校准、论文协议快照、图表材料、support、rotation audit 和 evidence runner。
- `src/chronaris/runtime/` 与 `src/chronaris/serving/`：运行时服务、schema contract 和 replay。
- `src/chronaris/llm_preprocessing/`：LLM preprocessing、harness、slicing、comparison 和 reporting。
- `src/chronaris/archive/`：历史公开 benchmark 代码。

## 当前脚本入口

- `scripts/feature_export/run_export.py`
- `scripts/modeling/train_backbone.py`
- `scripts/modeling/train_multitask.py`
- `scripts/evaluation/dingxin/*.py`
- `scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py`
- `scripts/evaluation/public_datasets/*.py`
- `scripts/evidence/*.py`
- `scripts/runtime/*.py`
- `scripts/llm_preprocessing/*.py`
- `scripts/archive/legacy_public_benchmark/*.py`

运行 Python 默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## 当前产物入口

主要入口：

- E3 synthetic no-training 执行：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-execution/`
- E3 小规模 Dingxin dry run：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-dry-run/`
- E3 synthetic evaluator validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-evaluator-validation/`
- E3 小规模 Dingxin evaluator validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-evaluator-validation/`
- E3 第三方来源审计：`docs/artifacts/runs/2026-07-07_fusion-stream-thirdparty-source-audit/`
- E3 开发计划：`docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/`
- 论文协议快照：`docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/`
- 指标校准：`docs/artifacts/runs/2026-07-02_metric-calibration/`
- 选定模型汇总：`docs/artifacts/runs/2026-07-02_selected-model-summary/`
- 选定模型再评估：`docs/artifacts/runs/2026-07-02_selected-model-reevaluation/`
- 流角色融合：`docs/artifacts/runs/2026-07-02_stream-role-fusion/`
- 任务头校准：`docs/artifacts/runs/2026-07-02_task-head-calibration/`
- 跨证据矩阵：`docs/artifacts/runs/2026-07-02_cross-evidence-matrix/`
- 鼎新第三方对比：`docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison/`
- 公开融合消融：`docs/artifacts/runs/2026-07-02_public-fusion-ablation/`
- 公开模型对比：`docs/artifacts/runs/2026-07-01_public-model-comparison/`
- 公开融合校准：`docs/artifacts/runs/2026-07-01_public-fusion-calibration/`
- NASA 公开融合确认：`docs/artifacts/runs/2026-05-09_public-fusion-nasa-full-confirm/`
- 公开主线汇总：`docs/artifacts/runs/2026-05-08_public-mainline-uab-robust-prior-r1/`
- 公开优化输入：`docs/artifacts/runs/2026-05-08_public-opt-nasa-prepared-v2/`、`docs/artifacts/runs/2026-05-08_public-opt-uab-robust-prior-r1/`、`docs/artifacts/runs/2026-05-08_public-opt-uab-heat-specialist-r1/`
- semantic support baseline：`docs/artifacts/runs/2026-05-06_semantic-support-baseline/`
- 公开融合 screen：`docs/artifacts/runs/2026-05-06_public-fusion-screen-round2/`
- NASA 公开融合短确认：`docs/artifacts/runs/2026-05-06_public-fusion-nasa-confirm/`
- UAB 公开融合确认：`docs/artifacts/runs/2026-05-06_public-fusion-uab-confirm/`
- NASA 公开优化结果：`docs/artifacts/runs/2026-05-06_public-opt-nasa-round1/`
- 公开优化准备根：`docs/artifacts/runs/2026-05-06_public-opt-nasa-prepared/`、`docs/artifacts/runs/2026-05-06_public-opt-uab/`、`docs/artifacts/runs/2026-05-06_public-opt-uab-torch/`
- 鼎新优化包基线：`docs/artifacts/runs/2026-05-04_dingxin-opt-package/`
- deep baseline 准备根：`docs/artifacts/runs/2026-05-01_deep-real-sortie-prepared/`、`docs/artifacts/runs/2026-05-01_deep-comparison-prepared/`
- 公开 deep baseline 全 LOSO 对比：`docs/artifacts/runs/2026-05-01_full-loso-deep-comparison/`
- case-study：`docs/artifacts/runs/2026-04-29_case-study/`
- 特征导出 clean roots：`docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/`、`docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/`
- 特征导出 closure：`docs/artifacts/runs/2026-04-27_feature-export-closure/`
- alignment/fusion 诊断输入：`docs/artifacts/runs/2026-04-22_alignment-e-baseline/`、`docs/artifacts/runs/2026-04-22_alignment-f-full/`、`docs/artifacts/runs/2026-04-22_alignment-g-baseline/`、`docs/artifacts/runs/2026-04-22_alignment-g-min/`

完整产物导航见 `docs/artifacts/ARTIFACTS.md`。

## 已验证（命名迁移轮）

1. `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall src scripts tests`
2. `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q`，结果为 `213 passed, 8 skipped, 78 warnings`。
3. 活动路径和文本命名审计通过；旧编号只保留在 archive、cleanup 和 migration map。
4. `docs/artifacts/runs/` manifest、registry、result matrix、resume command 旧路径审计通过；没有残留 `docs/artifacts/assets`、`docs/reports/assets`、`docs/artifacts/task_eval` 或 `scripts/task_eval` 引用。
5. 入口文档路径存在性检查通过：55 个真实路径，0 缺失。
6. `git diff --check`、`git lfs status`、`git lfs fsck` 通过。

## 本轮 E3 execution 校验

- 已新增 `tests/evaluation/fusion_stream_structure/`，覆盖合同、预处理、指标、CLI synthetic 输出和 optional evaluator wrapper。
- 已完成 synthetic evaluator validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-evaluator-validation/`。四类方法均可用；ClaSP `completed:8`，STUMPY `completed:8`，CLaP `clap_unavailable:8`；`metric rows=108`，`completed=88`，`unavailable=20`。
- 已完成小规模 Dingxin evaluator validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-evaluator-validation/`。`chronaris` / `naive_time_sync` 可用，`mult` / `contiformer` 为 `method_unavailable`；ClaSP `completed:4`，STUMPY `completed:4`，CLaP `clap_unavailable:4`；`metric rows=54`，`completed=44`，`unavailable=10`。
- 已完成 MulT / ContiFormer source audit：`docs/artifacts/runs/2026-07-07_fusion-stream-thirdparty-source-audit/`。当前 repo artifact 与本机外置备份中没有可复用融合表示流或 Dingxin 第三方模型 checkpoint；仅发现任务预测、检索 rank、scalar diagnostics、raw sequence bundle 和其他路径 checkpoint，因此不实现 adapter、不运行四方法 Dingxin E3。
- `claspy 0.2.8` / `stumpy 1.14.1` 已安装并通过 import 与 wrapper smoke test；STUMPY import 由代码设置 `NUMBA_DISABLE_CUDA=1`，避免当前 WSL/CUDA 探测崩溃。
- 未改 confirmed metrics，未回写论文协议快照，未删除历史检索 artifact。
- 已验证：`compileall` 通过；`pytest -q tests/evaluation/fusion_stream_structure` 为 `21 passed, 1 warning`；全量 `pytest -q` 为 `234 passed, 8 skipped, 318 warnings`。
