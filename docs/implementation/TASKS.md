# Chronaris 当前任务

更新时间：2026-07-08

## 当前任务

命名迁移已完成（见 `docs/maintenance/2026-07-05_name-migration-map.md`）。本轮（2026-07-07/2026-07-08）已执行 **fusion stream structure evaluation（E3 融合表示流结构评价）** 第一批编码、第二轮 evaluator validation、第三方来源审计、deep baseline representation export、四方法 Dingxin validation、结果审查和 Chronaris 受控候选优化：新增独立评价层、CLI、测试、no-training dry run、MulT / ContiFormer source audit、回归任务 held-out pooled embedding 导出、四方法结构评价、论文可用性判断，以及固定 split / labels / evaluator 下的 Chronaris 候选开发与 locked confirmation；confirmed metrics、论文协议快照和历史检索 artifact 均未改动。

- 计划入口：`docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/`（report / input_contract / metric_contract / implementation_plan / acceptance_checklist / evidence_manifest）。
- 执行入口：`src/chronaris/evaluation/fusion_stream_structure/` 与 `scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py`。
- 当前 validation 产物入口：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-evaluator-validation/` 与 `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-evaluator-validation/`。
- 第三方来源审计入口：`docs/artifacts/runs/2026-07-07_fusion-stream-thirdparty-source-audit/`，结论为 `C. no_reusable_sources`。
- deep baseline 表示导出入口：`docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/`，协议为回归任务（`T2_next_window_physiology_response`）+ `leave_one_view_out` + seed17 + `pooled_embedding`；MulT / ContiFormer 均完成 3 folds、各 108 行 OOF embedding。
- 四方法 Dingxin E3 validation 入口：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/`，方法为 `chronaris` / `naive_time_sync` / `mult` / `contiformer`，本 run `training_invoked=false`。
- E3 结果审查入口：`docs/artifacts/runs/2026-07-08_e3-result-review/`，结论为 `D. needs_chronaris_optimization`；当前 E3 对 Chronaris 是 mixed、非差异性结果，不建议作为论文正文优势证据；已生成下一轮受控优化 prompt。
- Chronaris 受控优化开发入口：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-dev/`，注册 8 个候选、8 个 smoke 通过、8 个 dev run 完成，Pareto 选择 `chr_v2_residual_delta_h64`。
- Chronaris locked confirmation 入口：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-confirm/`，锁定候选为 `chr_v2_residual_delta_h64`；回归任务相对旧 Chronaris 显著改善，但仍略弱于固定 MulT / ContiFormer 回归 baseline，E3 只出现一个 motif 正向信号。
- Chronaris OOF 表示导出入口：`docs/artifacts/runs/2026-07-08_chronaris-oof-representation-export/`，8 个候选共 972 行 held-out pooled embedding，表示族为 `Chronaris_T2_response_lovo_seed17_pooled_embedding`。
- 历史首轮 fallback 产物入口：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-execution/` 与 `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-dry-run/`。
- 当前环境已安装并验证 `claspy 0.2.8` / `stumpy 1.14.1`；ClaSP 与 STUMPY 真实 evaluator 已启用，CLaP 在短序列上保留结构化 `clap_unavailable`。

验收关注点（本轮 execution）：

- E3 long 表固定 `evidence_quadrant = fusion_stream_structure`，不并入既有 leaderboard。
- 小规模 Dingxin dry run 只使用既有 feature export 和可复用 `feature_values`；source audit 确认 `mult` / `contiformer` 当前无可复用融合表示流或可加载 checkpoint，已写 `method_unavailable`，未训练补齐。
- deep baseline export run 只训练 MulT / ContiFormer；导出的 `fusion_feature_*` 来自 held-out fold inference 的 `pooled_embedding`，不包含 logits、预测值、rank、embedding norm 或 attention diagnostics。
- 四方法 Dingxin E3 validation 只消费已导出的表示和既有 `chronaris` / `naive_time_sync` 融合流；该结构评价不写单一 winner，也不替代分类任务和回归任务。
- Chronaris 受控优化只在新 run root 训练和导出候选表示；固定 MulT / ContiFormer baseline、固定四方法 E3 evaluator、固定 split / labels，不回写论文协议快照或 confirmed metrics。
- locked candidate `chr_v2_residual_delta_h64` 的论文建议位置是补充或附录诊断：它修复旧 Chronaris 回归任务明显落后问题，并给出一个 E3 motif 正向信号，但不能写成整体优于 MulT / ContiFormer。
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
- E3 deep baseline 表示导出：`docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/`
- E3 四方法 Dingxin validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/`
- E3 结果审查与论文可用性判断：`docs/artifacts/runs/2026-07-08_e3-result-review/`
- E3 Chronaris 受控优化开发：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-dev/`
- E3 Chronaris locked confirmation：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-confirm/`
- E3 Chronaris OOF 表示导出：`docs/artifacts/runs/2026-07-08_chronaris-oof-representation-export/`
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
- 已完成 deep baseline representation export：`docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/`。`training_invoked=true`；MulT / ContiFormer 各完成 3 个 leave-one-view-out folds、各 108 行 held-out pooled embedding；`deep_baseline_oof_embeddings_long.csv` 和 checkpoint manifest 已写出。
- 已完成四方法 Dingxin E3 validation：`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/`。`training_invoked=false`；四方法各 108 行输入；`metric rows=160`，`completed=124`，`unavailable=36`；ClaSP `completed:12`，STUMPY `completed:12`，CLaP `clap_unavailable:12`。
- 已完成 E3 结果审查：`docs/artifacts/runs/2026-07-08_e3-result-review/`。四方法各 108 行输入均完整；当前 E3 指标大多并列或为零信号，CLaP 全部 unavailable；结论为 `D. needs_chronaris_optimization`，E3 不进入正文主证明，只作为附录诊断或下一轮 Chronaris 受控优化依据。
- 已完成 Chronaris 受控候选优化：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-dev/` 注册 8 个候选、8 个 smoke 通过、8 个 dev run 完成；Pareto 选择 `chr_v2_residual_delta_h64`，依据为回归任务改善、分类任务无明显退化和一个目标 E3 正向信号。
- 已完成 Chronaris locked confirmation：`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-confirm/`。锁定候选回归任务 RMSE 为 347.009106，相对旧 Chronaris 838.121039 改善，但仍略弱于 MulT 344.288975 与 ContiFormer 344.335110；分类任务 macro-F1 从 0.173333 到 0.166667，balanced accuracy 保持 0.333333；E3 `motif_event_consistency` 从 0 到 1，其他目标结构指标仍并列。
- 已完成 Chronaris OOF 表示导出：`docs/artifacts/runs/2026-07-08_chronaris-oof-representation-export/`。`training_invoked=true`；8 个候选、972 行 held-out pooled embedding、selected candidate 含 dev 与 locked confirmation 两组 216 行；confirmed metrics 与论文协议快照未改。
- `claspy 0.2.8` / `stumpy 1.14.1` 已安装并通过 import 与 wrapper smoke test；STUMPY import 由代码设置 `NUMBA_DISABLE_CUDA=1`，避免当前 WSL/CUDA 探测崩溃。
- 未改 confirmed metrics，未回写论文协议快照，未删除历史检索 artifact。
- 已重新验证：`compileall` 通过；`pytest -q tests/evaluation/fusion_stream_structure` 为 `34 passed, 1 warning`；全量 `pytest -q` 为 `247 passed, 8 skipped, 318 warnings`；`git diff --check` 通过。
