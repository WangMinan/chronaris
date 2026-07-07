# Chronaris 当前任务

更新时间：2026-07-06

## 当前任务

命名迁移已完成（见 `docs/maintenance/2026-07-05_name-migration-map.md`）。本轮（2026-07-06）新增任务是 **fusion stream structure evaluation planning（E3 融合表示流结构评价开发计划）**：只产出计划文档，不编码、不训练、不改 confirmed metrics。

- 计划入口：`docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/`（report / input_contract / metric_contract / implementation_plan / acceptance_checklist / evidence_manifest）。
- 计划结论：保留 T1/T2；T3 叙事降级为历史检索诊断 / 片段级复盘前置参考（artifact 不删除）；新增 E3，第一批实现 ClaSPy + STUMPY，TICC 列备选。
- 后续编码任务**需等待人工 review 本计划后再执行**。

验收关注点（本轮 planning）：

- 计划文档齐全且与既有 T1/T2/T3、论文协议快照、claim boundary 不冲突。
- 未创建 `src/scripts/tests` 实现文件或空占位。
- 未改 confirmed metrics、未回写 `result_matrix_long.csv` / `experiment_registry.csv` / `claim_boundary_table.csv`。
- `git diff --check`、`compileall` 通过。

## 当前代码结构

- `src/chronaris/feature_export/`：标准化融合特征导出、导出 manifest 读取和相关 profile。
- `src/chronaris/modeling/`：公共建模组件、GPU runtime helper、backbone 与 multitask training。
- `src/chronaris/evaluation/dingxin/`：鼎新真实数据弱监督任务、组件消融、第三方模型对比和任务头校准。
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

## 本轮 planning 校验

- 本轮仅新增/更新 docs，未改 `src/scripts/tests`。
- `compileall src scripts tests`、`git diff --check` 通过。
- 未改 confirmed metrics，未回写论文协议快照。
