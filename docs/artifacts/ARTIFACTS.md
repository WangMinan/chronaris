# Chronaris 产物索引

更新时间：2026-07-11

## 目录定位

当前可引用产物入口统一放在 `docs/artifacts/runs/`。目录命名采用 `YYYY-MM-DD_intent`，避免把当前入口继续绑定到历史阶段编号。

历史阶段编号报告、旧资产目录和兼容入口已经移入 `docs/artifacts/archive/`。清理和迁移记录放在 `docs/artifacts/cleanup/` 与 `docs/maintenance/`。

## 当前核心 runs

- `runs/2026-07-10_aviation-simulation-audit/`：**方法无关航空人机异步双流仿真正式审计**。被忽略的重型 bundle 约 1,018 MB，包含训练/验证/锁定测试 96/24/48 条潜在架次和每条六个观测版本，共 1,008 个 scenario；本目录保留 oracle validation、paired latent 审计、19 项验收、split/profile/seed 隔离和 4 张已抽查中文图。低/中/高负荷全局占比 23.3%/43.0%/33.6%，G1 残差中位数最坏 0.0114，G2 残差 95% 分位最坏 0.0421，干净场景时延 ±1 秒命中率 100%。该证据只承担仿真真值与压力机制验证，不等价于新增鼎新真实数据。
- `runs/2026-07-10_dingxin-input-snapshot/`：**鼎新原始异步点冻结紧凑证据**。实际 5.3 MB 高频值保存在被忽略的 `artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot/`；本目录只提交文件 SHA-256、点数、时间范围、measurement 分布和字段排除合同。2 个 sortie 的航电点均为 28,824，3 个 view 的生理点均为 905，6/6 与既有窗口 manifest 一致，20/20 机动标签源字段明确禁止进入新分类输入。本 run 未训练模型、未生成下游指标。
- `runs/2026-07-10_fixed-data-audit/`：**固定数据、字段语义与下游任务合同审计**。确认现有 2 个 sortie、3 个 view、111 个窗口可构造 96 个机动分类上下文和 93 个未来生理响应上下文；5 个外层折均完成，MySQL 字段语义解析无错误。该 run 不含模型训练结果；窗口均值只用于 G1 目标可行性验证，正式原始点实验将使用窗口中位数。历史对齐后投影因可能包含机动标签源信息，不具备新的防泄漏分类主结果资格。
- `runs/2026-07-06_fusion-stream-structure-plan/`：**融合表示流结构评价开发计划（E3）**，是 planning 文档，不是实验结果。包含 `report.md`、`input_contract.md`、`metric_contract.md`、`implementation_plan.md`、`acceptance_checklist.md`、`evidence_manifest.json`。本轮未编码、未训练、未改 confirmed metrics；后续编码任务需等待人工 review 本计划后再执行。
- `runs/2026-07-03_thesis-protocol-snapshot/`：论文协议快照，包含 `experiment_registry.csv`、`result_matrix_long.csv`、`result_matrix_summary.csv`、`claim_boundary_table.csv`、`thesis_protocol_summary.json` 和 `report.md`。
- `runs/2026-07-02_metric-calibration/`：指标校准，保留分类任务校准、公开路线校准、检索任务沿用边界和 GPU/runtime 记录。
- `runs/2026-07-02_selected-model-summary/`：选定模型汇总。
- `runs/2026-07-02_selected-model-reevaluation/`：选定模型再评估。
- `runs/2026-07-02_stream-role-fusion/`：流角色融合。
- `runs/2026-07-02_task-head-calibration/`：任务头校准。
- `runs/2026-07-02_cross-evidence-matrix/`：跨证据矩阵。
- `runs/2026-07-02_dingxin-thirdparty-comparison/`：鼎新真实数据第三方模型对比。
- `runs/2026-07-02_public-fusion-ablation/`：公开融合消融。
- `runs/2026-07-01_public-model-comparison/`：公开模型对比。
- `runs/2026-07-01_public-fusion-calibration/`：公开融合校准和 GPU profiling 子目录。
- `runs/2026-06-21_thesis-materials-report-figures/`：论文图表材料。
- `runs/2026-06-19_dingxin-leakage-safe-ablation/`：鼎新防泄漏组件消融。
- `runs/2026-06-19_rotation-audit-figure-refresh/`：rotation audit 图件刷新。
- `runs/2026-06-14_llm-preprocessing-context/`：LLM preprocessing context。
- `runs/2026-06-14_llm-preprocessing-comparison/`：LLM preprocessing 对比。
- `runs/2026-06-13_runtime-schema-contract/`：runtime schema contract。
- `runs/2026-06-13_dingxin-weak-label-sweep-resume/`：鼎新弱监督任务扫描 stable resume。
- `runs/2026-06-13_dingxin-weak-label-sweep-partial/`：鼎新弱监督任务扫描 partial。
- `runs/2026-06-07_evidence-closure/`：证据闭环。
- `runs/2026-06-07_dingxin-multitask-real-closure/`：鼎新 weak-label multitask real closure。
- `runs/2026-06-07_dingxin-weak-label-multitask-sweep/`：鼎新弱监督任务 multitask sweep。
- `runs/2026-06-07_dingxin-opt-package/`：鼎新优化包。
- `runs/2026-06-07_dingxin-component-ablation/`：鼎新组件消融。
- `runs/2026-06-07_public-adapter-calibration/`：公开数据适配校准。
- `runs/2026-06-07_public-transfer-boundary/`：公开数据 transfer boundary。
- `runs/2026-06-07_rigid-body-diagnostics/`：刚体约束诊断。
- `runs/2026-06-07_rotation-audit-closure/`：rotation audit closure。
- `runs/2026-06-07_semantic-support/`：semantic support。
- `runs/2026-06-07_semantic-event-support/`：semantic event support。
- `runs/2026-06-07_runtime-inference-service/`：runtime inference replay。
- `runs/2026-06-07_midterm-evidence-pack/`：中期证据包。
- `runs/2026-05-09_public-fusion-nasa-full-confirm/`：NASA 公开融合确认。
- `runs/2026-05-08_public-mainline-uab-robust-prior-r1/`：公开主线汇总。
- `runs/2026-05-08_public-opt-nasa-prepared-v2/`、`runs/2026-05-08_public-opt-uab-robust-prior-r1/` 与 `runs/2026-05-08_public-opt-uab-heat-specialist-r1/`：公开数据准备、稳健 prior 和 heat specialist 优化证据。
- `runs/2026-05-06_semantic-support-baseline/` 与 `runs/2026-05-06_anchor-windows/`：semantic support baseline 与 anchor windows。
- `runs/2026-05-06_public-fusion-screen-round2/`：公开融合 screen round2。
- `runs/2026-05-06_public-fusion-nasa-confirm/`：NASA 公开融合确认。
- `runs/2026-05-06_public-fusion-uab-confirm/`：UAB 公开融合确认。
- `runs/2026-05-06_public-opt-nasa-round1/`：NASA 公开优化结果。
- `runs/2026-05-06_public-opt-nasa-prepared/`、`runs/2026-05-06_public-opt-uab/` 与 `runs/2026-05-06_public-opt-uab-torch/`：公开数据准备和优化输入。
- `runs/2026-05-04_public-opt-uab-prepared/` 与 `runs/2026-05-04_dingxin-opt-package/`：UAB 公开准备根和鼎新优化包基线。
- `runs/2026-05-01_deep-real-sortie-prepared/` 与 `runs/2026-05-01_deep-comparison-prepared/`：deep baseline 真实架次和公开序列准备根。
- `runs/2026-05-01_full-loso-deep-comparison/`：公开 deep baseline 全 LOSO 对比。
- `runs/2026-04-29_case-study/`：case-study summary 与 ablation table。
- `runs/2026-05-02_feature-export-e-allwindow-clean/` 与 `runs/2026-05-02_feature-export-f-allwindow-clean/`：特征导出 clean roots。
- `runs/2026-04-27_feature-export-closure/`：带 causal fusion summary 的特征导出 closure，用于真实架次序列准备和 case-study 默认入口。
- `runs/2026-04-22_alignment-e-baseline/`、`runs/2026-04-22_alignment-f-full/`、`runs/2026-04-22_alignment-g-baseline/` 与 `runs/2026-04-22_alignment-g-min/`：alignment/fusion 早期诊断输入，供 support summary 追溯。

## 引用规则

- 当前状态先看 `docs/STATE.md`。
- 当前执行队列先看 `docs/implementation/TASKS.md`。
- 论文需求先看 `docs/requirements/SPEC.md`。
- 论文图表和实验表优先从 `runs/2026-07-03_thesis-protocol-snapshot/experiment_registry.csv` 与 `result_matrix_long.csv` 反查原始路径和边界。
- 公开 UAB/NASA 结果必须写成公开数据适配、校准或上下文构造第二输入流证据。
- 鼎新结果必须写成鼎新真实数据弱监督任务证据，不写成人工专家真值。
- LLM preprocessing 只写成字段语义归一、规则复核、semantic hints、runtime explanation 和 pending human review packet。
- E3（融合表示流结构评价）结果单独成表，`evidence_quadrant = fusion_stream_structure`，是无监督结构诊断，不与 T1/T2/T3 confirmed metrics 混算，也不替代 T1/T2；T3 仅叙事降级，artifact 不删除。

## 清理记录

- `cleanup/20260705-name-migration-cleanup.md`：本轮职责命名迁移和产物整理记录。
- `../maintenance/2026-07-05_name-migration-map.md`：old path 到 new path 的迁移表。
- 更早 cleanup 记录保留在 `cleanup/`，用于追溯历史删除、外置备份和 LFS 决策。
