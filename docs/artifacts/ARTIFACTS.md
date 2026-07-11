# Chronaris 产物索引

更新时间：2026-07-11

## 目录定位

当前可引用产物入口统一放在 `docs/artifacts/runs/`。目录命名采用 `YYYY-MM-DD_intent`，避免把当前入口继续绑定到历史阶段编号。

历史阶段编号报告、旧资产目录和兼容入口已经移入 `docs/artifacts/archive/`。清理和迁移记录放在 `docs/artifacts/cleanup/` 与 `docs/maintenance/`。

## 当前核心 runs

- `runs/2026-07-11_dingxin-application-targets/`：**鼎新两项弱监督任务独立目标归档**。机动强度分类保留 G1 五折训练阈值和 96 个上下文标签，每折 train/test 均覆盖低、中、高三类，20 个标签源字段继续全部禁止进入输入。生理响应从冻结原始点重算当前/未来 5 秒窗口中位数，字段筛选、IQR 与高响应阈值只用训练上下文；因原始点在 181 秒结束，3 个末端候选不足完整 5 秒未来区间，正式目标保留 90/93。五折两个任务共 10 个 archive，恢复 10/10 复用，原 snapshot 哈希不变，12/12 验收通过。约 248 KB 目标与阈值文件留在被忽略目录，本 run 未训练模型或生成任务指标。
- `runs/2026-07-11_application-consumer-smoke/`：**六方法应用型下游消费者闭环冒烟验证**。从 16 条 G1 仿真训练轨迹构造 64 个跨状态上下文和 32/16/16 profile 隔离划分；六方法共导出 18 份 96 点、64 维表示。固定线性探针、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码生成 384 条全部可计算的 smoke-only 指标、256 条方向归一融合增益和 30 条以 4 条留出轨迹为独立单位的配对统计。删除 Chronaris 表示、MiniROCKET 和 TCN 后均只重建缺失组件，未删除文件哈希不变，预测哈希一致；12/12 验收通过。约 15 MB 表示、消费者模型和逐样本预测留在被忽略目录，本 run 不形成模型排名。
- `runs/2026-07-11_shallow-baseline-adapter-smoke/`：**两个单流与朴素时间同步生产适配器冒烟验证**。仿真三划分与鼎新三个不同视图分别完成生理单流、航电单流和朴素时间同步留出折导出，共 6 个输出；恢复复核 6/6 复用，14/14 验收通过。未来观测扰动与非激活模态扰动对历史输出的最大变化均为 0。该 run 只验证生产适配器、因果边界和训练折隔离，未运行公共自监督训练或下游指标；约 4.3 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_deep-baseline-adapter-smoke/`：**MulT 与 ContiFormer 任务头前生产适配器冒烟验证**。仿真与鼎新各完成两个方法的留出折导出，共 4 个 96 点、64 维输出；恢复复核 4/4 复用，15/15 验收通过。严格未来扰动对历史输出的最大变化为 0，生理与航电历史扰动均产生非零表示变化。该 run 只验证因果深度基线、双流计算路径和训练折隔离，未运行公共自监督训练或下游指标；约 11 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_chronaris-continuous-adapter-smoke/`：**Chronaris 连续融合生产主干机制验证**。原始异步双流经独立 ODE-RNN、96 点连续查询、逐项物理可用性和 0–5/5–15/15–30 秒因果融合形成 64 维表示；仿真与鼎新各完成一个留出折输出，恢复 2/2 复用，21/21 验收通过。未来扰动不改变历史完整模型输出，而无因果掩码消融在两套数据上均暴露未来敏感性；四项固定消融共 8 次有限值前向通过。该 run 未执行公共自监督训练或下游指标；约 2.0 MB 检查点和稠密表示留在被忽略目录。
- `runs/2026-07-11_common-pretraining-loop-smoke/`：**六方法公共预训练—折外表示—线性下游闭环验证**。从仿真 train split 的 16 个不同 G1 profile 构造 8/4/4 折，五个方法共享增强、遮挡重构、短期预测和时延判别训练 1 epoch，朴素时间同步只拟合训练折无监督变换。六方法三种 role 共 18 个输出恢复 18/18 复用；自动删除 Chronaris 留出折后单项重建哈希一致。workload 真值只在五个 checkpoint 完成后打开，固定 Logistic/Ridge 产生 72 条 smoke-only 指标；20/20 验收通过。约 37 MB checkpoint、表示和逐样本预测留在被忽略目录。
- `runs/2026-07-11_representation-contract-smoke/`：**统一双流输入与融合表示合同冒烟验证**。鼎新 12 个生理字段、955 个跨架次同序航电字段和 20 个标签源排除项通过同一批次合同；仿真只读取 7/12 个观测字段。六个方法接口均生成 96 点、64 维合同探针表示，样本标识、查询时间、有效掩码和留出折检查点来源一致，14/14 验收通过。该 run 只证明基础设施和恢复机制贯通，不是六种模型结果；探针检查点与稠密表示位于被忽略目录。
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
- 融合表示流结构评价历史结果单独成表，`evidence_quadrant = fusion_stream_structure`，作为无监督结构诊断，不与已确认的分类、回归和检索指标混算，也不替代主分类与回归任务；历史检索产物保留，但不再承担主叙事。

## 清理记录

- `cleanup/20260705-name-migration-cleanup.md`：本轮职责命名迁移和产物整理记录。
- `../maintenance/2026-07-05_name-migration-map.md`：old path 到 new path 的迁移表。
- 更早 cleanup 记录保留在 `cleanup/`，用于追溯历史删除、外置备份和 LFS 决策。
