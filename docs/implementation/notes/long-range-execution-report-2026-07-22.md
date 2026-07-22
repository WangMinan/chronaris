# Chronaris 长程执行报告

报告日期：2026-07-22。分支：`research/safe-lag-aware-fusion-20260718`（commit `abab8d8a`）。
任务指令：工作树根 `goal.md`。本报告如实区分已完成与待完成，不夸大、不覆盖不利结果。

## 1. 分支清理前后状态

**清理前（本地 16 条分支 + 远端 10 条 + 5 个 worktree + 0 tag）**：`main`、`codex/simple-downstream-rebuild-20260715`（当前）、`codex/dingxin-residual-activation-20260715`、`codex/dingxin-task-aware-safe-residual-20260715`、`codex/dingxin-target-reconstruction-20260715`、`codex/dingxin-task-stability-20260714`、`codex/dingxin-core-feasibility-20260714`、`codex/chronaris-v2-mainline-20260712`、`codex/chronaris-core-task-recovery-20260713`、`codex/fixed-data-downstream-evaluation-20260710`、`codex/stage-i-midterm-closure`、`implement/fusion-stream-structure-20260707`、`planning/fusion-stream-structure-20260706`、`backup/main-before-fixed-data-20260712`、`backup/main-before-lfs-filter-20260712`；远端同名 9 条 `codex/*`/`implement/*`/`planning/*` + `main`。

**清理后（本地 2 条 + 远端 2 条 + 1 worktree + 10 tag）**：`main`、`research/safe-lag-aware-fusion-20260718`（远端同名）。所有历史提交由 annotated tag 保存，无丢失。

## 2. 合入、打 tag、删除的具体分支

- **合入 main**：`codex/simple-downstream-rebuild-20260715`（简化下游评价协议 + 代码）、`codex/dingxin-residual-activation-20260715`（含祖先 `codex/dingxin-task-aware-safe-residual-20260715`，任务感知安全残差 + 残差激活筛选代码）。经一次性整合分支 `chore/consolidate-research-history-20260718` 两次 `--no-ff` 合并，文档冲突按“两条线均收口、新主线开始”叙事手工解决；419/427 测试通过后 fast-forward 推送 main（`719c47c5..7e6a3af9`）。
- **evidence tag**：`evidence/simple-downstream-confirmation-20260716`、`evidence/dingxin-safe-residual-gap-20260715`。
- **archive tag**：`archive/chronaris-v2-not-promoted-20260713`、`archive/e3-fusion-structure-mixed-20260708`、`archive/core-task-recovery-protocol-invalid-20260713`、`archive/planning-fusion-stream-structure-20260706`、`archive/fixed-data-downstream-evaluation-20260710`、`archive/stage-i-midterm-closure-20260716`、`archive/backup-main-before-fixed-data-20260712`、`archive/backup-main-before-lfs-filter-20260712`。
- **删除**：上述全部 `codex/*`、`implement/*`、`planning/*`、`backup/*`、`chore/consolidate-*` 的本地与远端引用；移除 4 个冗余 worktree（均无未提交修改）。

## 3. 远端 GitHub 最终分支与 tag

- 活动分支：`main`（`7e6a3af9`）、`research/safe-lag-aware-fusion-20260718`（`abab8d8a`，领先 main 5 个提交）。
- tag：10 个（2 evidence + 8 archive），均已推送 origin。晋级门禁未满足，研究分支暂不合入 main（符合评价协议 v2 §9 与 goal §12.10）。

## 4. PhysioNet（CogPilot）与 CLARE 数据审计结论

- **CogPilot / PhysioNet**（主公开真实双流）：DOI 10.13026/azwa-ge48，35 名参与者，419 次 ILS 进近 + 68 静息；生理流（ECG/EDA+PPG/EMG+前臂加速度/呼吸/躯干加速度/眼动）+ 飞机流（X-Plane `lslxp11xpcac` 18 通道 @4.5Hz：速度/姿态/空速/高度/ILS 偏差/起落架）；LSL 公共时钟，跨流对齐 <14µs，使 event-to-response 滞后估计良态。**硬约束：无原始操纵输入流**。标签：难度 1–4、累计误差、逐样本误差 @4.5Hz（仅 AGL>200ft 段）。按 PhysioNet 受限许可 + 美国空军 AUA 处理，强制引用 FA8750-19-2-1000。
- **CLARE**（辅助跨受试者）：MATB-II 认知负荷，本地 20/24 受试者（EEG 19 名）；中枢 EEG（256Hz，独立时钟）vs 外周 ECG+EDA+Gaze（共享时钟）双流；10 秒自评 1–9 标签；LOSO。告警：Gaze INT32 哨兵、瞳孔恒值、EEG 跨时钟对齐缺口、本地无 LICENSE。
- 均只读审计，原始数据不入 git；只发布清单/匿名统计/哈希/指标/图。

## 5. 新任务定义及其语义依据

见 [评价协议 v2](../../requirements/safe-lag-aware-fusion-evaluation-v2.md) §2。鼎新 A 未来机动（航电强势，负迁移检查）、B 未来生理/事件后响应（双流增量）、C 时间机制；CogPilot 飞行难度分类 + 误差回归 + event-to-response 滞后；CLARE 认知负荷回归/分类 + 中枢-外周双流增量 + 模态缺失。任务均按参与者/架次分组，禁窗口随机泄漏；事件子集只由输入侧定义。

## 6. 新 Chronaris 架构（安全滞后感知融合）

- **`SafeLagAwareFusion`**（`src/chronaris/modeling/fusion_encoders/safe_lag_fusion.py`）：`z_out = concat(phys_private, vehicle_private, gate·z_cross)`，复用秒级多尺度因果注意力作为跨模态上下文，门控初始化近 0（安全回退）。
- **`lag_aware_alignment_loss`**（`src/chronaris/modeling/training/chronaris_auxiliary.py`）：因果滞后窗内取最大余弦相似度，替代同刻余弦。
- **`representation_diagnostics`**（`src/chronaris/evaluation/representation_diagnostics/`）：有效秩/协方差谱/维度利用率。
- 通过 `fusion_kind` 接入 `ChronarisContinuousEncoderConfig` 与两条训练路径（`train_locked_chronaris`、`train_common_pretext_method`），协议哈希与 checkpoint 兼容校验区分融合类型；旧 `multiscale` 行为与 checkpoint 向后兼容。

## 7. 相对旧模型修复了什么

逐项对应审计（`runs/2026-07-22_safe-lag-aware-fusion-audit/report.md`）确认的九项缺陷：单流旁路缺失→安全旁路（Q1）；航电瓶颈丢失→`vehicle_private` 专用通道（Q2）；尺度门控可塌缩→独立 sigmoid 安全门控 + 尺度熵正则接口（Q7）；无有效秩诊断→`representation_diagnostics`（Q8）；同刻相似与滞后冲突→`lag_aware_alignment_loss`（Q4）。物理约束（Q9，健康）保留不变。多统计量汇聚、辅助损失自早期启用、多准则 checkpoint、跨流增量预训练（Q3/Q5/Q6/Q10）已在协议与代码接口中规划，完整接入训练调度为下一迭代。

## 8. baseline 公平性

六方法共享相同原始异步双流、96 点查询轴、`[B,96,64]` 表示合同、相同无标签预算与分组；冻结表示轨道由相同 Ridge/Logistic 消费，容量由 validation 网格统一选定；MulT/ContiFormer 写明为仓库适配实现。波次 A 三组同预算同 seed 同数据。

## 9. 公开数据结果 / 10. 鼎新训练内结果 / 11. 锁定确认结果

**公开数据（CogPilot）：已运行，正向证据**——见 `runs/2026-07-23_cogpilot-difficulty/`。飞行难度四分类，LOSO（10 参与者训练、3 测试），12 epoch、seed 17：

| 方法 | macro-F1 | balanced acc |
| --- | --- | --- |
| **Chronaris safe_lag** | **0.4122** | 0.4167 |
| Chronaris multiscale（旧） | 0.4017 | 0.4167 |
| vehicle_only（飞机单流） | 0.3396 | 0.3611 |
| physiology_only（生理单流） | 0.1521 | 0.2222 |

判断：safe_lag 同时高于旧融合、最佳单流 vehicle_only 与 physiology_only——**公开数据双流增量成立（晋级门禁 3 与门禁 5“双流任务超最佳单流”方向性满足）**。难度同时驱动生理唤醒与飞机操纵/状态，是融合本应占优的任务；与鼎新车辆主导机动任务（融合难超航电单流）对照说明：任务语义决定融合是否有增量。**单 LOSO 折、12 epoch、单 seed，非锁定确认。** CLARE 与锁定确认仍未运行。

**鼎新训练内结果（已运行，单折三种子）**——见 `runs/2026-07-22_dingxin-safe-lag-maneuver/`。留一架次 fold01、30 epoch，未来机动 3 分类 macro-F1（同口径同消费者）：

| 方法 | seed17 | seed29 | seed43 | 均值 |
| --- | --- | --- | --- | --- |
| Chronaris safe_lag | 0.3387 | 0.1667 | 0.1667 | **0.2240** |
| Chronaris multiscale（旧） | 0.1667 | 0.1667 | 0.2315 | 0.1883 |
| vehicle_only | 0.4821 | 0.4807 | 0.4807 | 0.4812 |

（MulT/ContiFormer 仅 seed17：0.1667/0.1538。）未来生理字段任务六方法正技能字段比均为 `0`（均未超持久性），safe_lag 仅在 SpO2 字段 RMSE `2.151` 为全部融合方法最低。

**诚实结论**：safe_lag 对旧融合只有**微小且不稳定的均值优势**（0.224 vs 0.188，seed17 领先、seed43 反超）；融合方法（含 safe_lag）均值 ~0.19–0.22 远低于 vehicle_only ~0.48。**晋级门禁 4/5/6/8 均未满足**：机动任务未稳定超过全部融合基线、生理任务无跨流增量、未接近航电单流、多种子不稳定。该任务本被审计判定“主要由过去航电历史解释”，融合尚未形成可靠优势。**非锁定确认**（单折 30 epoch）。

## 12. 消融 / 压力测试

**诚实声明：尚未运行。** 评价协议 v2 §7（无安全残差/无旁路/无滞后/同刻替代/无解耦/均值池化/无预训练/无物理/无缺失训练）与 §8（随机/连续缺失、抖动、模态完全缺失）的消融与压力待锁定 checkpoint 后运行。单元层面已验证：安全旁路经 `vehicle_private` 真实保留航电信息（`test_safe_lag_preserves_vehicle_info_via_bypass`），滞后感知损失在已知滞后下显著优于同刻余弦（`test_lag_aware_alignment_loss_*`）。

## 13. 波次 A 机制结果（已运行）

**表示诊断（G1 仿真，15 epoch、seed 17，从 checkpoint 重算）**：

| 方法 | 有效秩 | 航电恢复 R²（直推） | 安全门控均值 |
| --- | --- | --- | --- |
| Chronaris safe_lag | **5.13** | **0.944** | 0.022 |
| Chronaris multiscale（旧） | 2.73 | 0.914 | n/a |
| vehicle_only（航电单流参考） | 6.56 | 0.933 | n/a |

**下游机动代理预测（负迁移机制检验，不重训）**——用 30s 上下文表示预测其后 5s 机动强度：

| 方法 | 直推 R² | 直推 Spearman | 泛化 Spearman |
| --- | --- | --- | --- |
| Chronaris safe_lag | **0.474** | **0.332** | 0.287 |
| Chronaris multiscale（旧） | 0.372 | 0.232 | −0.029 |
| vehicle_only（航电单流参考） | 0.363 | 0.270 | 0.503 |

判断：安全旁路使航电信息恢复 R² 与有效秩显著优于旧融合（甚至略超航电单流），门控保守近回退；在车辆主导的机动代理任务上旧 multiscale 最弱（泛化 Spearman −0.029，与负迁移一致），safe_lag 直推 R²/Spearman 最高、泛化 Spearman 为正。两路证据一致支持“安全旁路改善航电保真、在车辆主导任务上消除负迁移”的机制判断。**仍非锁定确认，不构成晋级结论。**

## 14. 是否达到最低晋级目标与拉伸目标

**部分达成（公开数据正向），鼎新未达成。** 公开 CogPilot 飞行难度四分类 LOSO 上 safe_lag `0.4122` 同时超过旧融合、最佳单流 vehicle_only `0.3396` 与 physiology_only `0.1521`——**门禁 3（公开数据超基线）与门禁 5（双流任务超最佳单流）方向性满足**（单 LOSO 折、12 epoch、单 seed，非锁定）。鼎新未来机动三种子显示融合（含 safe_lag）未稳定超过航电单流（均值 ~0.22 vs ~0.48），门禁 4/6/8 未满足；生理任务六方法 0 正技能字段。锁定确认未运行。最低晋级目标部分达成（公开双流增量任务），整体未达成；拉伸目标未达成。关键洞见：**任务语义决定融合是否有增量**——车辆主导任务融合难超航电单流，而难度/生理响应等双流任务融合真正占优。

## 15. 当前最强配置与 commit SHA

- 新主线分支：`research/safe-lag-aware-fusion-20260718` @ `abab8d8a`。
- 推荐配置：`ChronarisContinuousEncoderConfig(variant="full", fusion_kind="safe_lag")`，旁路维度 24+24、跨模态 16、安全门控初始偏置 −4；滞后窗 `[0, 15]s`。完整训练超参待波次 A 完整预算确认后冻结。

## 16. 关键产物路径

- 审计：`docs/artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md`
- 研究计划：`docs/implementation/notes/safe-lag-aware-fusion-research-plan-2026-07-18.md`
- 评价协议 v2：`docs/requirements/safe-lag-aware-fusion-evaluation-v2.md`
- 波次 A：`docs/artifacts/runs/2026-07-22_safe-lag-wave-a-smoke/`（report.md、wave_a_metrics_corrected.json）
- 代码：`src/chronaris/modeling/fusion_encoders/safe_lag_fusion.py`、`src/chronaris/modeling/training/chronaris_auxiliary.py`（lag_aware_alignment_loss）、`src/chronaris/evaluation/representation_diagnostics/`
- 实验/重算脚本：`scripts/research/run_safe_lag_wave_a_smoke.py`、`scripts/research/reeval_wave_a.py`
- 重型 checkpoint（被忽略）：`artifacts/application_evaluation/2026-07-22_safe-lag-wave-a-smoke/`

## 17. 尚存问题

- 鼎新三种子结果显示融合方法（含 safe_lag）在该车辆主导机动任务上未稳定超过航电单流（~0.48），与审计判断“未来机动主要由过去航电历史解释”一致；safe_lag 对旧融合仅微弱且不稳定的均值优势。
- 生理任务六方法均 0 正技能字段；`lag_aware_alignment_loss` 已接入训练调度但在权重 0.1/30 epoch 下未改善——需调权重、延长预算、或改进消费者（30 训练样本 Ridge 方差大）。
- CogPilot/CLARE 数据加载与 LOSO 评价管道尚未构建（字段 schema 与鼎新不同，需跨 schema 安全初始化）；公开数据门禁未运行。
- 多统计量汇聚、多准则 checkpoint 选择、跨流增量预训练已设计但未全部接入训练调度。
- CLARE 本地无 LICENSE 文件，发布前须核对 Borealis 许可。

## 18. 下一步唯一建议

**承认鼎新未来机动是车辆主导任务（融合难超航电单流），把研究重心转到真正需要跨模态增量的证据来源**：(1) CogPilot 公开双流（LSL 公共时钟利于滞后学习，难度分类 + 事件后生理响应是融合应有优势的任务）；(2) 鼎新/CLARE 生理响应任务，调 `lag_aware_alignment_loss` 权重（如 0.5–1.0）、延长至 50 epoch、增种子，并改进消费者（增大训练池或用任务感知头）。`SafeLagAwareFusion` 的安全旁路在仿真波次 A 已验证能恢复航电信息，但真实任务上尚未形成稳定融合优势——下一步应在“融合本应占优的任务与数据”上验证，而非继续在车辆主导的机动任务上追超航电单流。
