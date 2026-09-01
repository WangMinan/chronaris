# Chronaris 当前状态

更新时间：2026-09-01

## 一句话状态

论文主线已切换到 `research/thesis-continuous-semantic-fusion-202609`。2026 年 7 月安全滞后感知融合研究确认了单流旁路和安全门控的结构价值，但其公开数据、鼎新滞后对照和波次 A 结果受到训练确定性、重复归一化、测试折缩放或上下文时间轴问题影响，已退出毕业论文主结果。当前按评价协议 v3 先修复实验可信度，再实现可学习事件语义查询、显式时移监督和原生时间公开数据适配；协议与机制是硬门，真实任务双流增量是论文增强目标。

## 当前执行入口

- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- **当前论文主线**：分支 `research/thesis-continuous-semantic-fusion-202609`。
  - 评价协议 v3：[requirements/thesis-continuous-semantic-fusion-evaluation-v3.md](requirements/thesis-continuous-semantic-fusion-evaluation-v3.md)
  - 现场保护与基线预检：[artifacts/runs/2026-09-01_thesis-mainline-preflight/report.md](artifacts/runs/2026-09-01_thesis-mainline-preflight/report.md)
  - 可信训练第一批修复：[artifacts/runs/2026-09-01_training-trust-repair/report.md](artifacts/runs/2026-09-01_training-trust-repair/report.md)
  - 原生时间输入与 Euler 子步修复：[artifacts/runs/2026-09-01_native-time-euler-repair/report.md](artifacts/runs/2026-09-01_native-time-euler-repair/report.md)
  - 历史证据标签：`evidence/safe-lag-exploration-20260724`
- **当前研究主线（安全滞后感知融合）**：分支 `research/safe-lag-aware-fusion-20260718`。审计、研究计划与评价协议 v2 已提交：
  - 当前 Chronaris + CogPilot/CLARE 审计：[artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md](artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md)
  - 研究计划：[implementation/notes/safe-lag-aware-fusion-research-plan-2026-07-18.md](implementation/notes/safe-lag-aware-fusion-research-plan-2026-07-18.md)
  - 评价协议 v2：[requirements/safe-lag-aware-fusion-evaluation-v2.md](requirements/safe-lag-aware-fusion-evaluation-v2.md)
  - 波次 A 工程冒烟（机制验证）：[artifacts/runs/2026-07-22_safe-lag-wave-a-smoke/report.md](artifacts/runs/2026-07-22_safe-lag-wave-a-smoke/report.md)
  - 鼎新未来机动单折确认（真实下游）：[artifacts/runs/2026-07-22_dingxin-safe-lag-maneuver/report.md](artifacts/runs/2026-07-22_dingxin-safe-lag-maneuver/report.md)
  - CogPilot 飞行难度分类（公开数据正向）：[artifacts/runs/2026-07-23_cogpilot-difficulty/report.md](artifacts/runs/2026-07-23_cogpilot-difficulty/report.md)
- 历史已收口线 1 — 简化下游评价（最新真实数据协议冻结确认）：
  - 协议：[requirements/simple-downstream-evaluation-v1.md](requirements/simple-downstream-evaluation-v1.md)
  - 实施说明：[implementation/notes/simple-downstream-rebuild-2026-07-15.md](implementation/notes/simple-downstream-rebuild-2026-07-15.md)
  - 简化任务协议与兼容性审计：[artifacts/runs/2026-07-16_simple-downstream-protocol/report.md](artifacts/runs/2026-07-16_simple-downstream-protocol/report.md)
  - 选定配置预训练工程冒烟：[artifacts/runs/2026-07-16_simple-downstream-pretraining-smoke/report.md](artifacts/runs/2026-07-16_simple-downstream-pretraining-smoke/report.md)
  - 六方法完整表示工程冒烟：[artifacts/runs/2026-07-16_simple-downstream-representations-smoke/report.md](artifacts/runs/2026-07-16_simple-downstream-representations-smoke/report.md)
  - 固定下游算法工程冒烟：[artifacts/runs/2026-07-16_simple-downstream-consumer-smoke/report.md](artifacts/runs/2026-07-16_simple-downstream-consumer-smoke/report.md)
  - 正式预训练确认：[artifacts/runs/2026-07-16_simple-downstream-pretraining-confirmation/report.md](artifacts/runs/2026-07-16_simple-downstream-pretraining-confirmation/report.md)
  - 正式表示导出确认：[artifacts/runs/2026-07-16_simple-downstream-representations-confirmation/report.md](artifacts/runs/2026-07-16_simple-downstream-representations-confirmation/report.md)
  - 正式下游评价：[artifacts/runs/2026-07-16_simple-downstream-confirmation/report.md](artifacts/runs/2026-07-16_simple-downstream-confirmation/report.md)
  - 论文证据与独立复核：[artifacts/runs/2026-07-16_simple-downstream-thesis-evidence/report.md](artifacts/runs/2026-07-16_simple-downstream-thesis-evidence/report.md)
- 历史已收口线 2 — 任务感知安全残差（含残差激活筛选）：
  - 教师辅助残差激活与任务解耦差距报告：[artifacts/runs/2026-07-15_dingxin-residual-activation-task-decoupled/gap_report.md](artifacts/runs/2026-07-15_dingxin-residual-activation-task-decoupled/gap_report.md)
  - 残差激活预注册：[implementation/notes/dingxin-residual-activation-2026-07-15.md](implementation/notes/dingxin-residual-activation-2026-07-15.md)
  - 鼎新任务感知安全残差阶段 2 差距报告：[artifacts/runs/2026-07-15_dingxin-task-aware-safe-residual/gap_report.md](artifacts/runs/2026-07-15_dingxin-task-aware-safe-residual/gap_report.md)
  - 任务感知安全残差融合预注册：[implementation/notes/dingxin-task-aware-safe-residual-2026-07-15.md](implementation/notes/dingxin-task-aware-safe-residual-2026-07-15.md)
- 鼎新目标重构与统一条件确认：[artifacts/runs/2026-07-15_dingxin-target-reconstruction-confirmation/gap_report.md](artifacts/runs/2026-07-15_dingxin-target-reconstruction-confirmation/gap_report.md)
- 鼎新目标重构短预算筛选：[artifacts/runs/2026-07-15_dingxin-target-reconstruction/gap_report.md](artifacts/runs/2026-07-15_dingxin-target-reconstruction/gap_report.md)
- 目标重构预注册：[implementation/notes/dingxin-target-reconstruction-2026-07-15.md](implementation/notes/dingxin-target-reconstruction-2026-07-15.md)
- 鼎新任务协议修复与跨视图稳定化：[artifacts/runs/2026-07-14_dingxin-task-stability/gap_report.md](artifacts/runs/2026-07-14_dingxin-task-stability/gap_report.md)
- 前轮鼎新核心任务可行性审计：[artifacts/runs/2026-07-14_dingxin-core-feasibility/gap_report.md](artifacts/runs/2026-07-14_dingxin-core-feasibility/gap_report.md)
- 未推送核心恢复工作盘点：[implementation/notes/unpublished-core-recovery-inventory-2026-07-14.md](implementation/notes/unpublished-core-recovery-inventory-2026-07-14.md)
- 详细实施计划：[implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md)
- 长程运行手册：[implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)
- 固定数据证据策略：[requirements/foundation/fixed-data-evidence-strategy.md](requirements/foundation/fixed-data-evidence-strategy.md)
- 真实/仿真任务协议：[requirements/downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md)
- 仿真生成器规格：[requirements/synthetic-benchmark-spec.md](requirements/synthetic-benchmark-spec.md)
- 双流与融合表示合同：[requirements/model-contracts/application-fusion-stream-contract.md](requirements/model-contracts/application-fusion-stream-contract.md)
- G1 固定数据审计：[artifacts/runs/2026-07-10_fixed-data-audit/report.md](artifacts/runs/2026-07-10_fixed-data-audit/report.md)
- G2a 原始点冻结：[artifacts/runs/2026-07-10_dingxin-input-snapshot/report.md](artifacts/runs/2026-07-10_dingxin-input-snapshot/report.md)
- G2b 仿真基准审计：[artifacts/runs/2026-07-10_aviation-simulation-audit/report.md](artifacts/runs/2026-07-10_aviation-simulation-audit/report.md)
- G3a 统一表示合同冒烟验证：[artifacts/runs/2026-07-11_representation-contract-smoke/report.md](artifacts/runs/2026-07-11_representation-contract-smoke/report.md)
- G3b.1 浅层基线生产适配器冒烟验证：[artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/report.md](artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/report.md)
- G3b.2 深度基线生产适配器冒烟验证：[artifacts/runs/2026-07-11_deep-baseline-adapter-smoke/report.md](artifacts/runs/2026-07-11_deep-baseline-adapter-smoke/report.md)
- G3b.3 Chronaris 连续融合生产主干冒烟验证：[artifacts/runs/2026-07-11_chronaris-continuous-adapter-smoke/report.md](artifacts/runs/2026-07-11_chronaris-continuous-adapter-smoke/report.md)
- G3b.4 公共预训练与线性下游闭环冒烟验证：[artifacts/runs/2026-07-11_common-pretraining-loop-smoke/report.md](artifacts/runs/2026-07-11_common-pretraining-loop-smoke/report.md)
- G4.1 应用型下游消费者闭环冒烟验证：[artifacts/runs/2026-07-11_application-consumer-smoke/report.md](artifacts/runs/2026-07-11_application-consumer-smoke/report.md)
- G4.2 鼎新应用任务目标归档：[artifacts/runs/2026-07-11_dingxin-application-targets/report.md](artifacts/runs/2026-07-11_dingxin-application-targets/report.md)
- G4.2 鼎新原始双流上下文与目标绑定：[artifacts/runs/2026-07-11_dingxin-context-bindings/report.md](artifacts/runs/2026-07-11_dingxin-context-bindings/report.md)
- G4.2 鼎新外层折训练内验证划分：[artifacts/runs/2026-07-11_dingxin-inner-splits/report.md](artifacts/runs/2026-07-11_dingxin-inner-splits/report.md)
- G4.2 鼎新主协议首折公共预训练与表示导出：[artifacts/runs/2026-07-11_dingxin-fold-pretraining-smoke/report.md](artifacts/runs/2026-07-11_dingxin-fold-pretraining-smoke/report.md)
- G4.2 鼎新五折公共预训练与统一表示聚合审计：[artifacts/runs/2026-07-11_dingxin-five-fold-pretraining/report.md](artifacts/runs/2026-07-11_dingxin-five-fold-pretraining/report.md)
- G4.2 鼎新五折冻结表示 consumer 工程冒烟：[artifacts/runs/2026-07-11_dingxin-consumer-smoke/report.md](artifacts/runs/2026-07-11_dingxin-consumer-smoke/report.md)
- G4.2 鼎新 inner-train 嵌套目标：[artifacts/runs/2026-07-11_dingxin-nested-targets/report.md](artifacts/runs/2026-07-11_dingxin-nested-targets/report.md)
- G4.2 鼎新嵌套目标 validation-only consumer：[artifacts/runs/2026-07-11_dingxin-nested-validation/report.md](artifacts/runs/2026-07-11_dingxin-nested-validation/report.md)
- G5 编码器候选筛选全链路 smoke：[artifacts/runs/2026-07-11_encoder-candidate-screen-smoke/summary.md](artifacts/runs/2026-07-11_encoder-candidate-screen-smoke/summary.md)
- G5 seed 17 正式编码器候选筛选：[artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/summary.md](artifacts/runs/2026-07-11_encoder-candidate-screen-seed17/summary.md)
- G6 鼎新锁定重训单折协议验证：[artifacts/runs/2026-07-12_dingxin-locked-pretraining-smoke/report.md](artifacts/runs/2026-07-12_dingxin-locked-pretraining-smoke/report.md)
- G6 仿真预训练到鼎新适配协议验证：[artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-smoke/report.md](artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-smoke/report.md)
- G6 仿真预训练到鼎新无标签适配重训：[artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-pretraining-coalesced/report.md](artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-pretraining-coalesced/report.md)
- G6 仿真预训练适配统一表示：[artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-representations-coalesced/report.md](artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-representations-coalesced/report.md)
- G6 仿真预训练适配冻结下游评估：[artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-consumers-coalesced/report.md](artifacts/runs/2026-07-12_dingxin-synthetic-pretrain-adapt-consumers-coalesced/report.md)
- G6 Chronaris 机制消融重训协议验证：[artifacts/runs/2026-07-12_simulation-chronaris-ablation-pretraining-smoke/report.md](artifacts/runs/2026-07-12_simulation-chronaris-ablation-pretraining-smoke/report.md)
- G6 Chronaris 机制消融表示协议验证：[artifacts/runs/2026-07-12_simulation-chronaris-ablation-representations-smoke/report.md](artifacts/runs/2026-07-12_simulation-chronaris-ablation-representations-smoke/report.md)
- G7 Chronaris 四机制锁定消融下游评估：[artifacts/runs/2026-07-12_simulation-chronaris-ablation-consumers/report.md](artifacts/runs/2026-07-12_simulation-chronaris-ablation-consumers/report.md)
- G8 固定数据下游评估论文证据包：[artifacts/runs/2026-07-12_downstream-evidence-pack/report.md](artifacts/runs/2026-07-12_downstream-evidence-pack/report.md)
- G7 G2 锁定压力场景生成审计：[artifacts/runs/2026-07-12_aviation-simulation-locked-stress-audit/report.md](artifacts/runs/2026-07-12_aviation-simulation-locked-stress-audit/report.md)
- G6 仿真三随机种子五方法锁定重训：[artifacts/runs/2026-07-12_simulation-locked-pretraining/report.md](artifacts/runs/2026-07-12_simulation-locked-pretraining/report.md)
- G6 仿真三随机种子六方法统一表示：[artifacts/runs/2026-07-12_simulation-locked-representations/report.md](artifacts/runs/2026-07-12_simulation-locked-representations/report.md)
- G6 鼎新公共因果时间箱五方法运行时验证：[artifacts/runs/2026-07-12_dingxin-coalesced-pretraining-smoke/report.md](artifacts/runs/2026-07-12_dingxin-coalesced-pretraining-smoke/report.md)
- G6 鼎新 Chronaris GPU 运行时验证：[artifacts/runs/2026-07-12_dingxin-coalesced-chronaris-gpu-smoke/report.md](artifacts/runs/2026-07-12_dingxin-coalesced-chronaris-gpu-smoke/report.md)

## 已锁定事实

- 后续不把新增鼎新一手双流、人工工作负荷评价或专家事件标注作为依赖。
- 现有鼎新范围固定为 2 个 sortie、3 个 view、111 个 5 秒窗口。
- 当前真实数据任务使用 90 个完整未来 view-context；机动任务按唯一 vehicle-context 去重或加权，生理任务按 view-context 评价。
- 留一架次是主要分组确认，留一视图只作同一飞行过程中的视图适配诊断；不做显著性声明。
- 过去 30 秒内的正常运动学历史允许进入新主协议，未来 5 秒目标窗口、身份字段和绝对飞行进程禁止进入模型。
- 新主结果固定为统一任务无关表示加相同 Ridge/Logistic；正式结果打开后不再调参。
- 鼎新主任务固定为未来机动连续分数预测、辅助强度分类和未来生理字段预测；历史任务字段只用于兼容。
- 仿真器生成原始异步双流和独立 oracle，不生成任何方法的融合向量。
- 主比较固定为生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris。
- 六方法统一导出 96 点、64 维任务无关表示，并以相同无标签池化形成 64 维窗口表示；冻结表示评价是新主结果，既有端到端结果只作过拟合诊断，不再重训。
- 融合表示结构诊断只放附录，不参与模型选择。
- UAB/NASA 保持公开数据适配和上下文构造第二输入流证据，不等价于鼎新真实航电流。

## 已收口的固定数据长程记录

- 仿真 clean 主线：18/18 方法—随机种子 consumer、1152 条指标、768 条双流增益和 90 条轨迹级配对统计全部完成，7/7 门禁通过。Chronaris 在线性探针的仿真负荷分类/回归分别以 macro-F1 0.5110、RMSE 0.1928 居首，在持续时间约束分段的 frame macro-F1 0.4687、segmental F1@0.25 0.5319 居首；MiniRocket 负荷任务与边界 F1/延迟由 MulT 领先，因此结论是可解释的分项优势，不是全面第一。
- 仿真压力主线：3 seeds × 35 场景 × 6 方法共 630 份 `[192,96,64]` 表示和 630/630 个冻结 consumer 评估全部完成，两级门禁分别为 5/5、6/6。共生成 20,160 条指标、4,032 条退化斜率和 630 条 48 轨迹配对统计；所有 consumer 和阈值冻结自 G1 clean，压力场景不重训、不调参。Chronaris 在最高单因素压力下 15/21 个主指标组合位于前二，但随机和连续缺失的主指标平均退化斜率为 -0.0733/-0.1264，均排第六。
- 时间机制主线：144 份 G1 表示、24 个 Ridge 探针、840 个 G2 场景评价、3360 条指标和 630 条 48 轨迹配对统计全部完成，表示/consumer 门禁为 5/5、6/6。Chronaris 的时钟偏移 MAE 0.8883 秒、响应时延 MAE 7.4859 秒及两项容差命中率在四方法中最佳；MulT 的响应时延 Spearman 0.3584 高于 Chronaris 0.1766，排序相关性不作为 Chronaris 优势。
- 端到端辅助表：18/18 方法—随机种子、54 份独立表示、864 条指标和 7/7 门禁完成。MulT 在标签微调后的负荷分类和机动分段领先；Chronaris 仅在负荷回归 RMSE 0.2666 略居首。Chronaris 微调后的 macro-F1/RMSE/分段 frame macro-F1 为 0.3145/0.2666/0.3638，弱于冻结主表 0.5110/0.1928/0.4687，说明现有小样本端到端适配产生过拟合，冻结任务无关表示继续作为论文主表。
- 鼎新 real-only 主线：75/75 个训练单元、270/270 份 `frozen_task_agnostic_v1` 统一表示、90/90 个冻结 consumer 和 5,040 条指标全部完成，三级门禁分别为 8/8、6/6、8/8。主协议只用三个留一视图折汇总，两个留一架次折只作辅助；不报告窗口级显著性。Chronaris 的高生理响应 AUPRC 0.8839 和最差折平均 0.6516 均居首，相对最佳单流增益 0.0094；机动强度 Macro-F1 0.7394 与生理响应 RMSE 0.3365 均低于最佳基线。
- 设备调度：公共增强已固定在 CPU，模型 tensor 才送入 GPU；鼎新五方法的合并输入 GPU 冒烟连续完成且无 launch failure。正式队列继续保持单 CUDA 进程，checkpoint 记录设备历史，重复故障才原地迁移 CPU。
- CUDA 故障点已收敛到增强阶段的小粒度索引算子；公共增强、pretext target 和错误时移现固定在 CPU 确定性构造，再只把模型输入与 target tensor 送入 GPU。单 epoch CUDA 冒烟已确认 `training_device=cuda`、`augmentation_device=cpu`，鼎新基线队列将在严格单进程下采用该路径，若仍失败再按设备历史迁移 CPU。
- 锁定表示混合设备冒烟已完成：seed 17 六方法在 baseline CUDA、Chronaris CPU、朴素同步 CPU/PCA 下导出 train/validation/G2 共 18 份表示，耗时约 2 分 41 秒，18/18 输出、7/7 验收通过；任务 oracle 保持关闭。
- Chronaris 机制消融：四个变体 × 三个随机种子的锁定重训完成 12/12 个 checkpoint，每个变体—种子均导出仿真训练、验证与锁定测试三角色，共 36/36 份表示；12/12 个统一 consumer、768 条指标、384 条完整模型方向归一优势和 72 条 48 轨迹配对统计完成，三级门禁为 6/6、5/5、6/6。完整模型相对去连续演化在分类/回归/分段上为 +0.0691/+0.0278/+0.0142，相对去因果掩码为 +0.0060/+0.0082/+0.0662，相对单尺度时延为 +0.0056/+0.0118/+0.0623；去物理约束为 -0.0193/+0.0118/+0.0044，不写成全任务一致贡献。
- 消融表示冒烟：无物理约束变体已从锁定 checkpoint 回载并导出 G1 train、G1 validation、G2 held-out 三角色共 3 份统一表示，672 个上下文耗时约 22 秒，5/5 门禁通过。
- 上游完成后按门禁顺序自动进入六方法统一表示、validation 选参 consumer、G2 clean 锁定指标、35 场景压力曲线和四项 Chronaris 机制消融。

## 分支与历史实现

- 当前分支从 `origin/main` 建立；基线包含 2026-07-06 融合表示结构评价规划。
- `implement/fusion-stream-structure-20260707` 保留为历史实现分支，不整体合并。
- 后续只选择性复用 OOF/checkpoint manifest、resume、结构化 unavailable 和 ClaSP/STUMPY wrapper 思路，不移入旧 checkpoint、图件、大型 manifest 或 E3 候选选择逻辑。

## 本轮新增锁定链路

- 鼎新正式训练器只允许 inner-train/validation batch provider，outer-test 请求会 fail closed；训练折归一化、公共早停损失、checkpoint hash、设备与资源峰值均写入 manifest。
- 鼎新正式表示导出要求 75 个 checkpoint 全部完成后才允许打开 outer-test 原始输入，统一导出 3 seeds × 5 folds × 6 methods × 3 roles 共 270 份 `[N,96,64]` 表示。
- 鼎新正式下游消费者只在 validation 网格选择 Logistic/Ridge/MiniRocket 超参数；三个留一视图折作为主统计单位，两个留一架次折只作辅助，不报告窗口级显著性。
- Chronaris 四项消融已经接入与完整模型相同的锁定训练、checkpoint 回载、G1→G2 表示和正式 consumer 协议；完整模型与消融以 48 条潜在轨迹做配对差异。
- 仿真预训练到鼎新无标签适配已实现跨 schema 安全初始化：仅复制同名且形状一致的任务无关参数，重新初始化字段相关输入/重构层；源 checkpoint 哈希、复制张量和元素比例进入协议。
- 鼎新表示与 consumer 现从上游训练协议继承表示族：real-only 固定为 `frozen_task_agnostic_v1`，仿真预训练适配固定为 `synthetic_pretrain_real_adapt_v1`；两条轨道即使复用相同导出/consumer 代码也不会在 manifest 或主表中混写。
- 时间偏移与响应时延恢复任务已形成独立门禁链路：四种双流方法先导出 G1 六场景 train/validation 表示，再用 G1 validation 选择统一 Ridge 探针，最后只在 G2 的 35 个压力场景上评价；G2 不参与拟合或选参。
- 仿真端到端微调辅助链路已实现：六方法共享 `1e-4`、20 epoch、patience 5，联合训练线性负荷分类/回归头与两层因果 TCN；五个可训练编码器更新完整主干，朴素同步作为非参数 head-only 控制。微调表示单独写入 `end_to_end_finetuned_v1` 并显式声明使用任务标签，等待冻结主表完成后运行。
- 下游论文证据包已正式生成：锁定读取鼎新主折、仿真 clean、七因素压力、时间机制恢复、四项消融、端到端辅助表、仿真预训练适配和既有 UAB/NASA 公开适配证据，形成 8 层证据、7 幅中文图、预声明主指标表、54 项迁移增量表、证据矩阵和结论边界，9/9 门禁通过。7 图已逐图抽查并移除机器折 ID、轨迹 ID和未解释英文。
- 因果时间合并、OvR 高维求解器和批量持续时间解码接入后，完整测试为 `370 passed, 8 skipped, 319 warnings`；新增批量解码逐位等价、GPU 稳定性、表示族、微调和证据包测试均通过。
- 100 ms 因果时间箱、同时间观测聚合、缺失模态和旧 checkpoint 拒绝恢复的聚焦测试为 `10 passed`；run-level protocol 显式记录箱宽、时间戳策略和方法无关约束。
- MiniRocket 高维逻辑回归求解器已在 G2 指标比较前锁定：显式 OvR `liblinear` 对所有方法共用，64 维线性探针保留 `lbfgs`；应用消费者与鼎新消费者聚焦测试 `13 passed`。
- 持续时间约束解码已将相互独立的 batch 维向量化；`[192,96,5]` 真实 logits 从逐样本外推 77.55 秒降至 1.76 秒，约 43.95 倍。随机批次与真实前四样本均逐位等价，压力 consumer 从空根重启。
- 仿真 clean 锁定结果显示双流协同：Chronaris 相对最佳单流的持续时间约束分段 frame macro-F1 平均增益 0.1194、segmental F1@0.25 平均增益 0.1109，三个 seed 的最小增益仍为 0.0913/0.0774；其边界 F1@1s 平均增益为 -0.0053，边界定位优势不成立。

## 本轮已完成

- 建立固定数据、公开数据、仿真和结构诊断四层证据职责。
- 锁定 30 秒上下文、5 秒预测窗口、真实任务标签公式和 train-only 阈值。
- 锁定 G1/G2 生成族、96/24/48 潜在架次、成对压力等级和 oracle 合同。
- 锁定六方法统一表示、公共 pretext、Chronaris 秒级多尺度因果 lag 和四项消融。
- 锁定 leave-one-view-out 主协议、leave-one-sortie-out 辅助协议、MiniRocket/TCN/Viterbi 下游配置和三 seed 确认。
- 明确原始 snapshot、checkpoint 和 dense predictions 只进入被忽略的 `artifacts/application_evaluation/`。
- 新增固定数据应用评估包、只读审计 CLI、30 秒上下文构造、外层分组划分和训练折标签构造。
- 通过 MySQL 元数据把载机 TSPI 字段与目标机、质量字段分离；未解析字段不会回退为标签源。
- 每个 sortie 识别 10 个载机机动标签源字段；训练折自动剔除双 IQR 为 0 的速度、航向和过载语义组，实际使用 3 轴加速度、俯仰和滚转。
- 生理响应审计确认 12 个唯一 EEG/SpO₂ 字段，5 个外层折均完成且无元数据错误。
- 生成 `data_manifest`、字段角色、缺失率、sampling、fold 阈值、标签、split、overlap、进度和恢复命令等 G1 产物。
- 将既有对齐后投影判定为可能包含机动标签源信息，明确拒绝把它直接用于新的防泄漏分类主结果。
- 按锁定的 181 秒范围只读冻结两个 sortie 的原始点：每个 sortie 一份共享航电、每个 view 一份按 pilot 过滤的生理文件。
- 原始 snapshot 使用确定性 gzip JSONL 和 SHA-256；5.3 MB 高频值位于 `artifacts/application_evaluation/`，未进入 Git/LFS。
- 三个 view 的生理点均为 905；两个 sortie 的航电点均为 28,824，和既有 37 窗口逐项完全一致。
- 20 个机动标签源字段都能在 snapshot 中复核，并全部进入原字段、统计、差分、变化率和标准化副本的排除合同。
- `--resume` 在 2.47 秒内校验并复用 5 个文件，没有重复查询数据库。
- 实现 G1 状态空间与 G2 事件样条两个异构生成族，公开 API 只接收场景、飞行员参数档案和随机种子。
- 生成器把潜在轨迹与观测过程分离；相同 latent ID 的六个场景只改变采样、时钟、缺失、额外时延和噪声。
- 正式生成训练/验证/锁定测试 96/24/48 条潜在架次、1,008 个观测场景；profile、latent seed 和生成族跨 split 隔离。
- 全局低/中/高仿真负荷占比为 23.3%/43.0%/33.6%；每个锁定测试 profile 均有高负荷区间。
- G1 物理残差中位数最坏 0.0114，G2 残差 95% 分位最坏 0.0421；干净场景时延 ±1 秒命中率 100%。
- 正式重型 bundle 约 1,018 MB，只在被忽略目录；compact audit 约 1.2 MB，4 张中文图已逐张检查可读性。
- 独立 audit CLI 可在 3.13 秒内重建验收和图表，不重新生成重型 bundle。
- 新增任务无关 `chronaris.representation` 基础层，统一 30 秒原始异步双流批次、96 点查询轴和 `[B,T,64]` 融合表示。
- 仿真加载器只接受 `raw_dual_stream.npz` 的六个观测字段；真值、标签或额外字段注入会直接失败。
- 鼎新加载器从固定 snapshot 构造 12 个生理字段和 955 个跨架次同序航电字段，并在输入前排除全部 20 个机动标签源字段。
- 训练折中位数/四分位距归一化与主成分分析均记录拟合样本哈希，锁定测试样本重叠会直接失败。
- 检查点注册表、严格融合表示序列化、留出折导出、样本/查询顺序哈希和缺失输出恢复均已实现。
- 六个方法接口使用合同探针完成 6/6 导出与 6/6 恢复复用；该结果只证明接口贯通，不是六种模型效果。
- G3a 紧凑证据约 144 KB；约 276 KB 的探针检查点和稠密表示保留在被忽略目录。
- 新增按字段逐项 forward-fill 的公共因果查询层；相同时间的重复观测按稳定顺序取最后一项，任何查询只读取当前及历史观测。
- 生理单流和航电单流复用同一个 `ContinuousTimeSingleStreamEncoder`，主干为开启因果注意力的连续时间编码器；两者只在输入投影维数上不同。
- 朴素时间同步无可训练参数，使用训练折中位数/四分位距归一化和无监督主成分投影，不跨越未来观测。
- 仿真三划分与鼎新三个不同视图共完成 6 个生产适配器导出；查询轴均为 96 点、输出均为 64 维，14/14 验收通过。
- 对未来观测增加大幅扰动，当前及历史查询输出最大变化为 0；对非激活模态增加扰动，两个单流输出最大变化也为 0。
- 仿真单流参数量为 110,208/111,168；鼎新单流为 111,168/292,224；朴素同步参数量为 0。该审计不代表任务性能排名。
- G3b.1 紧凑证据约 68 KB；约 4.3 MB 检查点和稠密表示位于被忽略目录。
- MulT 生产适配器使用双向跨模态因果注意力和因果自注意力，ContiFormer 生产适配器显式启用连续时间因果注意力；两者都只导出任务头前时序状态。
- vendored MulT 已补齐 key padding mask 传递，位置编码兼容非连续张量；历史非因果调用保持默认行为。
- 仿真与鼎新各完成 MulT/ContiFormer 留出折导出，共 4 个 `[B,96,64]` 表示，恢复复核 4/4 复用，15/15 验收通过。
- 对未来观测施加扰动，当前及历史输出最大变化为 0；分别扰动生理/航电历史时，四组表示的最小变化为 0.9862/1.322，两个输入流均真实进入计算图。
- 仿真 MulT/ContiFormer 参数量为 1,014,784/112,512；鼎新输入维数更高，对应 1,196,800/294,528。该审计不构成任务性能排名。
- G3b.2 紧凑证据约 64 KB；约 11 MB 检查点和稠密表示位于被忽略目录。
- 新增原始异步双流到 ODE-RNN 的严格桥接，padding 不触发更新；两流在 96 点公共查询轴上读取连续潜态并记录观测更新、正时间演化、查询次数和最大时间间隔。
- Chronaris 主融合按真实秒数使用 0–5、5–15、15–30 秒三个互斥可见域；空尺度从门控归一化中排除，历史固定点数窗口不进入新 checkpoint。
- 物理一致性清单逐项区分 active、disabled 和 unavailable，并记录 count、raw value、weighted value 和不可用原因；仿真/鼎新分别有 5/4 项可计算。
- 无连续演化、无物理、无因果掩码和单尺度时延四项消融已由同一配置枚举生成，字段级 diff 只命中目标机制；两套数据共 8 次有限值前向通过。
- 仿真与鼎新各完成 1 个 Chronaris 留出折导出，恢复复核 2/2 复用；未来扰动对历史输出最大变化为 0，生理/航电历史扰动最小变化为 0.1058/1.3205。
- 无因果掩码消融在仿真/鼎新上的未来反事实变化为 0.4875/0.7773，证明该消融真实打开未来可见域。
- Chronaris 仿真/鼎新参数量为 123,222/306,186；单样本前向约 0.14/1.24 秒。G3b.3 紧凑证据约 76 KB，约 2.0 MB 检查点与表示位于被忽略目录。
- 新增方法无关增强执行器，实际执行整段模态、连续区间、随机点、时钟偏移和时间抖动，并保留增强观测到原始观测的逐点来源索引。
- masked reconstruction 只在查询来源确实被删除或替换的位置计算；短期预测和固定 `{-10,-5,5,10}` 秒错误时移判别共用同一 target/augmentation ID。
- 五个可训练方法共享同一个可微分编码接口、三个公共头、AdamW 候选 A 和训练 step；无有效目标结构化 unavailable，checkpoint 协议或代码 hash 改变时拒绝错误恢复。
- 从仿真 train split 的 16 个不同 G1 profile 各取一条 clean-asynchronous 轨迹，按 8/4/4 划分训练/验证/留出；路径中没有 validation 或 locked_test。
- 五个方法各训练 1 epoch、2 step，三个公共目标产生 30 条 active 记录；累计训练 7.29 秒，其中 Chronaris 5.89 秒，其他方法 0.26–0.54 秒。
- 朴素时间同步只拟合训练折无监督变换；六方法三种 role 共导出 18 份表示，样本、查询轴和掩码一致，恢复 18/18 复用。
- 自动删除 Chronaris 留出折表示后只重建该项，重建前后 SHA-256 一致，其余 17 项保持复用。
- 仿真 workload 真值只有在五个 checkpoint 完成后才打开；固定 Logistic/Ridge 产生 72 条全部可计算的 smoke-only 指标，不进入 confirmed metrics 或模型选择。
- G3b.4 紧凑证据约 180 KB；约 37 MB checkpoint、表示、target/prediction 只在被忽略目录。此前 35 MB 开发计时目录已由正式 run 取代并清理。
- 从 16 条 G1 仿真训练轨迹各取 30/60/90/120 秒四个上下文，形成 64 个跨状态样本和 32/16/16 的 profile 隔离划分；未来 5 秒负荷与 96 点机动状态真值只在五个可训练 checkpoint 和 18 份表示完成后打开。
- 六方法在相同样本、查询轴和 checkpoint 下重新导出 18 份应用上下文表示；删除 Chronaris 留出表示后只重建该项，SHA-256 保持一致，随后 18/18 恢复复用。
- 固定线性探针、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码均已实现；分类、回归、校准、frame/segment/boundary/edit/delay 指标共 384 条且全部可计算。
- MiniROCKET 对窗口内恒定潜在维采用统一训练折方差过滤；六方法保留维数均写入模型清单，不读取验证或留出标签。
- 输出 256 条方向归一融合增益和 30 条以 4 条留出轨迹为独立单位的配对统计接口；全部标记 smoke only，不用于模型排序。
- 删除 Chronaris MiniROCKET 与 TCN 模型后分别只重建缺失组件，未删除模型哈希保持不变；TCN 初始化、dropout 与优化共用隔离 seed，重建预测哈希一致。
- G4.1 紧凑证据约 404 KB；约 15 MB 表示、消费者模型和逐样本预测只在被忽略目录，12/12 验收通过。
- 机动强度弱监督分类已把 G1 五个外层折的训练折阈值、96 个上下文标签和 20 个禁止输入的标签源字段固化为独立目标；每折 train/test 均覆盖低、中、高三类。
- 生理响应正式目标从冻结原始点重新计算当前与未来 5 秒窗口中位数；字段覆盖、IQR 缩放和高响应阈值只使用各折可用训练上下文。
- 原始点审计发现每个 view 的最后一个候选上下文只剩约 1 秒未来观测；正式目标从 93 个候选收敛为 90 个完整目标，3 个末端上下文结构化标记不可用。
- 原始点中位数与 G1 窗口均值兼容目标在五折上的 Spearman 为 0.9349–0.9432；两者相关但不相同，后者只保留为口径追溯。
- 五折两个任务共生成 10 个独立确定性 archive；真实 `--resume` 复用 10/10，archive 与阈值文件重写哈希稳定，原 snapshot 五个文件哈希保持不变。
- G4.2 目标归档紧凑证据约 108 KB，约 248 KB 目标 archive 位于被忽略目录，12/12 验收通过。
- 96 个目标上下文中 93 个拥有完整 30 秒原始输入；三个 `context_end_0036` 实际为 25.991 秒部分末窗，未放宽合同或虚构 181–185 秒数据。
- 五折绑定后，机动分类可用唯一上下文为 93 个，生理响应同时要求完整未来 5 秒，因此为 90 个；所有不可用原因随 fold/task 逐行保存。
- 统一原始输入 schema 为 12 个生理字段和 955 个航电字段；20 个机动标签源字段在 raw-to-index 映射层即删除，93 个可用上下文最大相对时间为 29.999 秒。
- 原始值不预生成稠密 bundle。允许字段被缓存为 CSR 结构，包含 10,255,756 个 float32 值、数组净大小 78.6 MB；完整 96 上下文审计耗时 12.34 秒、峰值内存 767 MB。
- 缓存与直接 gzip 切片输出逐值一致；10 个目标 archive、阈值和 5 个 snapshot 文件重新校验哈希，外层 train/test group 无交集。
- G4.2 上下文绑定紧凑证据约 632 KB，不产生新的原始值副本或模型 checkpoint，12/12 验收通过。
- 五个外层折均划分为 inner-train、validation、overlap embargo 和 outer-test；每折 93 个完整输入只属于一种角色，全部角色互斥且可复现。
- 外层训练组含两个架次时完整留出一个训练架次；只含同一架次时按末端七个时间块验证，并删除与验证窗口存在 30 秒原始区间重叠的 5–10 个上下文。
- 五折 inner-train/validation 规模依次为 31/31、31/31、38/14、19/7、38/14；共享航电流上的原始时间区间重叠数为 0。
- 分类三个角色均覆盖低、中、高三类；生理响应三个角色均有有限连续目标和高/非高两类。现有阈值只允许固定配置 smoke，正式候选筛选前必须以 inner-train 重拟合嵌套目标。
- G4.2 训练内划分紧凑证据约 208 KB，本 run 不训练模型、不读取 outer-test 指标且不形成候选排名，11/11 验收通过。
- 新增按小批量懒加载的精确训练折归一化、公共预训练和 OOF 表示导出；31 个训练上下文不再物化为整折稠密原始张量。
- 朴素同步使用 inner-train 因果 forward-fill 与随机化主成分分析（PCA）；checkpoint 显式记录 solver/random state，旧元数据缺失时结构化重建而非错误恢复。
- 留一视图主协议第一折的 inner-train/validation/outer-test 各 31 个上下文；五个可训练方法各完成 1 epoch、31 step，朴素同步仅拟合无监督变换。
- 六方法共生成 6 个 checkpoint 和 18 份 `[N,96,64]` 表示；第二遍恢复复用 18/18，同角色样本、查询轴和 source hash 对齐。
- 五方法累计训练 213.63 秒，其中 Chronaris 148.73 秒；单次完整成功链路峰值 1967.4 MB。约 79 MB checkpoint/表示位于被忽略目录，紧凑证据约 292 KB，12/12 通过。
- 首折预训练未打开机动分类或生理响应目标，outer-test 只导出表示、不计算任务指标，因此不构成方法排名。
- 三个留一视图主协议折与两个留一架次辅助折全部完成；每折 6 个 checkpoint、18 份 train/validation/outer-test 表示，五折总计 30 个 checkpoint 和 90 份表示。
- 聚合审计逐项重验 90 份 archive/manifest、样本顺序、source hash、checkpoint 文件哈希与 inner-train fit hash；每折第二遍恢复均为 18/18。
- 五折五个可训练方法累计训练 1066.45 秒，全部 run 的最高峰值内存为 2047.1 MB；约 393 MB 重型产物位于被忽略目录，五个子 run 60/60、聚合 13/13 通过。
- 五折预训练均保持任务目标关闭、outer-test 指标关闭和单一公共输入 schema；该里程碑只证明真实双流表示链路闭环，不构成效果排名。
- 五折六方法冻结表示已接入固定线性模型与 MiniROCKET；30 个方法—折组合形成 60 个消费者组件，首次拟合累计 144.34 秒，第二遍恢复 60/60 且预测哈希 30/30 一致。
- 机动强度分类、生理响应回归和高生理响应识别共生成 1680 条 smoke-only 指标，全部可计算；方向归一双流增益接口生成 1120 条记录。
- 时间 embargo 后五折实际角色目标累计为 440 个机动分类上下文和 425 个生理响应上下文；三个未来区间不足样本继续缺席，没有以零值补齐。
- 约 36 MB 消费者模型与逐样本预测位于被忽略目录，紧凑证据约 680 KB，15/15 通过；鼎新指标与仿真指标保持独立目录。
- 当前 consumer 仍使用 outer-train 拟合的阈值，只服务固定配置工程冒烟；正式候选筛选前必须按 inner-train 重建嵌套目标，因此当前指标不用于模型排名。
- 五折机动语义尺度、分位阈值、生理字段有效性/IQR 和高响应阈值已全部改为 inner-train 拟合；validation 与 outer-test 只应用参数，不参与估计。
- 嵌套目标生成 10 个确定性 archive，机动分类 440 个角色上下文、生理响应 425 个可用角色上下文；snapshot 哈希保持不变，10/10 通过。
- 相对 outer-train 工程冒烟口径，五折共有 75/440 个机动类别和 51/425 个高响应标签变化；连续生理响应 Spearman 为 0.9787–0.9971，证明嵌套重拟合改变了尺度而非简单改名。
- 三个采用时间块与 embargo 的 validation 只覆盖中/高机动类；该真实分布漂移被保留，不移动阈值补类。后续 macro-F1 必须固定三类标签集合。
- 嵌套目标重型 archive 约 108 KB、紧凑证据约 292 KB；本 run 不训练模型、不生成指标或排名。
- 五折六方法使用嵌套目标复跑固定线性与 MiniROCKET，形成 30 个方法—折 bundle；第二遍恢复 60/60、预测哈希 30/30 一致。
- validation-only 共生成 840 条全部可计算指标和 560 条方向归一双流增益；所有指标的 threshold scope 均为 inner-train nested，role 唯一为 validation。
- outer-test 未进入评价循环，不生成预测或指标；约 35 MB 模型/预测被忽略，紧凑证据约 348 KB，12/12 通过。
- 三个缺少低机动类的 validation 触发预期的类别分布告警，但 macro-F1 固定三类集合；不移动阈值、不删折、不补类。
- 选定配置鼎新确认的第一个外层折已完成五方法训练；Chronaris 在第 28 epoch 早停、最佳 epoch 为 20。该折六方法 train/validation/outer-test 三角色共 18 份表示已导出，任务目标与 outer-test 指标仍关闭。
- Chronaris 锁定训练已把连续对齐、物理一致性和因果方向三项损失真正接入反向传播；前 10 epoch 权重为 0，随后渐进升权。早停始终只使用公共自监督验证损失。
- 锁定训练单 epoch 五方法实跑完成 5/5 checkpoint、9/9 验收；四个基线在 RTX 4090 上训练，Chronaris 使用 CPU，峰值内存 3.50 GB、总耗时 1 分 41 秒。
- 正式下游 consumer 已升级为 G1 validation 固定网格选参的 Logistic/Ridge 与 MiniRocket、两层 64-channel 残差因果 TCN、patience 6 早停和训练折持续时间约束；TCN 支持 CUDA 训练并返回可移植 CPU checkpoint。
- G2 压力扩展固定 7 个单因素的全部等级和 mixed-severe，共 35 个场景；同一潜在轨迹跨等级复用相同 observation seed，避免把随机噪声重采样混入退化斜率。
- 48 条 G2 潜在轨迹共生成 1,680 个压力观测版本，latent、trajectory 和 observation randomness 均严格成对，方法无关生成器 7/7 验收通过；约 1.7 GB 重型数据只位于被忽略目录。

## 当前收口状态

- 本轮新增聚焦测试 `9 passed`；按仓库真实运行拓扑加载原始工作区的被忽略历史工件、同时使用本分支源码与测试执行完整 pytest，结果为 `381 passed, 8 skipped, 319 warnings`。
- 本轮改动文件 Ruff、`compileall src scripts tests` 和 `git diff --check` 已通过；三幅中文审计图已逐图检查，重型因果查询缓存与候选状态保持在 Git 忽略目录。
- 上限门禁失败后的退出动作已经执行：不新增候选、不启动冻结专家安全融合、不打开外层测试，也不追加为了制造领先而设计的实验。

## 最终验收门

1. 所有正式 run 的 `evidence_manifest.json` 必须为 `completed`，真实、仿真、迁移、标签微调和公开适配继续分层。
2. 完整测试、`compileall`、本轮改动文件 Ruff 和 `git diff --check` 全部通过；全仓 Ruff 的 252 项历史导入/未使用符号基线单独记录，不混入本任务修复。
3. 七幅论文图保持中文字体、长标签、图例和数值可读，不展示内部折 ID、轨迹 ID 或未解释术语。
4. 工作树只包含源码、测试、紧凑表格、清单、报告和图；checkpoint、稠密表示与逐样本预测继续位于被忽略目录。

## 本轮验证

- G1 正式 run：`completed`，MySQL metadata error 为 0，5 个 fold 均为 `completed`。
- G1 focused tests：`7 passed`，覆盖分组隔离、test 值不影响阈值、未知字段 fail closed 和零 IQR 剔除。
- G1 CLI 与新增包 `compileall`：通过。
- G2a focused suite 合并后为 `10 passed`；正式 run 为 `completed`，resume 复核通过。
- G2b simulation/audit focused tests：`11 passed`；smoke 13/13、formal 19/19 验收通过。
- G3a representation focused tests：`20 passed`；G1–G3a 联合聚焦测试 `41 passed`。
- G3a 统一合同 smoke：鼎新/仿真输入、六方法接口、留出折来源和恢复共 `14/14` 通过。
- G3b.1 因果查询与浅层适配器测试：`11 passed`；联合表示/适配器聚焦测试 `31 passed`。
- G3b.1 生产适配器 smoke：仿真/鼎新 6 个导出、恢复 6/6 复用、`14/14` 验收通过。
- G3b.2 深度基线聚焦测试：`7 passed`；联合模型聚焦测试 `35 passed, 2 skipped`。
- G3b.2 生产适配器 smoke：仿真/鼎新 4 个导出、恢复 4/4 复用、`15/15` 验收通过。
- G3b.3 连续主干聚焦测试：`11 passed, 2 skipped`；融合编码器联合聚焦测试 `29 passed, 2 skipped`。
- G3b.3 生产主干 smoke：仿真/鼎新 2 个导出、恢复 2/2 复用、`21/21` 验收通过。
- G3b.4 增强/目标/训练/下游聚焦测试：`20 passed`；训练、表示与融合编码器联合聚焦测试 `42 passed`。
- G3b.4 六方法闭环 smoke：五个训练 checkpoint、18 个表示、72 个 smoke 指标，`20/20` 验收通过。
- G4.1 consumer 聚焦测试：`10 passed`；六方法应用 consumer smoke 生成 18 个表示、384 条指标、256 条融合增益和 30 条配对统计，`12/12` 验收通过。
- G4.2 目标归档与既有上下文/折聚焦测试：`7 passed`；10 个目标 archive、90 个完整生理响应目标，`12/12` 验收通过。
- G4.2 上下文绑定与加载器聚焦测试：`7 passed`；93 个完整原始输入、90 个完整响应绑定，`12/12` 验收通过。
- G4.2 训练内划分聚焦测试：`4 passed`；五折角色穷尽互斥、共享航电区间零重叠、任务覆盖和确定性重建均通过，正式 run `11/11` 验收通过。
- G4.2 流式归一化、随机化 PCA、懒加载公共训练与 OOF 导出聚焦测试：`21 passed`；主协议首折真实 run 为 5 个训练 checkpoint、6 个总 checkpoint、18 个表示，`12/12` 验收通过。
- G5 候选配置、早停、排名和恢复聚焦测试：`17 passed`；20 候选单 epoch G1 smoke 为 `6/6`，耗时 5 分 03 秒、峰值内存 4.34 GB。
- 最终完整测试：`372 passed, 8 skipped, 319 warnings`。
- `compileall src scripts tests`、本轮改动文件 Ruff、`git diff --check`、读者术语、密钥模式和重型产物忽略检查通过；全仓 Ruff 仍有 252 项历史基线，集中在本轮未修改的归档脚本、旧模块和合并测试。
- 当前 Git 改动中没有 raw snapshot、完整仿真 bundle、checkpoint 或稠密表示；拟入仓内容仅为代码、测试、紧凑清单和审计报告。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
