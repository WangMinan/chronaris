# Chronaris 当前状态

更新时间：2026-07-10

## 一句话状态

固定数据下游评估与完整论文实验长程 goal 已启动，当前分支为 `codex/fixed-data-downstream-evaluation-20260710`。本轮已把“不再依赖新增鼎新数据/人工评价”的研究边界、真实任务、仿真基准、统一融合表示和长程运行门禁细化为仓库规格；尚未新增源码、尚未训练模型、尚未修改 confirmed metrics。下一实施里程碑是 G1 固定数据与标签泄漏审计。

## 当前执行入口

- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 详细实施计划：[implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md)
- 长程运行手册：[implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)
- 固定数据证据策略：[requirements/foundation/fixed-data-evidence-strategy.md](requirements/foundation/fixed-data-evidence-strategy.md)
- 真实/仿真任务协议：[requirements/downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md)
- 仿真生成器规格：[requirements/synthetic-benchmark-spec.md](requirements/synthetic-benchmark-spec.md)
- 双流与融合表示合同：[requirements/model-contracts/application-fusion-stream-contract.md](requirements/model-contracts/application-fusion-stream-contract.md)

## 已锁定事实

- 后续不把新增鼎新一手双流、人工工作负荷评价或专家事件标注作为依赖。
- 现有鼎新范围固定为 2 个 sortie、3 个 view、111 个 5 秒窗口。
- 鼎新主任务改为机动强度弱监督分类和机动诱发生理响应预测；历史任务字段只用于兼容。
- 仿真器生成原始异步双流和独立 oracle，不生成任何方法的融合向量。
- 主比较固定为生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris。
- 六方法统一输出 `[B,T,64]`；冻结表示评价是主结果，端到端微调是辅助结果。
- 融合表示结构诊断只放附录，不参与模型选择。
- UAB/NASA 保持公开数据适配和上下文构造第二输入流证据，不等价于鼎新真实航电流。

## 分支与历史实现

- 当前分支从 `origin/main` 建立；基线包含 2026-07-06 融合表示结构评价规划。
- `implement/fusion-stream-structure-20260707` 保留为历史实现分支，不整体合并。
- 后续只选择性复用 OOF/checkpoint manifest、resume、结构化 unavailable 和 ClaSP/STUMPY wrapper 思路，不移入旧 checkpoint、图件、大型 manifest 或 E3 候选选择逻辑。

## 本轮已完成

- 建立固定数据、公开数据、仿真和结构诊断四层证据职责。
- 锁定 30 秒上下文、5 秒预测窗口、真实任务标签公式和 train-only 阈值。
- 锁定 G1/G2 生成族、96/24/48 潜在架次、成对压力等级和 oracle 合同。
- 锁定六方法统一表示、公共 pretext、Chronaris 秒级多尺度因果 lag 和四项消融。
- 锁定 leave-one-view-out 主协议、leave-one-sortie-out 辅助协议、MiniRocket/TCN/Viterbi 下游配置和三 seed 确认。
- 明确原始 snapshot、checkpoint 和 dense predictions 只进入被忽略的 `artifacts/application_evaluation/`。

## 当前未完成

- 尚未实现固定数据审计、fold-fitted 标签或原始 snapshot writer。
- 尚未实现 G1/G2 仿真器。
- 尚未打通任务无关 Chronaris 连续融合编码器。
- 尚未实现六方法统一 OOF 导出与应用下游 benchmark。
- 尚未运行 screen、locked confirmation、stress sweep、消融或论文证据包。

## 下一验收门

G1 固定数据审计必须在任何新训练前完成：

1. 对三个 view 建立 30 秒上下文和连续性清单。
2. 通过 MySQL 元数据识别机动标签字段，不允许未知字段全选 fallback。
3. 在每个外层训练折拟合标签阈值和生理目标尺度。
4. 证明标签源字段及确定性派生字段未进入模型输入。
5. 输出 split、字段角色、缺失率、样本覆盖和 overlap audit。

## 本轮验证

- 当前改动文档链接检查：77 个链接，0 缺失。
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall -q src scripts tests`：通过。
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q`：`213 passed, 8 skipped, 317 warnings`。
- `git diff --check`：通过。
- `git lfs status` 与 `git lfs fsck`：通过。
- 当前改动中没有 raw snapshot、bundle、checkpoint 或逐样本预测。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
