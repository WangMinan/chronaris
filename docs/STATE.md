# Chronaris 当前状态

更新时间：2026-07-10

## 一句话状态

固定数据下游评估与完整论文实验长程 goal 正在执行，当前分支为 `codex/fixed-data-downstream-evaluation-20260710`。G1 固定数据与标签泄漏审计已完成：111 个窗口形成 96 个机动分类上下文和 93 个未来生理响应上下文，5 个外层折均可构造训练折标签；本轮未训练模型、未修改既有确认指标。当前实施里程碑已推进到 G2a 鼎新原始异步点冻结。

## 当前执行入口

- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 详细实施计划：[implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md)
- 长程运行手册：[implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)
- 固定数据证据策略：[requirements/foundation/fixed-data-evidence-strategy.md](requirements/foundation/fixed-data-evidence-strategy.md)
- 真实/仿真任务协议：[requirements/downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md)
- 仿真生成器规格：[requirements/synthetic-benchmark-spec.md](requirements/synthetic-benchmark-spec.md)
- 双流与融合表示合同：[requirements/model-contracts/application-fusion-stream-contract.md](requirements/model-contracts/application-fusion-stream-contract.md)
- G1 固定数据审计：[artifacts/runs/2026-07-10_fixed-data-audit/report.md](artifacts/runs/2026-07-10_fixed-data-audit/report.md)

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
- 新增固定数据应用评估包、只读审计 CLI、30 秒上下文构造、外层分组划分和训练折标签构造。
- 通过 MySQL 元数据把载机 TSPI 字段与目标机、质量字段分离；未解析字段不会回退为标签源。
- 每个 sortie 识别 10 个载机机动标签源字段；训练折自动剔除双 IQR 为 0 的速度、航向和过载语义组，实际使用 3 轴加速度、俯仰和滚转。
- 生理响应审计确认 12 个唯一 EEG/SpO₂ 字段，5 个外层折均完成且无元数据错误。
- 生成 `data_manifest`、字段角色、缺失率、sampling、fold 阈值、标签、split、overlap、进度和恢复命令等 G1 产物。
- 将既有对齐后投影判定为可能包含机动标签源信息，明确拒绝把它直接用于新的防泄漏分类主结果。

## 当前未完成

- 尚未实现原始 snapshot writer 和两架次原始异步点冻结。
- 尚未实现 G1/G2 仿真器。
- 尚未打通任务无关 Chronaris 连续融合编码器。
- 尚未实现六方法统一 OOF 导出与应用下游 benchmark。
- 尚未运行 screen、locked confirmation、stress sweep、消融或论文证据包。

## 下一验收门

G2a 必须在六方法真实数据训练前完成：

1. 只读冻结白名单中两个 sortie 的生理与航电原始异步点。
2. 把原始值写入被忽略的 `artifacts/application_evaluation/`，Git 中只保留紧凑 manifest 和审计摘要。
3. 对齐 snapshot 的时间范围、measurement、字段名、点数与现有特征导出范围。
4. 在原始输入合同中执行 G1 标签字段排除；禁止继续使用可能已编码标签源的历史投影完成机动分类主实验。
5. 若 InfluxDB 原始点不可读，写结构化 unavailable 产物并继续仿真主线，不伪造原始数据。

## 本轮验证

- G1 正式 run：`completed`，MySQL metadata error 为 0，5 个 fold 均为 `completed`。
- G1 focused tests：`7 passed`，覆盖分组隔离、test 值不影响阈值、未知字段 fail closed 和零 IQR 剔除。
- G1 CLI 与新增包 `compileall`：通过。
- 完整测试：`220 passed, 8 skipped, 317 warnings`。
- `compileall src scripts tests` 与 `git diff --check`：通过；LFS 和读者术语检查将在本里程碑提交前再次执行。
- 当前改动中没有 raw snapshot、bundle、checkpoint 或逐样本预测；G1 产物均为汇总、manifest 和可追溯标签表。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
