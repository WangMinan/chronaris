# Chronaris 当前状态

更新时间：2026-07-10

## 一句话状态

固定数据下游评估与完整论文实验长程 goal 正在执行，当前分支为 `codex/fixed-data-downstream-evaluation-20260710`。G1 固定数据审计和 G2a 原始异步点冻结均已完成：111 个窗口形成 96/93 个应用上下文；两份共享航电文件与三份 view 生理文件共冻结 57,648/2,715 个原始点，6/6 点数对账、20/20 标签源排除检查通过。本轮未训练模型、未修改既有确认指标。当前实施里程碑已推进到 G2b 方法无关半物理仿真器。

## 当前执行入口

- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 详细实施计划：[implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-2026-07-10.md)
- 长程运行手册：[implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md](implementation/notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)
- 固定数据证据策略：[requirements/foundation/fixed-data-evidence-strategy.md](requirements/foundation/fixed-data-evidence-strategy.md)
- 真实/仿真任务协议：[requirements/downstream-evaluation-spec.md](requirements/downstream-evaluation-spec.md)
- 仿真生成器规格：[requirements/synthetic-benchmark-spec.md](requirements/synthetic-benchmark-spec.md)
- 双流与融合表示合同：[requirements/model-contracts/application-fusion-stream-contract.md](requirements/model-contracts/application-fusion-stream-contract.md)
- G1 固定数据审计：[artifacts/runs/2026-07-10_fixed-data-audit/report.md](artifacts/runs/2026-07-10_fixed-data-audit/report.md)
- G2a 原始点冻结：[artifacts/runs/2026-07-10_dingxin-input-snapshot/report.md](artifacts/runs/2026-07-10_dingxin-input-snapshot/report.md)

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
- 按锁定的 181 秒范围只读冻结两个 sortie 的原始点：每个 sortie 一份共享航电、每个 view 一份按 pilot 过滤的生理文件。
- 原始 snapshot 使用确定性 gzip JSONL 和 SHA-256；5.3 MB 高频值位于 `artifacts/application_evaluation/`，未进入 Git/LFS。
- 三个 view 的生理点均为 905；两个 sortie 的航电点均为 28,824，和既有 37 窗口逐项完全一致。
- 20 个机动标签源字段都能在 snapshot 中复核，并全部进入原字段、统计、差分、变化率和标准化副本的排除合同。
- `--resume` 在 2.47 秒内校验并复用 5 个文件，没有重复查询数据库。

## 当前未完成

- 尚未实现 G1/G2 仿真器。
- 尚未打通任务无关 Chronaris 连续融合编码器。
- 尚未实现六方法统一 OOF 导出与应用下游 benchmark。
- 尚未运行 screen、locked confirmation、stress sweep、消融或论文证据包。

## 下一验收门

G2b 必须在大规模模型 screen 前完成：

1. G1/G2 两个生成族的 API 不接受方法名、checkpoint 或候选配置。
2. 生成同一潜在轨迹的 clean/stress 成对观测，并输出状态、负荷、边界、时钟和响应时延 oracle。
3. 固定 96/24/48 潜在架次和 pilot/scenario/generator-family 隔离。
4. 通过 seed 复现、物理范围、状态覆盖、lag 恢复和生成族隔离测试。
5. 生成中文数据质量图，抽查标签、长标签、图例和数值可读性。

## 本轮验证

- G1 正式 run：`completed`，MySQL metadata error 为 0，5 个 fold 均为 `completed`。
- G1 focused tests：`7 passed`，覆盖分组隔离、test 值不影响阈值、未知字段 fail closed 和零 IQR 剔除。
- G1 CLI 与新增包 `compileall`：通过。
- G2a focused suite 合并后为 `10 passed`；正式 run 为 `completed`，resume 复核通过。
- 完整测试：`223 passed, 8 skipped, 317 warnings`。
- `compileall src scripts tests` 与 `git diff --check`：通过；LFS 和读者术语检查将在本里程碑提交前再次执行。
- 当前 Git 改动中没有 raw snapshot、bundle、checkpoint 或逐样本预测；G1/G2a 入仓产物均为汇总、manifest 和可追溯审计表。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
