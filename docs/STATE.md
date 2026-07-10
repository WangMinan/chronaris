# Chronaris 当前状态

更新时间：2026-07-11

## 一句话状态

固定数据下游评估与完整论文实验长程 goal 正在执行，当前分支为 `codex/fixed-data-downstream-evaluation-20260710`。固定数据审计、原始点冻结、方法无关仿真、统一表示基础设施以及两个单流/朴素时间同步生产适配器均已完成；最新生产适配器冒烟验证 14/14 通过。本轮尚未运行公共自监督训练或下游任务指标、未修改既有确认指标。当前实施里程碑已推进到 G3b.2 MulT 与 ContiFormer 生产适配器。

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
- G2b 仿真基准审计：[artifacts/runs/2026-07-10_aviation-simulation-audit/report.md](artifacts/runs/2026-07-10_aviation-simulation-audit/report.md)
- G3a 统一表示合同冒烟验证：[artifacts/runs/2026-07-11_representation-contract-smoke/report.md](artifacts/runs/2026-07-11_representation-contract-smoke/report.md)
- G3b.1 浅层基线生产适配器冒烟验证：[artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/report.md](artifacts/runs/2026-07-11_shallow-baseline-adapter-smoke/report.md)

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

## 当前未完成

- 尚未打通任务无关 Chronaris 连续融合编码器。
- MulT 与 ContiFormer 尚未从历史任务 wrapper 中分离为任务头前的生产适配器。
- 六方法尚未完成公共自监督训练；现有三个浅层基线只完成随机初始化/无监督变换的工程冒烟验证。
- 尚未实现应用下游算法与正式 benchmark。
- 尚未运行 screen、locked confirmation、stress sweep、消融或论文证据包。

## 下一验收门

G3b.2 必须在 Chronaris 连续主干改造前完成：

1. MulT 与 ContiFormer 必须导出任务头之前的 96 点时序状态并投影到 64 维。
2. 两个适配器都先经过公共因果查询层；跨模态注意力和连续时间自注意力不得重新引入未来信息。
3. 旧回归任务 checkpoint 不作为主表示初始化，适配器 checkpoint 明确记录 `label_used_for_encoder_training=false`。
4. 两个深度基线复用训练折归一化、共享增强 realization 和相同隐藏维/层数/训练预算合同。
5. 仿真与鼎新分别完成留出折导出、未来扰动审计、双流敏感性审计和恢复复核。

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
- 完整测试：`265 passed, 8 skipped, 317 warnings`。
- `compileall src scripts tests` 与 `git diff --check`：通过；LFS 和读者术语检查将在本里程碑提交前再次执行。
- 当前 Git 改动中没有 raw snapshot、完整仿真 bundle、checkpoint 或稠密表示；拟入仓内容仅为代码、测试、紧凑清单和审计报告。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
