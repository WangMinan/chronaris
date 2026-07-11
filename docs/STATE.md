# Chronaris 当前状态

更新时间：2026-07-11

## 一句话状态

固定数据下游评估与完整论文实验长程 goal 正在执行，当前分支为 `codex/fixed-data-downstream-evaluation-20260710`。G5 seed 17 正式筛选已经完成；鼎新正式三随机种子五折重训协议已通过单折 5/5 方法、8/8 验收并启动 75 个方法—折—随机种子队列。G6 仿真 seeds 17/29/43 正在从 checkpoint 继续；RTX 4090 在独占训练时再次发生驱动级 launch failure，未完成基线已按锁定配置迁移到 CPU，Chronaris 继续使用同批实测更快的 CPU 连续演化路径。G2 的 48 条潜在轨迹已扩展为 35 个严格成对压力场景，共 1,680 个观测版本并通过 7/7 审计；既有确认指标仍未修改。

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
- G6 Chronaris 机制消融重训协议验证：[artifacts/runs/2026-07-12_simulation-chronaris-ablation-pretraining-smoke/report.md](artifacts/runs/2026-07-12_simulation-chronaris-ablation-pretraining-smoke/report.md)
- G6 Chronaris 机制消融表示协议验证：[artifacts/runs/2026-07-12_simulation-chronaris-ablation-representations-smoke/report.md](artifacts/runs/2026-07-12_simulation-chronaris-ablation-representations-smoke/report.md)
- G7 G2 锁定压力场景生成审计：[artifacts/runs/2026-07-12_aviation-simulation-locked-stress-audit/report.md](artifacts/runs/2026-07-12_aviation-simulation-locked-stress-audit/report.md)

## 已锁定事实

- 后续不把新增鼎新一手双流、人工工作负荷评价或专家事件标注作为依赖。
- 现有鼎新范围固定为 2 个 sortie、3 个 view、111 个 5 秒窗口。
- 鼎新主任务改为机动强度弱监督分类和机动诱发生理响应预测；历史任务字段只用于兼容。
- 仿真器生成原始异步双流和独立 oracle，不生成任何方法的融合向量。
- 主比较固定为生理单流、航电单流、朴素时间同步、MulT、ContiFormer 和 Chronaris。
- 六方法统一输出 `[B,T,64]`；冻结表示评价是主结果，端到端微调是辅助结果。
- 融合表示结构诊断只放附录，不参与模型选择。
- UAB/NASA 保持公开数据适配和上下文构造第二输入流证据，不等价于鼎新真实航电流。

## 当前长程队列

- 仿真锁定重训：seeds 17/29 的五方法均已完成 50 epoch，队列已自动进入 seed 43；独占 CUDA 的重复驱动故障后只改变设备到 CPU，候选、预算、增强 realization 和验证规则保持不变，checkpoint 记录设备历史。
- 鼎新锁定重训：3 seeds × 5 个外层折 × 5 个选定配置，共 75 个训练单元；已保存第一训练单元的 37 个 epoch，当前等待仿真锁定训练收口后串行恢复，并默认沿用 CPU 故障降级路径。任务目标、outer-test 表示和指标仍保持关闭。
- 设备调度：并发故障后已改为单 CUDA 进程，但独占训练仍再次触发 launch failure；GPU 张量自检恢复后通过。长基线训练改走 CPU，后续只在短 TCN/微调队列重新评估 GPU，重复故障立即从 checkpoint 迁移到 CPU。
- CUDA 故障点已收敛到增强阶段的小粒度索引算子；公共增强、pretext target 和错误时移现固定在 CPU 确定性构造，再只把模型输入与 target tensor 送入 GPU。单 epoch CUDA 冒烟已确认 `training_device=cuda`、`augmentation_device=cpu`，鼎新基线队列将在严格单进程下采用该路径，若仍失败再按设备历史迁移 CPU。
- Chronaris 机制消融：四个变体 × 三个随机种子的 CPU 锁定重训已启动，与仿真基线的单 GPU 队列并行；后续仍需等待完整模型与消融表示后才能打开任务真值。
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
- 下游论文证据包生成器已实现：锁定读取鼎新主折、仿真 clean、七因素压力、时间机制恢复、四项消融、端到端辅助表和既有 UAB/NASA 公开适配证据，生成 7 幅中文图（含鼎新代表时间线与仿真 oracle 复盘）、预声明主指标表、证据矩阵、领先方法描述与 claim boundary；真实弱监督、仿真真值、标签微调和公开上下文适配不会混入同一结论层。
- 新增代码后完整测试收集 371 项且失败缓存为空；按既有 8 项条件跳过计算，为 `363 passed, 8 skipped`。新增 GPU 稳定性、表示族、微调和证据包定向测试均通过，`compileall`、Ruff 和 `git diff --check` 通过。

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

## 当前未完成

- G5 正式候选排序已完成；鼎新 seed 17 开发确认已完成前两个留一视图折并在第三折保留可恢复 checkpoint，现已由更完整的三 seed 五折锁定协议替代并停止冗余运行。压力数据生成已完成，压力表示、冻结 consumer、消融和论文证据包尚未完成。
- 仿真应用消费者目前仍是 16 条轨迹上的接口 smoke，不是完整 G1 开发筛选结果。
- 正式 G1→G2 consumer 代码已闭环但须等待 15 个锁定 checkpoint 与 54 份 clean 表示完成后才打开任务真值；当前尚未形成锁定任务指标。

## 下一验收门

G4.2 必须继续完成真实外层折的训练内验证、公共预训练和统一表示，再进入 screen：

1. 已完成：在每个外层训练组内按 view/sortie 和时间顺序固定 validation，并对共享航电原始区间实施 overlap embargo。
2. 归一化、增强和公共预训练只拟合 inner-train；outer-test 在候选冻结前只用于一次 smoke 结构验证，不参与选择。MiniROCKET 方差过滤和正式任务阈值在后续 consumer 阶段也必须只拟合 inner-train。
3. 已完成：五折六方法 checkpoint、train/validation/test 表示逐项匹配 schema、fit sample 和 source hash。
4. 已完成：真实外层折六方法表示—固定 consumer smoke；鼎新指标与仿真指标分目录保存，不形成混合平均分。
5. 已完成：按 inner-train 重建嵌套目标。
6. 已完成：仅在 validation 复跑固定 consumer，outer-test 指标保持关闭。
7. 已完成：G1 seed 17 四候选正式 screen 与预留 profile 确认。
8. 已收口：五个选定配置的 seed 17 开发确认完成前两个主折；第一折 18 份表示已导出。后续不补跑冗余开发折，统一由正式三 seed 五折协议承接。
9. 进行中：seeds 17/29/43 只重训五个唯一配置；G2 任务真值必须等 15 个 checkpoint 和 54 份 clean 表示全部完成后再打开。

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
- 完整测试：`352 passed, 8 skipped, 317 warnings`。
- `compileall src scripts tests` 与 `git diff --check`：通过；LFS 和读者术语检查将在本里程碑提交前再次执行。
- 当前 Git 改动中没有 raw snapshot、完整仿真 bundle、checkpoint 或稠密表示；拟入仓内容仅为代码、测试、紧凑清单和审计报告。

## 证据边界

- 鼎新结果写成真实双流弱监督任务证据，不写成人工专家真值。
- 仿真结果写成已知机制下的真值验证和压力测试，不替代真实数据。
- LLM 只用于字段语义归一、规则复核、场景说明和人工复核材料组织。
- negative/mixed 结果必须保留，不能通过追加未计划候选强行制造全面领先。
