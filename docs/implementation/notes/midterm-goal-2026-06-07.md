# Chronaris 中期答辩前目标

更新时间：2026-06-07

> 2026-06-07 同日补记：本文件保留为“中期前最小收敛”历史判断。当前事实源已经更新到 [../TASKS.md](../TASKS.md) 和 [../../STATE.md](../../STATE.md)：`main/origin/main` 已同步到 `57ca739`，`Phase D/E/F` 已进入二轮真实资产闭环；中期前策略也已经从“最小收敛、不扩公开线”升级为“P10 evidence runner + 五项主动证据”。因此，本文件第 5-7 节中关于“不建议继续扩大 UAB/NASA 搜索”“把刚体/语义/runtime 留到中期后”的判断只作为历史快照，不再作为当前执行入口。

## 1. 当前结论

当前仓库已经不是“最小原型刚起步”状态，而是已经进入中期前的收敛阶段。

按 `docs`、源码、git 记录和选题报告中的“先时序对齐，后语义融合”路线综合判断：

- `阶段 E / F / G(min) / H` 已完成真实链路收口，继续作为历史基线和阶段 I 输入依赖保留。
- `阶段 I Phase 0/1/2/3` 的公开数据基准历史收口已完成。
- `chronaris_opt` 已作为当前鼎新私有代理任务基准主线固化，T1/T2/T3 均为“私有代理任务证据”，不能写成人工真值论文任务。
- “公开数据支撑线已收口”已经固化；UAB `target_prior_median` 仍只能写成“公开数据适配器 / 校准证据”，不能写成双流连续对齐和因果融合模块本体的直接胜利。
- 当前最新中期证据入口是 `docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`，其中私有证据、公开证据、支撑性消融、关键窗口锚定、离线演示图件已经整编，并切换到最新 private benchmark summary。
- 当前 git 顶端提交 `HEAD=2055dec` 已提交到 `origin/main`，其内容主要覆盖阶段 I 论文主线重构 A/B 两阶段：主线边界校准、统一骨干训练入口、阶段 H 固定 checkpoint 推理导出、文档索引收敛。
- 当前工作区还有未提交改动，主要是阶段 I 论文主线重构 C 阶段及其真实产物：统一任务头、任务监督损失与因果融合损失、`risk_proxy / workload_proxy / event_replay_tag` 三类论文任务弱标签构造器、`stage_i_multitask_train` 联合训练入口、真实 Stage H multitask 闭环产物，以及私有证据 summary 中“代理任务证据 / 论文弱标签任务证据”的分层。

本轮已用指定解释器验证 C 阶段最小合约：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_multitask_train tests.test_stage_i_private_optimization tests.test_alignment_model_losses
```

结果：`Ran 10 tests in 5.827s`，`OK (skipped=2)`。测试中的 `ConstantInputWarning` 来自合成样本常量输入的 Spearman 计算，不改变当前合约判断。

## 2. 待完成术语说明

上一版文档里“刚体物理、语义事件融合、运行时推理”三个词写得太凝练。按选题报告原文口径，它们分别对应下面三件事：

1. 基于飞行刚体运动规律的物理一致性约束补强
   - 选题报告中的对应表述：航电流遵循刚体运动规律，拟引入飞行力学中的六自由度运动方程，并把物理一致性正则化项嵌入损失函数。
   - 代码上要做的事：把现有速度/加速度、姿态/角速度、平滑性、包络线等弱物理约束，整理成可选择、可诊断、可单测的物理约束族；让报告能说明哪些字段真正参与了刚体运动残差，哪些字段缺失时只能退回弱约束。
   - 它不是另起一个新模型，而是把阶段 F 的“物理一致性约束”从当前最小可用版本补强到更贴近选题报告的版本。
2. 基于语义查询与非对称因果掩码的事件级融合
   - 选题报告中的对应表述：从连续潜态流中提炼具备战术含义的事件级特征，设计语义查询向量与非对称因果掩码，强制遵循“外部环境影响人体状态”的单向因果规律。
   - 代码上要做的事：把当前阶段 G(min) 的时间步注意力，升级为能输出“机动事件 / 生理状态”等语义查询、事件标记向量、事件级归因摘要的融合模块。
   - 它不是普通注意力可视化，而是为了回答“哪些历史航电事件导致了当前生理状态变化”。
3. 离线飞参记录或在线传感器流的推理管线
   - 选题报告中的对应表述：设计能够处理高通量并发数据的推理引擎，实现对离线飞参记录或在线传感器流的实时解析，并产出结构化、富含语义的融合特征矩阵。
   - 代码上要做的事：新增流式窗口缓存、checkpoint 推理、预测输出和解释输出入口；`runtime_demo.py` 继续作为离线展示工具，不能把它直接写成实时推理引擎。
   - 它的目标是让系统能按窗口连续输出风险/负荷/事件解释，而不只是读已有报告资产。

## 3. Git 记录对应的推进脉络

近期主线提交可以这样理解：

- `9399977 feat: finish stage i and deep comparison assets`：阶段 I 历史收口、UAB/NASA、阶段 H case、深度基线资产落地。
- `a5dea37 feat: smoke test for chroaris with stage i`：阶段 H 全窗口 clean 资产与阶段 I 私有代理任务基准雏形落地。
- `a3c939e Promote chronaris_opt mainline and add Stage I public opt`：`chronaris_opt` 私有主线 package 固化，并启动 public opt。
- `fef794c fix: use gpu cuda to accelerate`：关键阶段 E/G/H/I 路线补 GPU 设备选择合约。
- `5bc253c`、`43a74a4`、`e1aa55f`：UAB/NASA public opt、torch/GPU、robust-prior 和 public mainline 逐步收敛。
- `579eb4c feat: 中期前变更闭环`：中期证据包、图件、离线演示、关键窗口锚定、支撑性消融入口整编。
- `2055dec fix: push forward for fixing stage a and b`：阶段 I 论文主线重构 A/B 的代码、计划文档、索引和测试回写。

因此，git 已提交历史停在 A/B 两阶段收敛；C 阶段当前在工作区通过了最小测试，但还没有进入 git 历史。

## 4. 编码层面还需要做什么

### P0：先冻结当前阶段 I 论文主线重构 C 阶段工作区

目标：把已通过测试的 C 阶段从“工作区改动”变成可追溯的主线状态。

必须完成：

1. 复查当前未提交文件，确认没有把用户临时修改混入 C 阶段提交。
2. 保留新增文件：
   - `src/chronaris/models/alignment/task_heads.py`
   - `src/chronaris/dataset/stage_i_real_task_builders.py`
   - `src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`
   - `tests/test_stage_i_multitask_train.py`
3. 保留并复查已修改文件中的 C 阶段合约：
   - `src/chronaris/models/alignment/losses.py`
   - `src/chronaris/pipelines/stage_i/stage_i_private_benchmark*.py`
   - `docs/implementation/notes/*`
   - `docs/artifacts/stage_i/README.md`
   - `docs/README.md`
   - `AGENTS.md`
4. 补跑：
   - `tests.test_stage_i_multitask_train`
   - `tests.test_stage_i_private_optimization`
   - `tests.test_alignment_model_losses`
5. 通过后再提交。

### P1：已补一条真实资产上的 C 阶段联合训练证据

当前 C 阶段已经不只停留在合成样本 / 私有资产形态的最小冒烟测试，而是已经补出一条可引用的真实资产运行记录。

建议下一步优先做：

1. 已用现有阶段 H 全窗口 E/F clean 资产构造 `risk_proxy / workload_proxy / event_replay_tag`。
2. 已运行 `stage_i_multitask_train`，输出：
   - `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
   - `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
   - `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
3. 已补 `docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`，并明确写成“论文任务弱标签证据”。
4. 当前这条证据按“主线闭环证据”使用，不宣称人工真值最优。

### P2：补强基于飞行刚体运动规律的物理一致性约束

这是当前最值得写进论文方法体的下一段代码工作。

目标：

- 把已有“弱物理约束族”升级为可命名、可诊断、可测试的“刚体运动物理约束族”。它对应选题报告中“航电流近似满足飞行刚体动力学方程，物理一致性正则化项嵌入损失函数”的表述。

建议最小落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `src/chronaris/models/alignment/physics_residuals.py`
- `src/chronaris/models/alignment/physics.py`
- `src/chronaris/models/alignment/losses.py`
- `tests/test_alignment_model_losses.py`

退出条件：

- 旧 `full` / `weak_physics` 配置不回归。
- 新的“刚体运动物理约束族”有单测和分项损失明细。
- 报告能说明当前架次哪些刚体运动残差被启用，哪些因为字段缺失只能退回弱约束。

### P3：补强基于语义查询与非对称因果掩码的事件级融合

目标：

- 从当前 `G(min)` 的时间步注意力，升级到“语义查询向量 + 战术事件标记向量 + 事件级归因摘要”。它对应选题报告中“从连续潜态流中提取战术事件，并进行单向因果融合”的表述。

建议最小落点：

- `src/chronaris/models/fusion/semantic_event.py`
- `src/chronaris/models/fusion/causal.py`
- `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
- `src/chronaris/pipelines/stage_i/stage_i_support.py`
- `tests/test_stage_i_support.py`

退出条件：

- 可以输出事件标记向量、语义查询到事件的注意力、事件归因摘要。
- 支撑报告能区分“时间步注意力”和“事件级归因”。

### P4：补离线飞参记录或在线传感器流的推理管线

目标：

- 不再把 `runtime_demo.py` 当实时推理引擎，而是新增真正的流式或准流式推理入口。它对应选题报告中“处理高通量并发数据、解析离线飞参记录或在线传感器流、产出融合特征矩阵”的工程目标。

建议最小落点：

- `src/chronaris/dataset/streaming_windows.py`
- `src/chronaris/serving/runtime_inference.py`
- `scripts/run_stage_i_runtime_inference.py`
- `tests/test_runtime_inference.py`

退出条件：

- 模拟流或本地回放流可以增量产出窗口、风险/负荷预测、注意力解释和事件归因解释。

## 5. 实验层面还需要做什么

中期前不建议继续扩大 UAB/NASA 搜索。当前更重要的事已经从“补齐 C 阶段闭环”切到“提交收口 + 把剩余方法体工作留到中期后”。

优先实验：

1. 刚体运动物理约束冒烟测试 / 消融
   - 目的：证明阶段 F 方法体已从弱物理约束向刚体运动物理约束前进。
   - 输出：物理约束分项损失、字段覆盖诊断、E/F/新物理约束对照。
2. 语义事件融合支撑性冒烟测试
   - 目的：补答辩中最容易展示的事件解释图/表。
   - 输出：事件归因、关键窗口解释、支撑报告补充。
3. runtime inference 最小闭环
   - 目的：把“离线演示”与“真正推理入口”分开。
   - 输出：流式窗口缓存、checkpoint 推理、解释输出脚本与测试。

不建议做：

- 继续扩大 CPU-heavy `sklearn` 或 UAB torch 候选搜索。
- 把 NASA/UAB 公开数据适配器结果改写成论文双流本体闭环。
- 把 `20251110_单01_ACT-2_涛_J20_26#01` 当作双流 Stage H view。
- 重写上游接收器或原始大文件入仓链路。

## 6. 中期答辩前最小收敛定义

如果时间只够做最小闭环，按下面顺序收敛：

1. 提交当前 C 阶段工作区。
2. 直接引用最新联合训练确认与 private benchmark 分层资产。
3. 直接引用最新中期证据包。
4. 准备答辩叙事：
   - 历史收口：E/F/G/H + 阶段 I Phase 0/1/2/3。
   - 私有主线：`chronaris_opt` 是私有代理任务证据。
   - 公开支撑：公开数据支撑线是适配器/校准证据。
   - 当前论文主线：A/B/C 三阶段已接上统一骨干、固定 checkpoint 推理导出、统一任务头、论文弱标签任务。
   - 未完成但方向明确：基于飞行刚体运动规律的物理一致性约束补强、基于语义查询与非对称因果掩码的事件级融合、离线飞参记录或在线传感器流的推理管线。

## 7. 当前一句话状态

当前项目已经具备中期答辩所需的历史实验资产、Phase C 真实联合训练证据、private/thesis 分层资产和新版中期证据包；真正剩下的主要是把这些工作区改动固化进 git，并在中期后把研发重心切到“刚体运动物理约束补强”“语义事件融合补强”和“runtime inference”，而不是继续扩旧公开数据基准。
