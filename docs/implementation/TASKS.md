# Chronaris 当前任务

更新时间：2026-07-11

## 当前长程 goal

在不依赖新增鼎新数据或人工标签的前提下，完成真实弱监督应用任务、模型无关半物理仿真、六方法统一融合表示、Chronaris 连续融合主干、下游评估、锁定实验、图表证据和状态文档闭环。

当前分支：`codex/fixed-data-downstream-evaluation-20260710`。

## 当前里程碑：G6 多随机种子锁定训练与 G7 压力基准

G4.1–G4.2、G5 seed 17 正式筛选和 G6 仿真三随机种子五方法锁定重训已经完成；仿真 15/15 checkpoint、9/9 门禁通过，正式统一表示正在导出。鼎新正式重训在运行时审计后新增 100 ms 方法无关因果时间箱：首个上下文航电/生理事件由 1779/150 点压缩为 300/120 点，Chronaris GPU 单 epoch 为 40.07 秒，旧未合并 checkpoint 已由协议门禁隔离。G7 的 35 场景 G2 压力数据已完成 7/7 审计，压力模型评价等待 clean 表示和 consumer 收口。

### 本轮新增进度

- 选定配置 seed 17 开发确认完成前两个留一视图折；第一折 Chronaris 在 epoch 28 早停、最佳 epoch 20。该冗余开发队列已停止，后续由正式三 seed 五折协议承接。
- 第 1 折六方法 train/validation/outer-test 表示完成 18/18 导出；这只冻结输入，不提前运行 outer-test consumer。
- Chronaris 锁定训练中的连续对齐、物理一致性和因果方向损失已真实参与反向传播，公共自监督损失仍是唯一早停依据。
- 100 ms 公共因果时间箱五方法单 epoch 实跑为 8/8；四个基线 GPU 耗时 8.59–16.46 秒，Chronaris CPU 为 91.86 秒、GPU 为 40.07 秒。鼎新正式三随机种子 run 使用新 ID、全方法单 GPU 串行和逐 epoch 恢复。
- 正式 consumer 使用 validation 固定网格选择 Logistic/Ridge 与 MiniRocket 参数；机动分段使用两层残差因果 TCN、kernel 5、dilation 1/2、patience 6，并可在 GPU 上训练。
- G2 压力扩展包含 7 个单因素的 34 个等级版本和 1 个 mixed-severe 版本；48 条轨迹共 1,680 个观测场景，同轨迹复用相同 observation seed，7/7 验收通过。
- G1→G2 clean 表示、正式 consumer、压力表示、冻结 consumer 复用、退化斜率和轨迹级配对统计的可恢复编排已经实现；按协议等待上游锁定 checkpoint 后再执行。
- 鼎新正式表示与 consumer 编排已实现：75 个 checkpoint 完整后才导出 270 份六方法三角色表示，validation 只负责选参，outer-test 只负责一次锁定评价。
- Chronaris 四项机制消融已接入锁定训练、checkpoint 回载、表示导出和相同下游 consumer；完整模型与消融按 48 条 G2 潜在轨迹配对。
- synthetic-to-real 轨道已实现形状安全的部分参数迁移和无标签鼎新适配；单方法跨 schema 冒烟复制 97.93% 目标编码器元素并通过 8/8 门禁，正式轨道等待仿真三 seed checkpoint 完整。
- 鼎新统一表示和 consumer 已显式继承并校验上游表示族，real-only 与 synthetic-to-real 使用不同 family 字段和独立 run，不会因复用脚本而混入同一结果表。
- 时间偏移/响应时延恢复已实现四方法专用表示与下游探针：G1 train 拟合、G1 validation 选择 Ridge 强度，G2 35 场景只评价；主统计单位固定为 48 条潜在轨迹。
- 仿真端到端微调辅助链路已实现并通过定向测试：五个可训练方法更新完整编码器，朴素同步只更新相同容量任务头；三任务联合损失只由 train 拟合、validation 早停，G2 held-out 只评价，输出独立 `end_to_end_finetuned_v1` 表。
- 论文证据包生成器已实现并通过 8/8 合成输入验收：正式上游完成后自动汇总至少 7 个独立证据角色（含既有 UAB/NASA 公开适配，迁移轨道完成后为 8 个），生成鼎新/仿真六方法主图、压力斜率热图、机制恢复图、消融图、鼎新代表时间线与仿真 oracle 复盘，并输出 7 图可追溯 figure manifest。
- 公共自监督增强已移到 CPU 确定性执行，模型前向和 target tensor 才进入 GPU；该路径完成单 epoch CUDA 实跑，可避开此前 `_feature_age` 的 WSL 小算子 launch failure，同时保持训练配置和增强 realization 不变。
- clean、压力和机制恢复表示导出已支持 baseline CUDA / Chronaris CPU 混合设备；seed 17 完整六方法三角色冒烟导出 18/18、7/7 通过，正式三 seed clean 表示队列正在复用该配置。

### G1–G4.1 已完成

- 六方法生产表示、Chronaris 连续主干、五方法公共预训练和朴素时间同步训练折无监督变换均已可恢复。
- 16 条仿真 train 轨迹的 8/4/4 smoke 完成 5 个训练 checkpoint、18 个折外表示和 72 条线性指标。
- workload 真值只在五个 checkpoint 完成后打开；预训练路径不读取 oracle，仿真 validation/locked_test 路径零命中。
- 删除一个 Chronaris 留出折表示后仅重建该项，重建 SHA-256 与原输出一致；闭环 20/20 验收通过。
- G3b.4 完成时完整测试为 `301 passed, 8 skipped`；G4.1 完成后为 `312 passed, 8 skipped`；G4.2 目标归档为 `314 passed, 8 skipped`；原始上下文绑定后为 `316 passed, 8 skipped`；训练内划分后刷新为 `318 passed, 8 skipped`。
- 16 条仿真训练轨迹各取四个跨状态上下文，形成 64 个样本和 32/16/16 profile 隔离；六方法应用上下文表示共 18 份，恢复 18/18 复用。
- 线性、MiniROCKET 10,000 kernels、两层因果 TCN 与训练折持续时间解码共产生 384 条可计算指标、256 条融合增益和 30 条轨迹级配对统计。
- MiniROCKET 训练折方差过滤、TCN 随机流隔离和组件级恢复均已固化；删除 Chronaris 表示、MiniROCKET、TCN 后只重建目标组件，12/12 验收通过。
- G4.1 紧凑证据位于 `2026-07-11_application-consumer-smoke`，约 15 MB 模型、表示和预测留在被忽略目录；所有指标均为 smoke only。
- G4.2 已生成五折两个任务共 10 个独立目标 archive：分类保留 96 个上下文，原始点中位数生理响应保留 90/93 个完整未来区间，3 个末端候选结构化不可用。
- 10/10 archive 恢复复用，原 snapshot 哈希不变，目标归档 `12/12` 验收通过；本阶段没有模型训练或任务指标。
- 96 个目标上下文中 93 个具有完整 30 秒输入，三个 25.991 秒部分末窗不可用；机动分类/生理响应最终绑定 93/90 个唯一上下文。
- 12 生理 + 955 航电字段采用 78.6 MB 允许字段 CSR 缓存，20 个机动标签源在映射层删除；完整绑定审计 12.34 秒、峰值 767 MB，12/12 通过。
- 五个外层折已形成互斥 inner-train/validation/overlap embargo/outer-test；inner-train/validation 规模为 31/31、31/31、38/14、19/7、38/14，共享航电原始时间区间零重叠，11/11 通过。
- 当前 outer-train 阈值只用于固定配置 smoke；正式候选筛选必须以 inner-train 重新拟合阈值和目标，不能据此提前比较方法。
- 主协议首折已完成五个可训练方法各 31 step、六个 checkpoint 和 18 份三角色表示；18/18 恢复、同角色六方法对齐和 2.5 GB 内存门均通过。
- 完整首折累计训练 213.63 秒、峰值 1967.4 MB，Chronaris 训练 148.73 秒；约 79 MB 重型产物仅位于被忽略目录。本 run 未打开任务目标或计算 outer-test 指标。
- 流式训练与首折表示接入后完整测试为 `323 passed, 8 skipped`。
- 五折总计 30 个 checkpoint、90 份表示，五个子 run 60/60、聚合审计 13/13；所有 archive、checkpoint hash、fit hash、角色样本和公共 schema 已重验。
- 五折五方法累计训练 1066.45 秒，最高峰值 2047.1 MB；约 393 MB 重型产物仅位于被忽略目录，任务目标与 outer-test 指标保持关闭。
- 五折聚合审计接入后完整测试为 `325 passed, 8 skipped`。
- 五折冻结表示 consumer 形成 30 个方法—折组合、60 个组件、1680 条全可计算 smoke 指标和 1120 条双流增益；首次拟合 144.34 秒，恢复 60/60，15/15 通过。
- 五折实际使用 440 个机动分类角色上下文和 425 个生理响应角色上下文；36 MB 重型模型/预测被忽略，真实与仿真指标保持分层。
- consumer 工程冒烟接入后完整测试为 `327 passed, 8 skipped`。
- 五折 inner-train 嵌套目标形成 10 个确定性 archive，机动分类/生理响应可用角色数为 440/425；75 个机动类别与 51 个高响应标签相对工程冒烟口径改变，10/10 通过。
- 三个时间块 validation 不含低机动类；保留真实分布并固定三类 macro-F1 合同，禁止通过调阈值补类。
- 嵌套目标与固定类别指标合同接入后完整测试为 `328 passed, 8 skipped`。
- 嵌套 validation-only consumer 形成 30 个方法—折 bundle、840 条全可计算指标和 560 条增益；恢复 60/60，outer-test 零指标，12/12 通过。

### 当前输入与不可变边界

- 表示合同固定为 `[sample,96,64]`，G4 consumer 不得回读原始双流为某个方法追加特征。
- 鼎新 target 来源固定为 G1 审计中的训练折阈值、字段角色和 30 秒上下文；标签源航电字段仍禁止进入机动分类输入。
- 仿真 target 只允许读取 `ground_truth.npz` 中预先列入任务合同的 `true_time_s`、`workload`、`maneuver_state`、`maneuver_type` 和事件边界；物理残差、生成参数和真实时延不进入应用任务 consumer。
- consumer 只拟合 representation train role；validation 用于固定网格选择，held-out 每个锁定配置只评价一次。
- G4.1 复用 `2026-07-11_common-pretraining-loop-smoke` 的六个 checkpoint，但为 64 个跨状态上下文重新导出 18 份表示；没有读取仿真 validation 或 locked_test，也不产生模型排名。
- G4.2 必须使用鼎新 snapshot 原始点重新构造外层折输入，不能把仿真输入维数的 checkpoint 直接套到鼎新，也不能回用可能含标签源信息的历史对齐投影。

### 子任务 G4.2-a：鼎新独立 target archive（已完成）

#### 鼎新机动强度弱监督分类

1. 对每个 leave-one-view-out 外层折，读取 G1 `fold_label_thresholds.csv` 的训练折阈值和 `fold_task_labels.csv` 的上下文标签；不在 G4 重新计算全局分位数。
2. 目标 archive 记录 context、view、sortie、fold、label、阈值 hash、标签源字段 hash 和 `weak_supervision=true`。
3. 类别不足的训练折写 unavailable；禁止移动阈值、合并测试类或用 held-out 分布补齐。
4. 表示样本必须与 target context 一一匹配；漏样本、重复样本和跨 view checkpoint 直接失败。

#### 鼎新机动诱发生理响应回归

1. 复用训练折拟合的生理基线和字段有效性，目标为未来 5 秒相对当前上下文末端的多指标变化幅度。
2. 同时生成连续响应值和训练折高响应阈值，但 G4 主回归 consumer 只使用连续值；高响应识别留作分类辅助表。
3. 输入窗口截至 context end，目标区间为其后 0–5 秒；未来生理点绝不进入表示或归一化器。
4. 目标无有效生理字段时按样本 unavailable，不以零变化替代。

执行结果补充：正式目标使用冻结原始点的当前/未来 5 秒窗口中位数，不再使用 G1 的窗口均值兼容统计；原始点只到 181 秒，三个 `context_end_0035` 不足完整未来区间，因此正式可用上下文为 90 个。

#### 仿真负荷与机动状态（G4.1 已完成）

1. 负荷提前评估：使用输入结束后 0–5 秒 workload 均值，训练折三分位形成三类，并保留连续回归值。
2. 机动状态分段：把 oracle `maneuver_state` 重采样到 96 点查询轴，固定状态为稳态、进入、持续、退出/恢复；同一重采样规则供六方法共享。
3. 边界真值由状态变化点独立生成，记录原始秒数和查询点索引；consumer 不读取 event duration 或 test transition 统计。
4. target archive 与表示目录分离，写入允许字段列表；任一未声明 oracle 字段访问由测试拦截。

### 子任务 G4.1-b：统一线性与 MiniRocket consumer（已完成）

1. 保留已完成的 Logistic/Ridge 线性探针，提取为正式 `FrozenRepresentationConsumer` 接口。
2. MiniRocket 输入统一转为 `[N,64,96]`，固定 `aeon==1.5.0`、10,000 kernels、seed 17；变换器只在 representation train role 拟合。
3. 分类变换后使用 Logistic/RidgeClassifier 固定网格，回归使用 Ridge；网格、标准化和最大迭代数按任务固定，不按方法改变。
4. 内层 validation 选择超参数后冻结；同一任务六方法共享被选中的 consumer 配置，不能让每种表示选择不同容量。
5. 记录 transform hash、fit sample hash、kernel 数、训练秒数、峰值内存和输出维数；恢复时逐项校验。

### 子任务 G4.1-c：TCN emission 与持续时间约束 Viterbi（已完成）

1. TCN 固定两层、64 channels、kernel size 3、dilation 1/2、dropout 0.1，使用严格因果左 padding；输出每个查询点的状态 emission logits。
2. 训练 loss 使用有效查询 mask 的 class-balanced cross entropy；类别权重只由训练 role 统计。
3. 转移概率、初始概率和每类持续时间上下限只在训练 role 估计；加 Laplace 1 平滑并记录计数。
4. 自研 duration-constrained Viterbi 不依赖 `hmmlearn`；validation/held-out 标签不能参与 transition、duration 或后处理阈值拟合。
5. 同时保留 raw TCN 与 TCN+Viterbi 两组结果，以区分表示/发射模型能力和时序先验增益。

### 子任务 G4.1-d：任务指标与方向合同（已完成）

| 任务 | 主指标 | 辅助指标 | 指标方向 |
| --- | --- | --- | --- |
| 鼎新机动分类 | macro-F1 | balanced accuracy、macro-AUPRC、最差折 | 越高越好 |
| 鼎新生理响应 | RMSE | MAE、Spearman、高响应 AUPRC | 误差越低；相关越高 |
| 仿真负荷分类/回归 | macro-AUPRC、RMSE | macro-F1、MAE、Spearman | 分项方向 |
| 仿真机动分段 | segmental F1 | frame macro-F1、boundary F1、edit、检测延迟 | F1/edit 越高；延迟越低 |

具体要求：

1. `metric_long.csv` 每行包含 dataset、task、method、consumer、seed、fold、role、metric、value、direction 和 status。
2. boundary F1 固定 ±1 秒、±2 秒两个容差；segmental F1 固定 IoU 0.10/0.25/0.50；不根据结果选择容差。
3. 校准输出 Brier score 与 ECE 10 bins；小样本空 bin 跳过但记录有效 bin 数。
4. regression/分类/分段 unavailable 分开记录，不能用另一个任务指标补位。

### 子任务 G4.1-e：双流增益、压力斜率和配对统计（接口已完成）

1. 对每个 fusion 方法计算 `Score_fusion - max(Score_physiology, Score_vehicle)`；误差型指标先转换方向后再计算。
2. 六方法必须使用相同 held-out trajectory/context 配对；缺任一样本时整组配对统计 unavailable。
3. 仿真以 trajectory 为独立单位，鼎新以 view/sortie 外层折为独立单位；禁止把 96 个查询点当作独立样本做显著性检验。
4. 实现 trajectory-level paired bootstrap 和 exact/permutation test；smoke 只验证接口，正式 locked confirmation 才报告区间和 p 值。
5. stress slope 只接收锁定 checkpoint 的成对场景结果；G4 consumer 开发期不运行 G2 stress。

### 子任务 G4.1-f：仿真应用 consumer smoke（已完成）

已生成 `docs/artifacts/runs/2026-07-11_application-consumer-smoke/`，沿用 16 条仿真 train 轨迹的六个 checkpoint，并为四个时间位置重新导出 18 份冻结表示：

1. 从 ground truth 生成 workload 分类/回归和 96 点机动状态 target archive，验证允许字段审计。
2. 六方法运行线性、MiniRocket 分类/回归、TCN 和 TCN+Viterbi；小样本指标全部标记 `smoke_only=true`。
3. 所有 consumer 使用同一 train/validation/held-out role，完成 fit hash、恢复和方法不变超参数审计。
4. 删除 Chronaris 留出表示、MiniROCKET transform 和 TCN checkpoint 后各只重建对应项；未删除组件哈希保持不变，预测哈希一致。
5. 生成完整 `metric_long`、fusion gain、边界/分段指标和配对接口输出；不进入候选选择。

### 本里程碑预期产物

- `task_target_manifest.json`、`target_availability.csv`
- `consumer_protocol.json`、`consumer_fit_manifest.json`
- `minirocket_status.csv`、`tcn_training_status.csv`、`viterbi_parameter_manifest.json`
- `metric_long.csv`、`fold_metrics.csv`、`segment_metrics.csv`
- `fusion_gain.csv`、`paired_statistics.csv`
- `acceptance_checks.csv`
- `report.md`、`claim_boundary.md`、`progress.json`、`run.log`、`resume_command.txt` 和 `evidence_manifest.json`

transform、consumer checkpoint、emission、逐样本预测与逐点状态序列已写入被忽略的 `artifacts/application_evaluation/2026-07-11_application-consumer-smoke/`；紧凑 run 为 12/12 验收通过。

### G4.1 验收（已通过）

- 三类 target archive 的来源、时间边界、训练折阈值和允许 oracle 字段全部通过 fail-closed 测试。
- MiniRocket 10,000 kernels、TCN 架构和 Viterbi 参数在六方法间完全一致，fit lineage 只含 train role。
- raw TCN 与 Viterbi 后处理都产生 frame、segment、boundary 和 edit 指标；test 标签不进入 transition/duration 拟合。
- classification、regression、segmentation、校准、fusion gain 和配对统计接口均输出方向明确的结构化结果。
- consumer checkpoint/transform 的完整恢复和单项删除重建通过，表示 checkpoint hash 保持不变。
- 完整测试、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查通过。

### G4.2 当前实现顺序

1. 已完成：生成鼎新机动分类与生理响应两个独立目标 archive，逐样本记录 G1 阈值、原始点中位数、snapshot 与外层折 hash。
2. 已完成：从固定 snapshot 构造与目标 context 一一对应的 30 秒原始异步双流，审计标签源字段排除、部分末窗和未来区间隔离。
3. 已完成：在 outer-train group 内建立防重叠 inner-train/validation，固化 leave-one-view-out 主协议和 leave-one-sortie-out 辅助协议。
4. 已完成：五折五方法公共预训练 checkpoint、朴素时间同步无监督变换和六方法 train/validation/test 表示全部生成。
5. 已完成：固定线性/MiniROCKET consumer 的真实数据 smoke，真实与仿真 metric root 分离。
6. 已完成：以 inner-train 重拟合机动阈值、生理字段尺度与高响应阈值，生成嵌套目标。
7. 已完成：只在 validation 复跑固定 consumer，并修正固定类别指标合同；outer-test 继续关闭。
8. 当前：冻结 G4 协议，运行 seed 17 四候选 screen；只使用仿真开发 validation 和鼎新 validation，不读取任何 outer-test/locked-test。

### G5 已完成的筛选基础设施

- A–D 候选已成为代码中的冻结配置：`64/1e-3/0.1`、`64/3e-4/0.1`、`32/1e-3/0.1`、`64/1e-3/0.2`；共同使用 2 层、4 heads 和 64 维输出合同。
- 候选 C 只缩小内部隐层，单流、MulT、ContiFormer 和 Chronaris 的对外表示仍为 `[N,96,64]`；旧 64 维 checkpoint 可严格加载。
- G1 只读取 96 个 train 与 24 个 validation profile 的 `raw_dual_stream.npz`；其中 23 个 validation 用于排序、1 个保留作开发确认，封存测试不进入批次。
- 验证增强固定为 seed 17、epoch 0；早停只使用遮挡重构、短期预测和时延判别三项公共损失，Chronaris 专属诊断不进入分数。
- 候选内逐 epoch 保存 `last.pt`，恢复时加载编码器、公共 head、优化器、最佳分数和 patience，从下一 epoch 继续；候选完成后直接复用 `best.pt`。
- 四候选逐损失做方法内 min-max，常量损失项归一化为 0；总分并列时按更小参数量、候选字母序确定唯一配置。
- 20 候选单 epoch smoke 已完成，耗时 5 分 03 秒、峰值内存 4.34 GB、重型 checkpoint 约 114 MB，6/6 验收通过。该结果只验证长跑链路，不作为正式候选结论。
- 本轮正式训练、鼎新下游、机制恢复与消融编排收口后完整测试为 `360 passed, 8 skipped, 317 warnings`，`compileall`、Ruff 与 `git diff --check` 通过。
- 正式 50 epoch/patience 8 筛选完成；Chronaris、MulT、ContiFormer、生理单流选择 A，航电单流选择 C。预留的第 24 个 G1 validation profile 只用于五个选定 checkpoint 的一次开发确认，不参与重新排序。
- Chronaris 参考轴采样已由逐样本重放改为数学等价的向量化查询，输出、审计与梯度对照通过；CPU 单 epoch 从约 60 秒降至 17.07 秒。RTX 4090 同批次为 49.87 秒，因此该主干正式 screen 使用 CPU，CUDA 支持保留给更适合并行的后续训练。

## 已锁定规范

- [固定数据证据策略](../requirements/foundation/fixed-data-evidence-strategy.md)
- [下游应用评估任务协议](../requirements/downstream-evaluation-spec.md)
- [仿真基准规格](../requirements/synthetic-benchmark-spec.md)
- [双流与融合表示合同](../requirements/model-contracts/application-fusion-stream-contract.md)
- [详细实施计划](notes/fixed-data-downstream-evaluation-2026-07-10.md)
- [长程运行手册](notes/fixed-data-downstream-evaluation-runbook-2026-07-10.md)

## 后续验收门

### G5：screen

- 已完成：seed 17、每个可训练方法四候选。
- 已完成：只用 G1 公共自监督验证损失排序，并在预留 profile 上确认选定配置。
- 已保持：未读取 G2 locked test，未使用结构诊断指标选择候选。
- 下一步：只重训每方法唯一选定配置并进入鼎新五折 validation；禁止让 20 个候选反复读取鼎新 validation。

### G6：locked confirmation

- 进行中：seeds 17、29、43 的五个唯一配置重训；仿真长基线在独占 GPU 重复失败后迁移到 CPU，Chronaris 仍使用同批基准更快的 CPU。
- 已完成：仿真 seeds 17/29/43 五方法均完成 50 epoch，15 个训练单元与 9/9 门禁全部通过；正式 clean 表示导出正在运行。
- 排队执行：鼎新 75 个选定配置锁定重训使用 `2026-07-12_dingxin-locked-pretraining-coalesced` 新根；CPU 构造增强、全方法单进程 GPU 训练。任何缺少 `model_input_contract.json` 的旧 checkpoint 均 fail closed，不与新输入混用。
- 待上游完成后自动执行：G1→G2 clean 三随机种子六方法表示、validation 选参、锁定 held-out 指标和 48 轨迹配对统计。
- 已实现待队列门禁打开：鼎新 270 份统一表示、主/辅助 split 正式 consumer，以及 Chronaris 四项固定消融的训练—表示—consumer 链路。
- 进行中：Chronaris 四项固定消融 × seeds 17/29/43 已启动 CPU 锁定重训；不占用当前唯一 CUDA 正式队列。
- 已验证：无物理约束变体完成 G1 train/validation 与 G2 held-out 三角色表示导出，3/3 输出、5/5 验收通过；正式消融表示等待 12 个变体 checkpoint。
- 已实现待上游门禁：synthetic-to-real 三 seed 五折无标签适配及其统一表示/consumer 复用。
- 已实现待上游门禁：端到端微调辅助表；三 seed 六方法、独立表示族和 48 轨迹统计均已编排，不得替代冻结表示主结果。

### G7：stress 与论文证据包

- 已完成：48 条 G2 轨迹 × 35 个严格成对观测版本，生成器不接收方法名，7/7 审计通过。
- 待锁定 checkpoint：跑全部单因素表示、冻结 consumer、退化斜率和 mixed-severe。
- 待锁定 checkpoint：导出 G1 六观测场景四方法表示，并在 G2 压力表示完成后运行绝对时钟偏移和主生理响应时延恢复。
- 结构诊断只放附录。
- 已实现待上游门禁：输出中文论文图表、证据矩阵、claim boundary 和新协议快照候选；正式运行后还需逐图抽查中文字体、长标签和数值可读性。

## 方法和任务固定项

### 方法

- 生理单流。
- 航电单流。
- 朴素时间同步。
- MulT。
- ContiFormer。
- Chronaris。

### 任务

- 机动强度弱监督分类。
- 机动诱发生理响应预测。
- 仿真负荷提前评估。
- 仿真机动状态分段。
- 时间偏移与响应时延恢复。

### 主下游算法

- Logistic/Ridge 线性探针。
- MiniRocket 10,000 kernels。
- 两层 64-channel TCN + 自研持续时间约束 Viterbi。

### 正式随机种子

- 开发：17。
- 锁定确认：17、29、43。

## 运行和产物规则

- Python：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。
- RTX 4090 正式训练最多单 CUDA 进程；独占训练重复失败的长队列必须记录设备历史并迁移 CPU，不再无限同配置重试。短 TCN/微调可在张量自检通过后单独使用 GPU，重复失败则同样迁移。
- 紧凑可引用产物：`docs/artifacts/runs/YYYY-MM-DD_intent/`。
- raw snapshot、dense bundle、checkpoint 和逐样本预测：`artifacts/application_evaluation/`，禁止入仓。
- 每个正式 run 必须有 progress、resume、protocol、fold metrics、claim boundary 和 evidence manifest。
- 每通过一个验收门，同步 `STATE.md`、本文件和 `ARTIFACTS.md`。

## 当前禁止事项

- 不整体合并 `implement/fusion-stream-structure-20260707`。
- 不把旧 E3 run 结果写入论文主表。
- 不继续围绕 `chr_v2_residual_delta_h64` 做无边界调参。
- 不让仿真器读取方法名称或评价结果。
- 不把仿真、公开数据和鼎新真实指标混成单一平均分。
- 不在元数据不足时用所有航电字段构造机动标签。
- 不修改既有 confirmed metrics，直到新的 locked confirmation 和 review 完成。

## 当前验证门

当前 G4.2 提交前必须通过：

1. 已通过：两项鼎新 target archive 的时间边界、训练折阈值、样本覆盖、字段 lineage 和 unavailable 测试。
2. 已通过：机动标签源字段在原始映射中零命中，未来生理点在输入中零命中，三个部分末窗结构化不可用。
3. 已通过：leave-one-view-out 与 leave-one-sortie-out 的 inner-train/validation/outer-test group 或时间块无重叠；后续所有变换必须只用 inner-train。
4. 已通过：五折 target、原始点 context 与表示 sample ID 一一对应；漏样本、重复样本、跨折 checkpoint 直接失败。
5. 至少一个真实外层折完成六方法表示和固定 consumer smoke，所有鼎新结果继续标记弱监督并与仿真指标分层。
6. G1–G4.1 聚焦测试保持通过，并运行完整 `pytest`、`compileall`、`git diff --check`、术语、密钥、LFS 和重型产物忽略检查。
