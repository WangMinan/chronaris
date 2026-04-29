# Coding Roadmap

更新时间：2026-04-29

## 1. 目的

这份文档用于回答两个问题：

1. `chronaris` 现在编码推进到哪一步了
2. 下一步应该先做什么，而不是同时做所有事

它不是论文计划书，也不是最终架构说明，而是面向实际编码推进的执行路线图。

## 2. 当前总目标

围绕已经入库的航空人机多源数据，逐步实现下面这条代码链路：

1. 读取 InfluxDB / MySQL 中的人机多源数据和元信息
2. 组织为统一 schema
3. 完成时间基准校正与窗口构建
4. 实现双流连续对齐
5. 实现因果约束跨模态融合
6. 输出标准化融合特征
7. 面向典型任务做对比、消融和案例验证

## 3. 当前阶段判断

当前仓库处于：

`阶段 A/B/C 已完成，阶段 E0 已完成 preview 路径，阶段 E/F/G(min) 已完成，阶段 H 已完成收口，阶段 I 已启动（Phase 0 + Phase 1 + Phase 2 已跑通，Phase 3 真实收口进行中）`

更具体地说：

- 已完成仓库初始化、分层、最小设计文档、统一对象骨架、时间语义约束和 v1 数据构建骨架
- 已开始真实 InfluxDB / MySQL 接入准备
- 已开始基于真实架次的 preview 联调
- 已开始单架次时间覆盖与窗口策略核验
- 已完成 overlap-focused E0 最小实验输入适配
- 已完成阶段 E 的执行计划固化与最小训练/验证切分策略实现
- 已完成实验室服务器 WSL Ubuntu 22.04 + RTX 4090 环境迁移与依赖验证
- 已完成阶段 E 最小训练闭环、`relative_mse` 真实回归与样本级诊断产物导出
- 已完成阶段 E 收口：`none / zscore_train` 对照实跑、阈值模板判定与 checkpoint 导出
- 阶段 E 收口事实统一沉淀于 `docs/planning/stage-e-closure-2026-04-21.md`
- 已完成阶段 F 完整物理约束族：RealBus 字段语义映射、组件级 physics breakdown、`E baseline` vs `E+F(full)` 真实对比实跑
- 阶段 F 收口事实统一沉淀于 `docs/planning/stage-f-closure-2026-04-22.md`
- 已完成阶段 G 最小因果融合原型：`F baseline` vs `F+G(min)` 真实对比实跑、非对称因果注意力诊断与事件贡献导出
- 阶段 G 收口事实统一沉淀于 `docs/planning/stage-g-closure-2026-04-22.md`
- 已完成阶段 H 标准化特征导出收口：双流 `validation` profile 三 view 导出、`load_stage_h_feature_run()` 下游读取验证、`20251110...` vehicle-only partial bundle 真实构建
- 阶段 H 收口事实统一沉淀于 `docs/planning/stage-h-closure-2026-04-27.md`
- 已完成阶段 I `Phase 0 + Phase 1`：UAB `87` session manifest、`416` 维 session 特征、双轨 baseline、主报告与机器资产导出
- 已完成阶段 I `Phase 2`：`3` 个真实双流 view case study、`4` 条 bundle-only 消融、`WARN` view 主线解释与同 sortie 双 pilot 对比
- 阶段 I `Phase 3` 代码入口已接入：window-level contract、`UAB window_v2`、`NASA CSM` attention-state、通用 baseline runner、`run_stage_i_phase3.py`
- 阶段 I `Phase 3` synthetic / live-compatible 测试已并入 `test_stage_i_pipeline.py`，`python -m unittest discover -s tests -p 'test_*.py'` 当前通过
- 阶段 I `Phase 3` 真实长跑仍在进行中；在引入最新 `EEG/ECG overlap` 边界修正后，尚未生成新的 closure 级主报告
- 阶段 I 当前事实统一沉淀于 `docs/planning/stage-i-data-plan-2026-04-29.md` 与 `docs/reports/stage-i-uab-baseline-2026-04-29.md`

## 4. 阶段拆解

### 阶段 A：仓库初始化与最小设计

目标：

- 固化仓库边界
- 固化分层
- 固化统一对象和最小 pipeline 设计

状态：

- 已完成

已完成项：

- 根级协作说明与范围文档
- `access / schema / dataset / pipelines` 分层骨架
- `pipeline-v1` 设计
- 生理/飞机时间精度约束
- `flight_task.flight_date` 语义固化
- 跨日时间补全预留

主要产物：

- [AGENTS.md](../../AGENTS.md)
- [architecture.md](../foundation/architecture.md)
- [data-contracts.md](../foundation/data-contracts.md)
- [pipeline-v1.md](../foundation/pipeline-v1.md)
- [models.py](../../src/chronaris/schema/models.py)
- [temporal.py](../../src/chronaris/access/temporal.py)

### 阶段 B：真实元信息与数据访问接入

目标：

- 打通真实 MySQL 元信息读取
- 打通真实 InfluxDB 时序数据读取
- 用真实架次跑通第一条最小数据链路

状态：

- 已完成 preview 路径

优先级：

- 最高

具体任务：

1. 实现 `MySQL metadata reader`
   当前状态：已完成
2. 先覆盖 `flight_task`
   当前状态：已完成
3. 再覆盖总线相关元信息
   当前状态：已完成
4. 实现 `Influx point reader`
   当前状态：已完成基础通用层
5. 基于单架次读取生理流和飞机流
   当前状态：已完成 preview 路径
6. 用真实数据构造 `SortieBundle`
   当前状态：已完成 preview 路径
7. 跑通 `DatasetPipelineV1`
   当前状态：已完成 preview 路径

当前已验证：

- 真实 MySQL 已可读出目标架次 `flight_task + flight_batch`
- 真实 MySQL 已可读出目标架次对应的 `storage_data_analysis / access_rule_detail / detail / structure`
- 真实 Influx 已可读出目标总线 measurement，并成功聚合回 `RawPoint`
- 已确认生理 bucket 与典型 tags / measurement 形状
- 已可基于真实 reader 构造目标架次的 preview `SortieBundle`
- 已可基于 preview bundle 生成真实 `DatasetBuildResult`

当前已知限制：

- 全量生理流直接读取仍然偏重，当前更适合先走 measurement 白名单 + 限流 preview 路径
- 以 5 秒窗口构建时，当前 preview 结果窗口数为 0
- 以 30 分钟窗口构建时，当前 preview 结果可生成 1 个联合窗口
- 上述限制已在阶段 C 通过 full-coverage probe 与 overlap-focused 路径核验，不再是当前主阻塞

建议首个联调对象：

- `20251005_四01_ACT-4_云_J20_22#01-2100448-10033`

退出条件：

- 能从真实库中拿到该架次的生理点、飞机点和元信息
- 能生成 `DatasetBuildResult`
- 能输出基础摘要和窗口统计

### 阶段 C：统一样本组织与数据核验

目标：

- 把“能读出来”提升为“数据可用”

状态：

- 已完成（以真实重叠核验为准）

具体任务：

1. 做字段映射一致性检查
2. 做时间范围与点数统计
   当前状态：已完成 preview 路径
3. 做生理/飞机流覆盖率检查
   当前状态：已完成 preview 路径
4. 做缺失、重复、异常时间点检查
5. 做单架次核验报告输出
   当前状态：已完成 preview 路径

退出条件：

- 对最小数据集中的目标架次，能稳定生成核验摘要
- 能明确知道哪些 measurement / 字段 / 时间范围可用

当前已验证：

- 已生成目标架次的 preview 核验摘要
- 在当前 preview 下，生理流结束于 `01:10:49Z`，飞机流开始于 `01:35:00Z`
- 两条流之间当前存在约 `1451000 ms` 的时间空窗
- 在当前 preview 下，`5s / 30s / 5min` 窗口都无法生成联合样本
- 在当前 preview 下，`30min` 窗口可以生成 `1` 个联合窗口
- 后续 full coverage probe 已确认真实完整数据存在重叠
- overlap-focused preview 已在 `5s` 窗口下生成 `25` 个联合窗口
- 这说明阶段 C 的核心结论已经变成：
  1. preview 首次不重叠是抽样策略问题
  2. 完整数据具备进入 E0 的基础
  3. 该结论已在后续 E0/E 阶段实现中吸收并闭环

### 阶段 D：数据集工程化与批量构建

目标：

- 把最小链路扩展成可复用、可批量运行的数据集工程

状态：

- 后置

已完成项：

- 单架次 `SortieBundle -> AlignedSortieBundle -> SampleWindow[]` 最小骨架

待完成项：

1. 加入真实 reader
2. 支持按 sortie 列表批量构建
3. 支持基础落盘格式
4. 支持数据集统计摘要

退出条件：

- 能基于真实架次批量生成窗口样本
- 能生成数据集级摘要

### 阶段 E0：单架次最小训练输入适配

目标：

- 在不先完成完整数据集工程化的前提下，做出能直接喂给模型的单架次实验输入

状态：

- 已完成 preview 路径

适用前提：

- 当前只有人工挑选出的最小可用架次
- 原始全量文件过于杂乱，不适合现在先做完整数据集治理

具体任务：

1. 基于目标架次完成真实读取
   当前状态：已完成 overlap-focused preview 路径
2. 完成单架次窗口样本到模型输入张量的转换
   当前状态：已完成 overlap-focused preview 路径
3. 固化最小训练/验证切分方式
   当前状态：已完成（当前固定 15/5/5）
4. 输出单架次实验摘要
   当前状态：已完成 preview 路径

退出条件：

- 连续对齐模型可以直接消费该单架次实验输入
- 不再依赖临时查询脚本和手工拼接

当前已验证：

- overlap-focused preview 已可稳定生成 `25` 个 E0 样本
- 当前 E0 preview 下：
  - physiology max numeric features: `12`
  - vehicle max numeric features: `21`
- 非数值字段已经被显式剔除并保留 dropped-field 记录
- `E0PreviewPipeline` 已可直接产出最小实验样本
- `AlignmentBatch` 已可由真实 E0 样本直接构建
- 当前状态已经具备进入阶段 E 的条件

### 阶段 E：连续对齐模型原型

目标：

- 实现双流连续潜态对齐最小原型

状态：

- 已完成（收口完成）

具体任务：

1. 基于 E0 定义稳定的模型输入张量协议
   当前状态：已完成最小可用版本
2. 实现生理流 / 飞机流双流编码
   当前状态：已完成最小前向原型
3. 实现连续时间状态推进
   当前状态：已完成最小前向原型
4. 实现最小重构损失
   当前状态：已完成最小可调用版本
5. 实现最小时间对齐损失
   当前状态：已完成最小可调用版本

当前已验证：

- 实验室服务器 `chronaris` 环境已在真实 shell 中确认可用
- 当前实际服务器环境可直接导入 `numpy`、`torch`、`torchdiffeq`、`chronaris`
- 当前实际服务器环境版本为 `numpy 2.4.4`、`torch 2.11.0+cu130`
- `torch.cuda.is_available()` 为 `True`
- 当前可见 GPU 为 `NVIDIA GeForce RTX 4090`
- 当前服务器实际环境与 `configs/environments/chronaris-stage-e-gpu.yml` 不完全一致，后续需决定是回写环境定义还是保持现状
- 已完成 `split / reference grid / torch batch` 基础模块
- 已完成确定性双流 ODE-RNN 最小 forward 原型
- 已完成最小 reconstruction loss 实现与单元测试
- 已完成共享参考时间轴上的最小 `alignment loss`
- 已完成最小 `train / validation / test` preview pipeline 与单元测试
- 已基于真实 overlap-focused E0 样本完成一次最小训练回归
- 历史上曾观察到 vehicle reconstruction loss 量级过大主导 total loss，后续已由 `relative_mse` 回归与收口对照缓解
- 已新增 `relative_mse` 重构损失模式并接入 `AlignmentPreviewConfig`
- 已完成本机最小损失缩放单测，验证 `relative_mse` 可缓解跨流量纲主导
- 服务器环境已完成 `torchdiffeq` 安装并通过 Stage E 相关 runtime 测试（10/10）
- 已完成 `relative_mse` 真实回归：`train/validation/test total` 分别约为 `2.03 / 2.03 / 2.03`
- 已完成共享参考时间轴中间态导出（`test` 分区 `3` 个样本，`16` 个参考点）
- 已提供回归可视化产出：训练曲线、重构曲线、参考时间轴 cosine 曲线，并自动追加到实验报告
- 已新增样本级投影诊断模块（mean/min/max cosine、L2 gap、L2 ratio）
- 已支持自动导出诊断产物：`projection_diagnostics_summary.json` 与 `projection_diagnostics_samples.csv`
- 已完成 `none / zscore_train` 真实对照实跑（seed 固定），并形成阶段收口主报告
- 已支持默认阈值模板评估（默认不强制单点 min cosine），收口判定为 `PASS`
- 已支持自动导出模型 checkpoint：`docs/reports/assets/<report-stem>/alignment_model_checkpoint.pt`

退出条件（当前状态：已满足）：

- 至少完成一个可训练最小实验
- 能输出对齐中间态
- 能给出可复现阈值判据并产出阶段收口报告

### 阶段 F：物理一致性约束

目标：

- 在连续对齐之上加入领域约束

状态：

- 已完成（收口完成）

具体任务：

1. 抽取飞机侧物理残差约束
2. 抽取生理侧平滑/包络约束
3. 接入训练目标
4. 做约束前后对比实验

当前实现进展（2026-04-22）：

- 已在 Stage E objective 中接入可开关的 Stage F 最小物理约束入口（默认关闭，不影响 E baseline）
- 已支持 `feature_first_with_latent_fallback / feature_only / latent_only` 三种约束模式
- 已接入飞机侧最小物理残差约束与生理侧平滑/包络约束
- 已在 preview pipeline / script / report 中接入 physics 指标与可视化导出
- 已扩展为完整物理约束族：飞机侧语义残差/平滑/包络/潜态 fallback，生理侧 EEG 平滑/对称通道/`spo2` 突变/包络/潜态 fallback
- 已接入 MySQL RealBus 字段语义映射，当前真实收口运行加载字段映射 `96` 个
- 已完成 `E baseline` vs `E+F(full)` 真实实跑，默认阈值模板均为 `PASS`
- 已补充对应单元测试、pipeline 测试、真实报告、诊断产物与 checkpoint

退出条件：

- 能量化说明约束是否改善稳定性或解释性
  当前状态：已满足。阶段 F 报告中 `E+F(full)` 的 test physics total 为 `1.539611`，physics component 非零且阈值 verdict 为 `PASS`。

### 阶段 G：因果融合模型原型

目标：

- 实现事件到生理状态的非对称跨模态融合

状态：

- 已完成（最小原型收口）

具体任务：

1. 从飞机连续潜态抽取事件级表示
2. 实现因果掩码
3. 实现非对称交叉注意力
4. 输出融合表示与中间态解释接口

当前实现进展（2026-04-22）：

- 已在 `src/chronaris/models/fusion` 实现 Stage G 最小非对称因果融合模块。
- 已按选题报告中的“当前生理状态作为 Query、历史航电事件作为 Key/Value、非对称因果掩码切断逆向信息流”约束实现单向融合。
- 已在 `src/chronaris/pipelines/causal_fusion.py` 实现从阶段 F 对齐中间态到融合摘要、注意力权重、事件贡献的导出。
- 已扩展单架次脚本，支持 `F baseline` vs `F+G(min)` 对比报告与 `causal_fusion_summary.json` / `causal_fusion_samples.csv` / `causal_attention_heatmap.png`。
- 已在真实单架次 overlap-focused preview 上完成最小融合闭环，默认阈值模板为 `PASS`。

退出条件：

- 能完成一次最小融合实验
- 能导出关键注意力或事件贡献信息
  当前状态：已满足。阶段 G 报告中 `F+G(min)` 导出 `3` 个 test 样本、`16` 个参考点、`96` 维融合表示，平均 causal option count 为 `8.500000`。

### 阶段 H：标准化特征导出

目标：

- 让模型输出变成可复用的数据产品

状态：

- 已完成（收口完成）

当前前置事实（2026-04-25）：

- 已完成第二个真实架次 `20251002_单01_ACT-8_翼云_J16_12#01` 的可用性盘点预验证，详见 `docs/reports/sortie-availability-preview-20251002-act8-j16-12.md`
- 该架次 MySQL 已确认 `collect_task_id=2100450`、`up_pilot_id=10035`、`down_pilot_id=10033`，且 `source_sortie_id` 为空，后续多架次入口需要支持 `collect_task + pilot_ids` 回退
- 该架次 Influx 生理侧已确认 `11` 个 measurement，飞机侧已确认 `BUS6000019110021` 到 `BUS6000019110026` 六个 measurement
- 这说明阶段 H 的 manifest 与 feature export 不能继续沿用阶段 E/F/G 单架次默认的固定 `pilot_id` / 固定 BUS measurement 假设

当前实现进展（2026-04-26）：

- 已在 `src/chronaris/access/stage_h_profile.py` 固化 Stage H profile resolver，正式支持：
  - `source_sortie_id` 与 `collect_task + up/down_pilot_id` 双路径 pilot 解析
  - sortie 级 physiology availability 探测
  - sortie 级完整 BUS family 解析
- 已在 `src/chronaris/pipelines/stage_h_export.py` 实现 run/sortie/view 三级 manifest、`feature_bundle.npz`、`intermediate_summary.json`、`projection_diagnostics_summary.json`、`causal_fusion_summary.json` 与 `window_manifest.jsonl` 导出
- 已在真实两条 sortie 上完成 Stage H v1 导出：
  - `20251005_四01_ACT-4_云_J20_22#01` 导出 `1` 个 pilot view
  - `20251002_单01_ACT-8_翼云_J16_12#01` 导出 `2` 个 pilot view
  - 主报告：`docs/reports/stage-h-export-v1-2026-04-26.md`
  - 机器资产根目录：`artifacts/stage_h/20260426T072340Z-stage-h-v1/`
- 已同步实现 partial-data v1 标准入口：
  - `configs/partial-data/stage-h-seed-v1.jsonl`
  - `src/chronaris/pipelines/partial_data.py`
  - 当前 `20251110_单01_ACT-2_涛_J20_26#01` 已进入 repo 内标准 manifest，但仍是 `manifest_only`

当前实现进展（2026-04-27）：

- 已在 `src/chronaris/features/stage_h_bundle.py` 增加 Stage H feature bundle 下游读取接口，固定校验 `feature_bundle.npz` 的六个核心键，避免下游评测继续直接依赖训练中间脚本。
- 已将 Stage H 导出配置拆出 `preview / validation / full_clip` 三档 profile：
  - `preview` 默认保留每个 measurement `500` 点查询上限，用于本地/服务器快速验证和防止误扫全量库；
  - `validation` 默认不加点数上限，但仍可复用显式导出范围；
  - `full_clip` 默认使用 sortie/collect-task clip 边界且不加点数上限。
- 已在 Stage H 主报告生成逻辑中输出 point-limit 语义，并把 `WARN` view 的失败阈值项写入报告，便于区分“导出失败”和“诊断阈值提醒”。
- 已将 `20251110_单01_ACT-2_涛_J20_26#01` 的 partial-data seed 补齐为真实 vehicle-only entry：
  - `bucket=bus`
  - 时间范围：`2025-11-10T00:00:00Z` 到 `2025-11-11T00:00:00Z`
  - BUS family：`BUS6000019110027` 到 `BUS6000019110031`
  - `tag_filters={"sortie_number":"20251110_单01_ACT-2_涛_J20_26#01"}`
- 已实现 partial-data 的分块 Influx vehicle-only reader、MySQL RealBus 字段过滤、每 `5s` 窗口每字段最多 `32` 点的有界导出。
- 已完成 Stage H 收口实跑：
  - 双流 `validation` profile 仍导出 `3` 个 view；
  - `load_stage_h_feature_run()` 可从 run manifest 读取三 view，三者 `fused_representation.shape` 均为 `(8, 16, 96)`；
  - `20251110...` vehicle-only partial 生成 `1478` 个窗口样本，`vehicle_only_feature_bundle.npz` 的 `values.shape=(1478, 105, 823)`。
  - 主报告：`docs/reports/stage-h-closure-2026-04-27.md`
  - 收口记录：`docs/planning/stage-h-closure-2026-04-27.md`
  - 机器资产根目录：`artifacts/stage_h/20260427T000000Z-stage-h-closure/`

具体任务：

1. 固化 Stage H v1 导出资产的下游消费接口与版本约定
   当前状态：已完成读取接口、固定键校验与真实 run manifest 读取验证。
2. 补齐 partial-data 的真实 vehicle-only 数据范围，使 seed 资产可进入单流样本导出
   当前状态：已完成，`20251110...` 已生成 window manifest 与 feature bundle。
3. 继续扩展轻量多架次 manifest 盘点，但不提前宣称多架次训练条件已齐备
   当前状态：转入阶段 I 前置准备，不作为阶段 H 收口阻塞。
4. 在上述基础上评估阶段 H 收口 gate，而不是仅凭单次脚本成功就直接关阶段
   当前状态：已完成真实实跑、判据可复现、测试覆盖与文档回写。

退出条件：

- 下游评测不需要直接依赖训练中间脚本
  当前状态：已满足。阶段 I 默认从 `load_stage_h_feature_run()` 和 partial vehicle-only bundle 消费 Stage H 资产。

### 阶段 I：下游任务验证

目标：

- 用典型任务验证整条链路的价值

状态：

- 进行中（`Phase 0 + Phase 1 + Phase 2` 已跑通，`Phase 3` 真实收口进行中）

总体分期：

1. `Phase 0`：数据 contract / 环境 / 评测入口固化
2. `Phase 1`：UAB 主数据集双轨 baseline
3. `Phase 2`：Stage H 真实双流资产 case study
4. `Phase 3`：NASA CSM 第二公开数据集、对比/消融补齐与阶段 I 收口

优先任务：

1. 认知负荷评估
2. 空中失能风险分析
3. 飞行事件复盘

退出条件：

- 至少完成一轮对比实验和一轮消融实验

当前已验证：

- 已完成 UAB `task_manifest.jsonl` 导出：`87` 个 session（`n_back=48`、`heat_the_chair=34`、`flight_simulator=5`）
- 已完成 `feature_table.parquet` 导出：`416` 维 session 级特征（EEG `404` / ECG `12`）
- 已确认 `n_back` 有 `9` 个 session 缺失 ECG，但不阻塞当前 CPU baseline
- 已完成 UAB 客观任务标签分类：
  - `heat_the_chair` 最优 `logistic_regression`：`macro-F1=0.6173`
  - `n_back` 最优 `random_forest_classifier`：`macro-F1=0.4966`
- 已完成 UAB 主观负荷回归：
  - `heat_the_chair` 最优 `random_forest_regressor`：`RMSE=1.4664`
  - `n_back` 最优 `random_forest_regressor`：`RMSE=4.5161`
- 当前机器资产根目录：`artifacts/stage_i/20260429T000000Z-stage-i-phase0-1-uab/`
- 当前主报告：`docs/reports/stage-i-uab-baseline-2026-04-29.md`
- 已完成 Phase 2 真实双流 case study：
  - `3` 个真实双流 view 全部纳入主线（`PASS=2`、`WARN=1`）
  - `4` 条 bundle-only 路径：`projection_refusion_baseline / no_event_bias / no_state_normalization / vehicle_delta_suppressed`
  - 已完成同 sortie 双 pilot 对比：`20251002_单01_ACT-8_翼云_J16_12#01`
  - `vehicle_delta_suppressed` 在 `3` 个 view 上均将 `mean_top_event_score` 压到 `0.0`
  - 当前 Phase 2 机器资产根目录：`artifacts/stage_i/20260429T000000Z-stage-i-phase2-case-study/`
  - 当前 Phase 2 主报告：`docs/reports/stage-i-case-study-phase2-2026-04-29.md`

## 5. 当前已完成的代码状态

下面这些编码已经完成，可作为后续开发基底：

- `access` 协议与内存实现
- 真实 MySQL / Influx CLI reader
- Stage B live loader factory
- `SortieLoader`
- `SortieMetadata / RawPoint / SortieBundle / SampleWindow` 等统一对象
- 时间基准对齐
- 窗口构建
- 飞机时间跨日组装工具
- RealBus 元信息派生骨架
- 基础单元测试
- Stage E 输入归一化模式（`none / zscore_train`）
- Stage E 样本级诊断阈值评估与收口对照报告
- Stage E 自动 checkpoint 导出
- Stage F 完整物理约束族、组件级 physics breakdown、MySQL RealBus 字段语义映射
- Stage F `E baseline` vs `E+F(full)` 收口对照报告、图表、JSON/CSV 诊断和 checkpoint
- Stage G 最小非对称因果融合、事件贡献摘要、注意力热力图与 `F baseline` vs `F+G(min)` 收口对照报告
- Stage H v1 双架次标准化导出、run/sortie/view 三级 manifest、固定键 `feature_bundle.npz`
- Stage H feature bundle 下游读取接口：`src/chronaris/features/stage_h_bundle.py`
- partial-data 标准 manifest、vehicle-only reader / builder、真实 vehicle-only feature bundle
- Stage I task manifest contract、UAB 数据适配、session 级特征导出与 UAB 双轨 baseline
- Stage I Phase 2 case-study asset loader、bundle-only 消融、`WARN` 解释与中文主报告

对应代码：

- [access](../../src/chronaris/access)
- [schema](../../src/chronaris/schema)
- [dataset](../../src/chronaris/dataset)
- [dataset_v1.py](../../src/chronaris/pipelines/dataset_v1.py)

## 6. 当前未完成但最该做的事

阶段 H 已完成收口，阶段 I 当前已经进入 `Phase 3` 真实收口阶段。当前不再优先扩展新模型结构，后续优先级收敛为“完成 window-level UAB + NASA CSM 真跑证据，再评估阶段 I 收口 gate”。

按优先级排序：

1. 完成 `Phase 3` 真实长跑，生成 `UAB window + NASA attention` 主结果、消融和 closure summary。
2. 完成 `Phase 3` 中文主报告与 planning closure 文档回写，再决定是否把阶段 I 标记为 completed。
3. 保持 `vehicle_only_feature_bundle.npz` 仍只用于单流预训练/补充诊断，不作为双流融合 view。
4. 在不破坏 frozen E/F/G(min)/H 与已完成 Phase 0/1/2 的前提下，评估阶段 I 收口 gate。

## 7. 当前不该提前做的事

1. 提前写完整训练框架
2. 把 `20251110...` vehicle-only 样本误提升为双流 Stage H 融合样本
3. 提前写大而全特征工程
4. 提前设计最终服务化接口
5. 跳过对比实验直接宣称约束有效

## 8. 路线图维护规则

后续每推进一个阶段，至少更新这三类信息：

1. 当前处于哪个阶段
2. 哪些任务从“未开始”变成“进行中/已完成”
3. 下一步的最高优先级是什么

建议更新频率：

- 每完成一个可验证里程碑就更新一次
- 不要求每次细小重构都更新
