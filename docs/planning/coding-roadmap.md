# Coding Roadmap

更新时间：2026-04-17

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

`阶段 A 已完成，阶段 E0 已完成 preview 路径，阶段 E 已启动`

更具体地说：

- 已完成仓库初始化、分层、最小设计文档、统一对象骨架、时间语义约束和 v1 数据构建骨架
- 已开始真实 InfluxDB / MySQL 接入准备
- 已开始基于真实架次的 preview 联调
- 已开始单架次时间覆盖与窗口策略核验
- 已完成 overlap-focused E0 最小实验输入适配
- 已开始阶段 E 的执行计划固化与最小训练/验证切分策略实现
- 已完成实验室服务器 WSL Ubuntu 22.04 + RTX 4090 环境迁移与依赖验证
- 还没有开始模型训练代码

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

- [AGENTS.md](D:/code/chronaris/AGENTS.md)
- [architecture.md](D:/code/chronaris/docs/foundation/architecture.md)
- [data-contracts.md](D:/code/chronaris/docs/foundation/data-contracts.md)
- [pipeline-v1.md](D:/code/chronaris/docs/foundation/pipeline-v1.md)
- [models.py](D:/code/chronaris/src/chronaris/schema/models.py)
- [temporal.py](D:/code/chronaris/src/chronaris/access/temporal.py)

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
- 这说明下一步应优先进入阶段 C，核验生理流和飞机流的真实时间覆盖关系与窗口策略

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

- 进行中

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
  3. 下一步重点是固化最小实验输入，而不是继续怀疑是否重叠

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

- 进行中

适用前提：

- 当前只有人工挑选出的最小可用架次
- 原始全量文件过于杂乱，不适合现在先做完整数据集治理

具体任务：

1. 基于目标架次完成真实读取
   当前状态：已完成 overlap-focused preview 路径
2. 完成单架次窗口样本到模型输入张量的转换
   当前状态：已完成 overlap-focused preview 路径
3. 固化最小训练/验证切分方式
   当前状态：已开始
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

- 已启动

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
- 当前已确认 `alignment loss` 可下降，但 vehicle reconstruction loss 量级过大并主导 total loss

退出条件：

- 至少完成一个可训练最小实验
- 能输出对齐中间态

### 阶段 F：物理一致性约束

目标：

- 在连续对齐之上加入领域约束

状态：

- 未开始

具体任务：

1. 抽取飞机侧物理残差约束
2. 抽取生理侧平滑/包络约束
3. 接入训练目标
4. 做约束前后对比实验

退出条件：

- 能量化说明约束是否改善稳定性或解释性

### 阶段 G：因果融合模型原型

目标：

- 实现事件到生理状态的非对称跨模态融合

状态：

- 未开始

具体任务：

1. 从飞机连续潜态抽取事件级表示
2. 实现因果掩码
3. 实现非对称交叉注意力
4. 输出融合表示与中间态解释接口

退出条件：

- 能完成一次最小融合实验
- 能导出关键注意力或事件贡献信息

### 阶段 H：标准化特征导出

目标：

- 让模型输出变成可复用的数据产品

状态：

- 未开始

具体任务：

1. 定义特征矩阵格式
2. 定义中间态落盘格式
3. 定义版本号与配置记录
4. 实现导出 pipeline

退出条件：

- 下游评测不需要直接依赖训练中间脚本

### 阶段 I：下游任务验证

目标：

- 用典型任务验证整条链路的价值

状态：

- 未开始

优先任务：

1. 认知负荷评估
2. 空中失能风险分析
3. 飞行事件复盘

退出条件：

- 至少完成一轮对比实验和一轮消融实验

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

对应代码：

- [access](D:/code/chronaris/src/chronaris/access)
- [schema](D:/code/chronaris/src/chronaris/schema)
- [dataset](D:/code/chronaris/src/chronaris/dataset)
- [dataset_v1.py](D:/code/chronaris/src/chronaris/pipelines/dataset_v1.py)

## 6. 当前未完成但最该做的事

服务器环境已经不再阻塞阶段 E，后续优先级重新收敛为：

按优先级排序：

1. 导出并检查共享参考时间轴上的中间态
2. 处理 vehicle stream 的尺度问题或 loss weighting 问题
3. 基于修正后的目标再跑一轮真实训练回归

## 7. 当前不该提前做的事

1. 提前写完整训练框架
2. 提前写大而全特征工程
3. 提前设计最终服务化接口
4. 在没有真实数据联调和 E0 输入适配前直接开双流模型训练

## 8. 路线图维护规则

后续每推进一个阶段，至少更新这三类信息：

1. 当前处于哪个阶段
2. 哪些任务从“未开始”变成“进行中/已完成”
3. 下一步的最高优先级是什么

建议更新频率：

- 每完成一个可验证里程碑就更新一次
- 不要求每次细小重构都更新
