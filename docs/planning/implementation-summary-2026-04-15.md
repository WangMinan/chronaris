# Implementation Summary - 2026-04-15

## 1. 当前仓库已经实现了什么

截至目前，`chronaris` 已经从空仓推进到“真实 preview 路径可运行”的状态。

已完成的核心能力：

1. 仓库边界、分层和执行路线图
2. 统一 schema 与时间语义约束
3. 真实 MySQL / Influx 访问层
4. 单架次 preview `SortieBundle`
5. `DatasetPipelineV1`
6. 单架次核验摘要
7. `E0` 最小实验输入适配
8. `AlignmentBatch` 模型前置批处理协议

## 2. 主要代码区域

### `src/chronaris/access`

- 真实 MySQL CLI reader
- 真实 Influx CLI reader
- real bus 元信息派生
- 生理查询上下文推导
- live reader / live loader factory
- Influx coverage probe

### `src/chronaris/schema`

- `SortieLocator`
- `RawPoint`
- `SortieBundle`
- `SortieMetadata`
- real bus / collect task 相关领域对象

### `src/chronaris/dataset`

- 统一时间基准
- 相对时间偏移
- 窗口切分
- `SortieDatasetBuilder`

### `src/chronaris/evaluation`

- 单架次流覆盖摘要
- 生理/飞机流重叠关系判断
- 多种窗口尺度可行性试验
- Markdown 报告输出

### `src/chronaris/features`

- `E0ExperimentSample`
- 数值流矩阵
- 非数值字段显式剔除规则

### `src/chronaris/pipelines`

- `DatasetPipelineV1`
- `E0PreviewPipeline`

### `src/chronaris/models/alignment`

- `AlignmentBatch`
- stream padding / mask / offsets 协议

## 3. 当前真实验证结果

目标架次：

- `20251005_四01_ACT-4_云_J20_22#01`

### 真实库事实

- 飞行日期语义来自 `flight_batch.fly_date`
- 总线 measurement 已确认：
  `BUS6000019110020`
- 当前生理 preview 已重点验证：
  `eeg`、`spo2`

### 阶段 C 关键结论

第一轮 preview 因为限流位置问题，看起来像“生理流和飞机流不重叠”。  
后续 full coverage probe 已证实：

- 生理流完整覆盖至少到 `2025-10-05T06:45:40Z`
- 飞机流覆盖 `2025-10-05T01:35:00Z` 到 `2025-10-05T01:38:00.764Z`

所以完整数据中存在真实重叠。

### overlap-focused preview 结果

在重叠区间附近裁剪后，使用：

- physiology measurements: `eeg`, `spo2`
- physiology point limit per measurement: `500`
- vehicle point limit: `500`
- sample window: `5s / 5s`

可以得到：

- preview `SortieBundle`
- `DatasetBuildResult`
- `25` 个联合窗口

## 4. E0 当前结果

当前 overlap-focused `E0` preview 已验证：

- E0 samples: `25`
- physiology max numeric features: `12`
- vehicle max numeric features: `21`

已知行为：

- 时间字符串和状态字符串会被显式丢弃
- 数值字段会稳定转成流矩阵
- dropped fields 会保留记录

## 5. 阶段 E 的就绪判断

当前已经满足进入模型编写阶段的最低条件：

1. 有稳定的 overlap-focused preview 输入
2. 有可重复生成的 `E0ExperimentSample`
3. 有可重复生成的 `AlignmentBatch`
4. 输入协议不再依赖手工拼接

## 6. 当前尚未完成的内容

这些还没做：

1. 最小训练/验证切分策略定稿
2. 真实双流连续对齐模型本体
3. 重构损失与对齐损失实现
4. 物理一致性约束
5. 因果融合模型

## 7. 依赖与现实约束

- 当前本地环境已确认有 `numpy`
- 当前本地环境未确认可直接使用 `torch`

因此：

- 现在可以进入模型编写阶段
- 但如果要直接落训练网络，实现前需要先决定是否补 `torch` 依赖

## 8. 相关文档

- [coding-roadmap.md](D:/code/chronaris/docs/planning/coding-roadmap.md)
- [validation-preview-20251005-act4-j20-22.md](D:/code/chronaris/docs/reports/validation-preview-20251005-act4-j20-22.md)
- [validation-overlap-preview-20251005-act4-j20-22.md](D:/code/chronaris/docs/reports/validation-overlap-preview-20251005-act4-j20-22.md)
- [e0-minimal-input.md](D:/code/chronaris/docs/models/e0-minimal-input.md)
- [e0-preview-20251005-act4-j20-22.md](D:/code/chronaris/docs/reports/e0-preview-20251005-act4-j20-22.md)
- [alignment-batch-contract.md](D:/code/chronaris/docs/models/alignment-batch-contract.md)
