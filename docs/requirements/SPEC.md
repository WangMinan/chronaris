# Chronaris 需求规格入口

更新时间：2026-09-10

## 1. 毕业论文目标

2026-09-09 按用户提供的导师会谈转述补充交付要求：10 月 15 日争取完整初稿，10 月 30 日为最迟完整稿节点，11 月 30 日前完成送盲审准备；实验需比较近期大模型增强方法，参考文献尽量达到 70% 以上为 2025 年以后工作，并通过项目应用和实测说明方法价值。年份按正式发表与预印本口径核验，不能按后续修订年份凑比例。具体证据缺口与条件性排期见[导师要求盘点](../review/stage/thesis-v4/advisor-status-20260909/README.md)；本补充不自动改变已冻结实验配置。

本仓库服务于“航空人机异构时序数据连续对齐与语义融合”方向的毕业设计。最终代码能力应支撑：

1. 从 MySQL / InfluxDB 中读取指定架次的人机多源数据及元信息。
2. 统一生理流和飞机时序流的 schema、时间参考和样本组织。
3. 建立双流连续潜态建模链路。
4. 引入物理一致性约束进行时间对齐。
5. 引入非对称因果掩码和语义事件融合。
6. 输出标准化融合特征、中间态和可解释证据。
7. 面向空中失能风险分析、认知负荷评估、飞行事件复盘开展对比、消融和案例验证。

当前采用[融合表示下游评价方案](thesis-downstream-representation-plan-20260909.md)：各方法处理相同原始观测，再分别训练相同类型的下游算法，使用相同任务标签比较。真实鼎新压力、精神状态与具体机动动作的独立业务标签当前不可获得；现有生理字段预测、机动强度及公开负荷任务支撑组件与功能验证。潜在特征允许作为重新训练的下游模型输入，不以取得另一组固定算法接口为前提。

## 2. 原始需求材料

原始 Word 文档保存在：

- [选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx](选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx)
- [选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx](选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx)

解析、摘录或修改 `.docx` 时必须使用 `$docx` skill 或文档插件。不要靠文件名或记忆猜测 Word 内容。

## 3. 仓库能力边界

默认承接：

- 下游研究与原型实现。
- 数据读取、组织、建模、导出、验证。
- 论文主线相关报告、证据包和代码 contract。

默认不承接：

- 历史文件接收器重写。
- 上游入库链路重建。
- 原始大数据文件入仓。

上游现状默认视为：

- 生理数据、飞机时序数据已进入 InfluxDB。
- 业务元数据已进入 MySQL。

## 4. 关键技术要求

- 生理流时间精度：微秒级。
- 飞机流时间精度：毫秒级。
- 飞机原始时间只有时分秒，完整日期来自 `flight_batch.fly_date`。
- 飞机完整时间拼接沿用 `TimeSequenceProcessor` 跨日规则。
- 历史采集记录标识（九月去重后不视为两个独立评价架次）：
  - `20251005_四01_ACT-4_云_J20_22#01`
  - `20251002_单01_ACT-8_翼云_J16_12#01`

当前鼎新评价使用保留单记录后的时间块与两个生理视图，划分和有效字段见[新方案](thesis-downstream-representation-plan-20260909.md)及[去重报告](../artifacts/runs/2026-09-08_v4-dingxin-deduplicated/report.md)。

## 5. 需求补充材料

- [foundation/project-scope.md](foundation/project-scope.md)
- [foundation/repo-layout.md](foundation/repo-layout.md)
- [foundation/architecture.md](foundation/architecture.md)
- [foundation/data-contracts.md](foundation/data-contracts.md)
- [foundation/pipeline-v1.md](foundation/pipeline-v1.md)
- [model-contracts/e0-minimal-input.md](model-contracts/e0-minimal-input.md)
- [model-contracts/alignment-batch-contract.md](model-contracts/alignment-batch-contract.md)
- [model-contracts/stage-e-prototype-design.md](model-contracts/stage-e-prototype-design.md)
- [model-contracts/stage-e-reference-repos.md](model-contracts/stage-e-reference-repos.md)

## 6. 固定数据下游评估补充规格

毕业论文后续实验不依赖新增鼎新一手数据或人工专家评价。研究目标和下一步执行以[新方案](thesis-downstream-representation-plan-20260909.md)为准；以下规格提供基础定义和历史来源，仅继承与新方案一致的内容：

- [foundation/fixed-data-evidence-strategy.md](foundation/fixed-data-evidence-strategy.md)：数据来源、证据层级、论文论断与失败边界。
- [downstream-evaluation-spec.md](downstream-evaluation-spec.md)：真实/仿真任务、标签公式、划分、下游算法、指标与模型选择。
- [synthetic-benchmark-spec.md](synthetic-benchmark-spec.md)：G1/G2 仿真生成族、观测场景、oracle、压力等级和验收。
- [model-contracts/application-fusion-stream-contract.md](model-contracts/application-fusion-stream-contract.md)：六方法共同消费的异步双流和 `[B,T,64]` 融合表示合同。

## 7. 协议继承与下一阶段

现有六方法实现来自 [v4 开发与实验合同](thesis-v4-development-plan.md)及其鼎新单记录时间块修订；[七月简化任务](simple-downstream-evaluation-v1.md)保留目标公式沿革，其留一架次、仅无标签训练等旧规则不再覆盖 v4 双路线与去重修订。

[阶段 1](../artifacts/runs/2026-09-10_v4-stage1-repair/report.md)已修复导出池化与失败终态，并明确旧检查点、序列及下游模型的复用范围；后续接入 TimeCMA 语言模型增强时序方法、Chronos-2 预训练时序模型，并尽量纳入 SensorLLM 传感器—语言对齐模型。新增方法须通过共同下游评价的表示与监督核验后进入完整比较；不以原生未来序列预测矩阵替代主任务。具体代码落点、验收、环境和论文节点统一见[新方案](thesis-downstream-representation-plan-20260909.md)。
