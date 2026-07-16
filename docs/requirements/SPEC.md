# Chronaris 需求规格入口

更新时间：2026-07-16

## 1. 毕业论文目标

本仓库服务于“航空人机异构时序数据连续对齐与语义融合”方向的毕业设计。最终代码能力应支撑：

1. 从 MySQL / InfluxDB 中读取指定架次的人机多源数据及元信息。
2. 统一生理流和飞机时序流的 schema、时间参考和样本组织。
3. 建立双流连续潜态建模链路。
4. 引入物理一致性约束进行时间对齐。
5. 引入非对称因果掩码和语义事件融合。
6. 输出标准化融合特征、中间态和可解释证据。
7. 面向空中失能风险分析、认知负荷评估、飞行事件复盘开展对比、消融和案例验证。

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
- 当前主线 sortie：
  - `20251005_四01_ACT-4_云_J20_22#01`
  - `20251002_单01_ACT-8_翼云_J16_12#01`

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

毕业论文后续实验默认不依赖新增鼎新一手数据或人工专家评价。当前下游评估、仿真和融合表示按以下规格执行：

- [foundation/fixed-data-evidence-strategy.md](foundation/fixed-data-evidence-strategy.md)：数据来源、证据层级、论文论断与失败边界。
- [downstream-evaluation-spec.md](downstream-evaluation-spec.md)：真实/仿真任务、标签公式、划分、下游算法、指标与模型选择。
- [synthetic-benchmark-spec.md](synthetic-benchmark-spec.md)：G1/G2 仿真生成族、观测场景、oracle、压力等级和验收。
- [model-contracts/application-fusion-stream-contract.md](model-contracts/application-fusion-stream-contract.md)：六方法共同消费的异步双流和 `[B,T,64]` 融合表示合同。

## 7. 当前简化下游评价规格

2026-07-16 起，鼎新真实数据的当前执行入口切换为 [鼎新简化下游评价协议 v1](simple-downstream-evaluation-v1.md)。该协议保留固定数据与统一表示基础设施，但以留一架次分组确认、未来机动连续预测和未来生理字段预测为主，不再自动续跑旧目标重构门禁。旧规格和产物继续作为历史研究记录，不覆盖新协议。
