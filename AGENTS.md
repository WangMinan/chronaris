# Chronaris 协作说明

## 1. 项目定位

`chronaris` 是“航空人机异构时序数据连续对齐与语义融合”仓库。

这个仓库默认承接的是下游研究与原型实现，而不是历史接收器那一层的文件入库工作。当前应把上游状态理解为：

- 生理数据、飞机侧时序数据已经进入 InfluxDB
- 业务元数据已经进入 MySQL
- 当前仓库要做的是读取、组织、建模、导出和验证

## 2. 代码层面的最终目标

后续代码工作默认朝下面这条主线收敛：

1. 从 InfluxDB / MySQL 读取指定架次的人机多源数据及其元信息。
2. 建立统一 schema、统一时间参考和统一样本组织方式。
3. 实现双流连续潜态建模，用于分别表示“生理流”和“航电/总线流”。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合，默认遵守“航电事件影响生理状态”的单向逻辑。
6. 输出标准化融合特征矩阵和中间态解释接口。
7. 面向典型任务开展对比实验、消融实验和案例验证。

典型下游任务包括但不限于：

- 空中失能风险分析
- 认知负荷评估
- 飞行事件复盘

## 3. 当前已知事实

### 理论依据

本项目的理论依据是我的选题报告与基金申请书，你可以调用 `docx` skill来执行读取。

+ `D:\0_大学\2025.9\选题\王旻安-西北工业大学硕士学位研究生论文选题报告表.docx`
+ `D:\0_大学\2025.9\2026年研究生创新基金申报\王旻安-4.6改-附件2.西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx`

### 历史说明文档

- `D:\0_大学\2024.9\实验室\数据中台\0_实采数据\AGENTS.md`

### 目前已明确导入的样例

- `D:\0_大学\2024.9\实验室\数据中台\0_实采数据\22_导入测试数据记录\20251005_四01_ACT-4_云_J20_22#01-2100448-10033\SZ生理数据`
- `D:\0_大学\2024.9\实验室\数据中台\0_实采数据\22_导入测试数据记录\20251005_四01_ACT-4_云_J20_22#01-2100448-10033\机载总线数据`

用户已明确：

- 工作语言为中文
- 上述对应文件内容已导入 InfluxDB
- 相关元信息已写入 `10.70.4.57` 的 MySQL `rjgx_backend`
- 生理数据时间精确到小数点后六位
- 飞机数据时间精确到小数点后三位
- 飞机数据原始记录只有时分秒，不天然带完整日期
- 当前真实库中起飞日期语义来自 `rjgx_backend.flight_batch.fly_date`
- 飞机数据的完整日期时间拼接逻辑应参考 `TimeSequenceProcessor`

这些样例可作为当前仓库第一批联调与验证对象。

## 4. 事实来源优先级

后续会话默认按下面优先级判断事实：

1. 当前仓库中的代码、文档与配置
2. `D:\code\zorathos\zorathos-data-model`
3. `D:\code\zorathos\zorathos-data-receiver`
4. 本地真实样例数据与历史 `D:\0_大学\2024.9\实验室\数据中台\0_实采数据\AGENTS.md`
5. 选题报告与基金申请书

如果不同来源冲突：

- 当前仓库内经过明确沉淀的结论优先于外部研究文档表述
- 外部源码事实优先于选题/基金中的论文化表述
- 对上游接收链路有疑问时，优先回看 `zorathos` 代码，不凭论文描述臆断

## 5. 范围边界

### 默认属于当前仓库

- 数据访问封装
- 统一字段与 schema
- 时间基准校正
- 样本窗口构建
- 连续对齐模型
- 因果融合模型
- 融合特征导出
- 评测与案例复盘
- 近实时推理接口预留

### 默认不属于当前仓库

- 重写历史文件接收器
- 重新建设上游数据入库链路
- 在仓库内存放大体量原始实采数据
- 试图一次性建立全量总线静态总模型

## 6. 目录约定

后续新增代码时遵守以下分层：

- `src/chronaris/access`: InfluxDB / MySQL 访问
- `src/chronaris/schema`: 统一 schema
- `src/chronaris/dataset`: 样本组织、窗口切分、时间基准
- `src/chronaris/models/alignment`: 连续对齐
- `src/chronaris/models/fusion`: 因果融合
- `src/chronaris/features`: 特征导出与中间态
- `src/chronaris/pipelines`: 训练/导出/验证流程
- `src/chronaris/serving`: 服务化或近实时接口
- `src/chronaris/evaluation`: 对比、消融、案例分析
- `configs`: 可复现配置
- `experiments`: 实验记录
- `scripts`: 一次性脚本
- `docs`: 说明性的文档

约束：

- 可复用逻辑必须进入 `src/chronaris`
- `scripts` 不承载核心业务实现
- notebook 若后续出现，只用于探索，不能成为唯一事实来源
- 在  `docs` 目录引入文档时默认使用中文

## 7. 工作默认顺序

处理任务时，优先按下面顺序定位：

1. 这是数据访问问题、样本问题、模型问题，还是评测问题？
2. 如果涉及总线字段或元数据映射，先查上游业务元数据来源，不要只盯模型代码。
3. 如果涉及时间错位，优先检查时间参考、窗口切分和对齐前处理，不要先怪融合模型。
4. 如果涉及下游效果不稳，先拆开看样本质量、对齐质量、融合质量和标签质量。

## 8. 时间语义补充

- 生理流默认直接携带完整时间戳，且应保留微秒级精度。
- 飞机流默认只携带时分秒及毫秒级小数，完整日期要结合 `flight_task.flight_date` 组装。
- 组装飞机完整时间戳时，默认沿用 `TimeSequenceProcessor` 的跨日规则：
  当前时刻小于上一条参考时刻时，日期偏移加一。
- 即使当前最小数据集不跨日，也不能把跨日逻辑写死删除，只能在实现上做显式预留。

## 9. 工程与安全约束

- 不要把数据库密码、token、连接串原样写入新文档或提交到版本库。
- 不要把原始大数据文件复制进仓库；仓库只保存代码、配置、轻量样例说明和实验结论。
- 对于临时验证脚本，能复用的部分要及时回收进正式模块。
- 做实验时尽量保留可复现实验配置、关键指标和所用架次范围。

## 10. 当前阶段的合理目标

当前仓库已经不再是空仓，当前合理目标按下面顺序推进：

1. 继续沿 `docs/planning/coding-roadmap.md` 推进当前阶段
2. 优先保住真实数据联调链路，而不是过早扩张到全量工程化
3. 在 overlap-focused preview 基础上推进 E0 和阶段 E
4. 后续再回补完整数据集工程化与批量构建

## 11. 当前实现状态

截至目前，仓库内已经完成这些能力：

- `access`
  已实现真实 MySQL / Influx CLI reader、live loader factory、时间探针与 preview 裁剪能力。
- `schema`
  已实现统一对象模型，覆盖 `SortieBundle`、`RawPoint`、`WindowConfig` 等核心契约。
- `dataset`
  已实现统一时间基准、相对时间偏移与窗口切分。
- `evaluation`
  已实现单架次核验摘要与窗口可行性分析。
- `features`
  已实现 `E0ExperimentSample` 和最小数值特征矩阵适配。
- `pipelines`
  已实现 `DatasetPipelineV1` 和 `E0PreviewPipeline`。
- `models/alignment`
  已实现 `AlignmentBatch` 这层模型前置批处理协议。

当前结论：

- 阶段 B 已完成 preview 路径。
- 阶段 C 已证明完整数据存在真实重叠，最初的“不重叠”是 preview 抽样问题。
- 阶段 E0 已完成 preview 路径，当前已经具备进入阶段 E 的条件。

## 12. 当前真实验证结论

目标架次：

- `20251005_四01_ACT-4_云_J20_22#01`

当前真实验证已确认：

- `flight_task` 不直接提供 `flight_date`，当前真实库里的飞行日期语义来自 `flight_batch.fly_date`。
- 总线数据原始完整时间戳拼接应继续遵循 `TimeSequenceProcessor` 的跨日语义。
- 目标总线 measurement 已确认可用：
  `BUS6000019110020`
- 目标生理数据在 `physiological_input` 中存在多个 measurement，当前 preview 重点已验证：
  `eeg`、`spo2`
- full coverage probe 已确认：
  生理流覆盖至少到 `2025-10-05T06:45:40Z`
- 同一架次的飞机流覆盖：
  `2025-10-05T01:35:00Z` 到 `2025-10-05T01:38:00.764Z`
- overlap-focused preview 已在 `5s` 窗口下生成 `25` 个联合窗口。

## 13. 配置与密钥约束

- 不要在仓库文档中记录明文数据库密码、Influx token 或其他密钥。
- 当前连接配置应从外部配置源读取，优先参考：
  `D:\code\zorathos\zorathos-data-receiver\data-receiver-human-machine\src\main\resources\human-machine.properties`
- 如需运行真实 reader，应通过现有配置加载逻辑或环境变量注入，不要把密钥再抄进仓库文件。

## 14. 继续编码时的默认策略

- 如果目标是继续研究主线，默认从 `docs/planning/coding-roadmap.md` 当前最高优先级继续。
- 如果目标是开始模型，优先走已经验证过的 overlap-focused E0 preview 路径。
- 如果阶段 E 代码需要训练框架，先确认本地依赖是否齐全，再决定是先补依赖还是先写纯协议骨架。

## 15. 本地 Conda 环境

目标复用环境名：

- `chronaris`

当前状态：

- 全局目录 `D:\env\anaconda3\envs\chronaris` 已存在
- 已验证可用：
  - `D:\env\anaconda3\envs\chronaris\python.exe --version`
  - `D:\env\anaconda3\envs\chronaris\python.exe -m pip --version`
- 当前验证结果：
  - `Python 3.11.15`
  - `pip 26.0.1`

结论：

- 当前 `chronaris` 环境可以作为后续默认复用环境
- 当前环境是“最小 Python 环境”定位，后续缺什么包再增量安装什么包

激活方式：

```powershell
conda activate chronaris
```

或直接调用：

```powershell
D:\env\anaconda3\envs\chronaris\python.exe
```

后续策略：

- 默认优先使用该环境执行后续 Python 代码
- 编码过程中缺什么包，再逐个增量安装什么包
- 不要一开始就装大而全依赖
- 尽量使用 `conda` 安装依赖，如果一定要使用 `pip` ，需要屏蔽本地的 `http_proxy` 以及 `https_proxy`。如果因为镜像源没有对应包导致下载失败，可以下载 `whl`到本地后手动安装
