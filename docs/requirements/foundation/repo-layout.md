# Repo Layout

更新时间：2026-09-05

## 1. 分层原则

这个仓库按“可复用库代码、运行入口、文档事实源、实验产物”分开组织。

核心原则：

1. `src/chronaris` 只放可复用模块。
2. `configs` 只放可复现实验或运行配置。
3. `scripts` 只放 CLI 编排，不承载核心业务逻辑。
4. `docs` 负责需求、状态、计划、产物和 review 记录。
5. `tests` 负责稳定合约与阶段回归验证。

## 2. 推荐目录

```text
chronaris/
├─ AGENTS.md
├─ README.md
├─ configs/
├─ docs/
│  ├─ README.md
│  ├─ STATE.md
│  ├─ implementation/
│  │  ├─ TASKS.md
│  │  └─ notes/
│  ├─ requirements/
│  │  ├─ SPEC.md
│  │  ├─ foundation/
│  │  ├─ model-contracts/
│  │  └─ 选题报告与基金申请书/
│  ├─ artifacts/
│  │  ├─ ARTIFACTS.md
│  │  ├─ runs/
│  │  ├─ archive/
│  │  └─ mid-term/
│  └─ review/
│     ├─ REVIEW.md
│     └─ stage/
├─ scripts/
├─ src/
│  └─ chronaris/
│     ├─ access/
│     ├─ schema/
│     ├─ dataset/
│     ├─ models/
│     │  ├─ alignment/
│     │  └─ fusion/
│     ├─ features/
│     ├─ pipelines/
│     ├─ serving/
│     └─ evaluation/
└─ tests/
```

## 3. 文档目录职责

### `docs/implementation`

执行入口。`TASKS.md` 负责总路线、阶段结构、当前任务队列和默认工作方式，`notes/` 保存历史计划、closure 和必要进度笔记。

### `docs/requirements`

需求入口。`SPEC.md` 回答“毕业论文最终要什么、代码仓需要提供什么能力”，`选题报告与基金申请书/` 保存原始 Word 文档，`foundation/` 和 `model-contracts/` 保存基础契约。

### `docs/artifacts`

产物入口。`ARTIFACTS.md` 负责索引报告、图、CSV、JSON、checkpoint、manifest。`runs/` 按日期和任务组织紧凑产物，`archive/` 保留早期报告与资产，`mid-term/` 保存中期答辩证据包。

### `docs/review`

Code review 产物入口。按阶段保存 review 范围、发现、修复状态、剩余风险和测试结果。

## 4. 兼容路径

为避免历史脚本和测试立即失效，以下旧路径保留为符号链接：

- `docs/planning` -> `docs/implementation/notes`
- `docs/foundation` -> `docs/requirements/foundation`
- `docs/models` -> `docs/requirements/model-contracts`

后续统一使用 `docs/artifacts` 与 `docs/requirements/选题报告与基金申请书`。

## 5. 代码层职责

### `src/chronaris/access`

- 访问 InfluxDB / MySQL。
- 封装查询、分页、时间范围和 sortie 级检索。
- 不承担训练逻辑。

### `src/chronaris/schema`

- 统一数据对象定义。
- 字段映射后的标准列命名。
- 样本级输入输出协议。

### `src/chronaris/dataset`

- 架次读取。
- 时间基准校正。
- 窗口切分。
- 样本构建。
- 标签或任务目标拼接。

### `src/chronaris/models/alignment`

- 双流连续潜态模型。
- 时间对齐损失。
- 物理一致性约束。

### `src/chronaris/models/fusion`

- 事件抽取。
- 因果掩码。
- 非对称跨模态融合。

### `src/chronaris/modeling` 与 `representation`

- `modeling/fusion_encoders`：六方法的统一编码器与当前连续融合适配。
- `modeling/training`：候选筛选、冻结预训练和检查点恢复。
- `representation`：观测批次、有效掩码、归一化、增强与标准化表示合同。
- `models/alignment` 与 `models/fusion` 提供可复用基础组件，当前主线通过编码器组装使用。

### `src/chronaris/feature_export` 与 `evidence`

- `feature_export`：早期特征导出流程和产物读取。
- `evidence`：证据汇总与历史材料生成；当前论文评价编排位于 `evaluation/application_tasks`。

### `src/chronaris/features`

- 融合特征矩阵导出。
- 中间态格式化。
- 特征版本管理。

### `src/chronaris/pipelines`

- 训练、导出、验证主流程编排。
- 连接配置、模块、评测和落盘。

### `src/chronaris/serving`

- 离线或准实时推理接口。
- 批处理/服务化输出边界定义。

### `src/chronaris/evaluation`

- 对比实验。
- 消融实验。
- 案例复盘。
- 指标计算与可解释性检查。
