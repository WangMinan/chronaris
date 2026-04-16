# Repo Layout

## 1. 分层原则

这个仓库建议按“可复用库代码”与“运行入口”分开，而不是按一次次实验胡乱堆脚本。

核心原则：

1. `src/chronaris` 只放可复用模块。
2. `configs` 只放可复现实验或运行配置。
3. `scripts` 只放一次性或轻量运维脚本，不承载核心业务逻辑。
4. `experiments` 记录实验方案、结论和对比说明，不替代正式代码。
5. `docs` 记录范围、事实、ADR、方法拆解与验证结论。

当前建议在 `docs` 内继续细分为：

1. `foundation`
2. `planning`
3. `models`
4. `reports`

## 2. 推荐目录

```text
chronaris/
├─ AGENTS.md
├─ README.md
├─ configs/
├─ docs/
│  ├─ foundation/
│  ├─ planning/
│  ├─ models/
│  ├─ reports/
│  └─ README.md
├─ experiments/
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

## 3. 各层职责

### `src/chronaris/access`

- 访问 InfluxDB / MySQL
- 封装查询、分页、时间范围和 sortie 级检索
- 不承担训练逻辑

### `src/chronaris/schema`

- 统一数据对象定义
- 字段映射后的标准列命名
- 样本级输入输出协议

### `src/chronaris/dataset`

- 架次读取
- 时间基准校正
- 窗口切分
- 样本构建
- 标签或任务目标拼接

### `src/chronaris/models/alignment`

- 双流连续潜态模型
- 时间对齐损失
- 物理一致性约束

### `src/chronaris/models/fusion`

- 事件抽取
- 因果掩码
- 非对称跨模态融合

### `src/chronaris/features`

- 融合特征矩阵导出
- 中间态格式化
- 特征版本管理

### `src/chronaris/pipelines`

- 训练、导出、验证的主流程编排
- 连接配置、模块、评测和落盘

### `src/chronaris/serving`

- 后续近实时推理接口
- 批处理/服务化输出的边界定义

### `src/chronaris/evaluation`

- 对比实验
- 消融实验
- 案例复盘
- 指标计算与可解释性检查

## 4. 初始化时不建议做的事

1. 先写一堆 notebook 再回头整理。
2. 把数据库查询直接散落在训练脚本里。
3. 让模型代码同时处理 schema 清洗、样本切分和指标计算。
4. 为了省事把“实验专用逻辑”写死进基础模块。
