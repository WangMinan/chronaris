# Chronaris 协作说明

> **codex 编码八荣八耻**
> - 以瞎猜接口为耻，以认真查询为荣。
> - 以模糊执行为耻，以寻求确认为荣。
> - 以臆想业务为耻，以人类确认为荣。
> - 以创造接口为耻，以复用现有为荣。
> - 以跳过验证为耻，以主动测试为荣。
> - 以破坏架构为耻，以遵循规范为荣。
> - 以假装理解为耻，以诚实无知为荣。
> - 以盲目修改为耻，以谨慎重构为荣。

## 1. 项目定位

`chronaris` 是“航空人机异构时序数据连续对齐与语义融合”仓库，服务于基于 [论文选题报告表](docs/requirements/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx) 的毕业设计。

本仓库默认承接下游研究与原型实现、数据读取组织、建模、导出、验证；默认不承接历史接收器重写、上游入库链路重建、原始大数据文件入仓。

## 2. 文档事实源

后续协作不要把阶段状态继续压在 `AGENTS.md`。请按下面入口读取事实：

- 总导航：[docs/README.md](docs/README.md)
- 当前状态：[docs/STATE.md](docs/STATE.md)
- 论文需求与仓库能力：[docs/requirements/SPEC.md](docs/requirements/SPEC.md)
- 执行入口与当前任务队列：[docs/implementation/TASKS.md](docs/implementation/TASKS.md)
- 进度与历史计划笔记：[docs/implementation/notes/README.md](docs/implementation/notes/README.md)
- 产物索引：[docs/artifacts/ARTIFACTS.md](docs/artifacts/ARTIFACTS.md)
- Review 产物入口：[docs/review/REVIEW.md](docs/review/REVIEW.md)

`docs/planning`、`docs/foundation`、`docs/models` 仍保留为兼容入口；AI coding 的当前入口以 `docs/implementation`、`docs/artifacts`、`docs/requirements` 为准。

## 3. 研究主线

默认沿这条链路推进：

1. 读取指定架次的人机多源数据及元信息。
2. 建立统一 schema、统一时间参考和统一样本组织。
3. 实现双流连续潜态建模。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合。
6. 输出标准化融合特征与中间态接口。
7. 面向空中失能风险分析、认知负荷评估、飞行事件复盘做对比、消融和案例验证。

## 4. 代码边界

新增可复用逻辑默认进入 `src/chronaris`：

- `access`：InfluxDB / MySQL 访问。
- `schema`：统一 schema。
- `dataset`：样本组织、窗口切分、时间基准。
- `models/alignment`：连续对齐与物理约束。
- `models/fusion`：因果融合与事件级语义融合。
- `features`：特征导出与中间态。
- `pipelines`：训练、导出、验证流程。
- `serving`：离线或准实时推理接口。
- `evaluation`：对比、消融、案例分析。

约束：

- `scripts` 只承载 CLI 编排，不承载核心业务实现。
- notebook 只用于探索，不能成为唯一事实来源。
- `docs` 默认使用中文。
- 大于 500 行的文件建议拆分，大于 800 行的文件必须拆分。

## 5. 环境约定

后续编码、测试、阶段脚本实跑默认使用服务器/WSL 的 `chronaris` conda 环境：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

不要因为 shell 停在 `base` 就直接运行 `python`。如需 MySQL 密码、InfluxDB token、连接串或 sudo 信息，可参考已被 `.gitignore` 纳管的 `docs/SECRETS.md`，但不要把其中内容写入其他文件。

当前数据库服务默认按 Docker 映射到本机端口处理：MySQL `127.0.0.1:3306`，InfluxDB `127.0.0.1:8086`。

## 6. 执行规则

- 开始任何阶段工作前，先读 [docs/STATE.md](docs/STATE.md) 和 [docs/implementation/TASKS.md](docs/implementation/TASKS.md)。
- 需要判断论文目标或 Word 原始材料时，先读 [docs/requirements/SPEC.md](docs/requirements/SPEC.md)；解析 `.docx` 必须使用 `$docx` skill 或文档插件。
- 需要引用报告、图、CSV、checkpoint、manifest 时，先读 [docs/artifacts/ARTIFACTS.md](docs/artifacts/ARTIFACTS.md)。
- 需要做 code review 时，把计划和结果落到 [docs/review/REVIEW.md](docs/review/REVIEW.md) 及对应阶段子目录。
- 每轮实现默认按 `目标锁定 -> 代码实现 -> 测试闭环 -> 文档回写 -> 冗余清理` 收敛。
- 阶段收口必须满足：真实实跑、判据可复现、测试通过、状态文档一致。
