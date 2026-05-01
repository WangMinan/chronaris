# Chronaris

`chronaris` 用于承接航空人机异构时序数据的下游研究与原型实现。

仓库内所有 Python 脚本、测试、阶段收口命令默认显式使用 `chronaris` 解释器：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`

当前仓库的目标不是重复做上游接收入库，而是围绕已经进入 InfluxDB 的生理/航电数据和已经进入 MySQL 的业务元数据，逐步实现下面这条链路：

1. 数据访问与统一表达
2. 样本构建与时间基准校正
3. 双流连续对齐建模
4. 因果约束跨模态融合
5. 标准化融合特征输出
6. 面向典型任务的评估与原型验证

建议先读：

- [docs/README.md](D:\code\chronaris\docs\README.md)
- [docs/foundation/project-scope.md](D:\code\chronaris\docs\foundation\project-scope.md)
- [docs/foundation/repo-layout.md](D:\code\chronaris\docs\foundation\repo-layout.md)
- [docs/planning/coding-roadmap.md](D:\code\chronaris\docs\planning\coding-roadmap.md)
- [AGENTS.md](D:\code\chronaris\AGENTS.md)
