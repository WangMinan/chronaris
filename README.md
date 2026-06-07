# Chronaris

`chronaris` 用于承接航空人机异构时序数据连续对齐与语义融合的下游研究与原型实现。

仓库内 Python 脚本、测试、阶段收口命令默认显式使用 `chronaris` 解释器：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

当前仓库不重复做上游接收入库，而是围绕已经进入 InfluxDB 的生理/航电数据和已经进入 MySQL 的业务元数据，推进数据访问、样本构建、双流连续对齐、物理约束、因果融合、标准化特征输出和典型任务验证。

建议先读：

- [AGENTS.md](AGENTS.md)
- [docs/README.md](docs/README.md)
- [docs/STATE.md](docs/STATE.md)
- [docs/implementation/TASKS.md](docs/implementation/TASKS.md)
- [docs/requirements/SPEC.md](docs/requirements/SPEC.md)
- [docs/artifacts/ARTIFACTS.md](docs/artifacts/ARTIFACTS.md)
