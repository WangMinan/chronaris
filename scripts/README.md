# Scripts

这里放一次性或轻量脚本，例如：

- 数据抽样
- 临时核验
- 批量导出
- 环境检查

如果脚本里出现了可复用逻辑，应回收到 `src/chronaris`，脚本只保留组装入口。

当前已提供：

- `run_stage_e_relative_preview.py`
  - 用 overlap-focused 配置直接运行一轮 Stage E relative-mse 回归
  - 自动从 `CHRONARIS_INFLUX_*` 或 `docs/SECRETS.md` 解析 Influx 连接信息
  - 输出报告到 `docs/reports/` 并打印 JSON 摘要
  - 自动追加样本级投影诊断区块（`Sample-Level Projection Diagnostics`）
  - 输出诊断产物到 `docs/reports/assets/<report-stem>/`：
    - `projection_diagnostics_summary.json`
    - `projection_diagnostics_samples.csv`
  - 额外输出可视化图片到 `docs/reports/assets/<report-stem>/`
  - 自动把图片链接追加到报告的 `Visual Artifacts` 区域
