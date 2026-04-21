# Scripts

这里放一次性或轻量脚本，例如：

- 数据抽样
- 临时核验
- 批量导出
- 环境检查

如果脚本里出现了可复用逻辑，应回收到 `src/chronaris`，脚本只保留组装入口。

当前已提供：

- `run_stage_e_relative_preview.py`
  - 用 overlap-focused 配置直接运行一轮 Stage E baseline / Stage F(min) preview
  - 支持 `none / zscore_train` 两种输入归一化模式（`--input-normalization-mode`）
  - 默认可按 Stage E 基线运行；开启 `--enable-physics-constraints` 后进入 Stage F(min)
  - 支持 `feature_first_with_latent_fallback / feature_only / latent_only` 约束模式
  - 支持设置 `--vehicle-physics-weight`、`--physiology-physics-weight`、`--physics-huber-delta`
  - 支持设置生理包络分位 `--physiology-envelope-quantile`
  - 自动从 `CHRONARIS_INFLUX_*` 或 `docs/SECRETS.md` 解析 Influx 连接信息
  - 输出报告到 `docs/reports/` 并打印 JSON 摘要
  - 自动追加 `Physics Constraint Diagnostics` 区块
  - 自动追加样本级投影诊断区块（`Sample-Level Projection Diagnostics`）
  - 支持诊断阈值模板与判定输出（`PASS / WARN`）
  - 输出诊断产物到 `docs/reports/assets/<report-stem>/`：
    - `projection_diagnostics_summary.json`
    - `projection_diagnostics_samples.csv`
  - 额外输出可视化图片到 `docs/reports/assets/<report-stem>/`
    - physics 开启时额外输出：
      - `train_validation_physics_loss.png`
      - `constraint_component_breakdown.png`
  - 自动把图片链接追加到报告的 `Visual Artifacts` 区域
  - 自动导出模型 checkpoint 到 `docs/reports/assets/<report-stem>/alignment_model_checkpoint.pt`
  - 可选一次运行完成 `none` 与 `zscore_train` 对照（`--compare-with-zscore-train`）
