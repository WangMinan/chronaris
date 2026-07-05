# 2026-07-05 Name Migration Cleanup

本轮执行本地命名迁移，不推送、不重跑重型实验、不改 confirmed metrics。

完成内容：

- `src/`、`scripts/`、`tests/` 按职责目录重组。
- 当前产物根迁移到 `docs/artifacts/runs/YYYY-MM-DD_intent/`。
- 历史阶段编号目录、旧报告入口和兼容 symlink 归档到 `docs/artifacts/archive/`。
- 当前 run 文本载荷中的旧路径已回写到新路径。
- 补充 `docs/artifacts/README.md` 与 `docs/maintenance/2026-07-05_name-migration-map.md`。

保留边界：

- 核心 CSV、JSON、PNG、MD 未删除，只移动或改名。
- 历史编号仍可在 `docs/artifacts/archive/`、`docs/artifacts/cleanup/` 和迁移表中追溯。
- 本轮未修改 confirmed metric 数值。

已验证：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall src scripts tests`
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q`，结果为 `213 passed, 8 skipped, 78 warnings`
- 活动路径和文本命名审计，旧编号只保留在 archive、cleanup 和迁移表
- 入口文档路径存在性检查，46 个真实路径，0 缺失
- `git diff --check`
- `git lfs status`
- `git lfs fsck`
