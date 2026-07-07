# Chronaris 当前状态

更新时间：2026-07-06

## 一句话状态

命名迁移已完成。本轮新增“融合表示流结构评价（E3）”开发计划（`docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/`），只产出计划文档：未编码、未训练、未改 confirmed metrics。后续编码需人工 review 本计划后再执行。

## 当前入口

- 代码：`src/chronaris/feature_export/`、`src/chronaris/modeling/`、`src/chronaris/evaluation/dingxin/`、`src/chronaris/evaluation/public_datasets/`、`src/chronaris/evidence/`、`src/chronaris/runtime/`、`src/chronaris/llm_preprocessing/`。
- 脚本：`scripts/feature_export/`、`scripts/modeling/`、`scripts/evaluation/dingxin/`、`scripts/evaluation/public_datasets/`、`scripts/evidence/`、`scripts/runtime/`、`scripts/llm_preprocessing/`。
- 测试：`tests/feature_export/`、`tests/modeling/`、`tests/evaluation/dingxin/`、`tests/evaluation/public_datasets/`、`tests/evidence/`、`tests/runtime/`、`tests/llm_preprocessing/`。
- 产物：`docs/artifacts/runs/`，当前协议入口为 `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/`。

历史阶段编号目录和旧报告入口已经移入 `docs/artifacts/archive/`；迁移表见 `docs/maintenance/2026-07-05_name-migration-map.md`。

## 证据边界

- 鼎新真实数据仍是现有弱监督构造任务和组件诊断证据，不包装成人工专家真值。
- 公开 UAB/NASA 结果仍是公开数据适配、公开数据校准和上下文构造第二输入流证据。
- 指标校准只采纳既有固定 reference 下的局部改善；检索任务继续沿用任务感知头优化的 confirmed retrieval 口径。
- LLM preprocessing 只作为字段语义归一、规则复核、运行解释和人工复核材料组织。

## 本轮本地整理

- 已完成 `src/`、`scripts/`、`tests/` 职责目录迁移。
- 已将当前核心 artifact 根迁移到 `docs/artifacts/runs/YYYY-MM-DD_intent/`。
- 已将历史阶段编号报告、旧资产目录和兼容 symlink 迁入 `docs/artifacts/archive/`。
- 已补充 `docs/artifacts/README.md` 和迁移记录。
- 已完成验证：`compileall`、全量 `pytest -q`、活动路径/文本命名审计、artifact manifest/resume command 旧路径审计、入口文档路径存在性检查、`git diff --check`、`git lfs status`、`git lfs fsck`。

## 后续队列

1. 人工 review `docs/artifacts/runs/2026-07-06_fusion-stream-structure-plan/` 计划；通过后按 `implementation_plan.md` 阶段执行 E3 编码（contracts→clasp→stumpy→CLI→可选 ticc）。
2. E3 编码必须遵守：只在新 run root 产出、外部库 gated import + fallback、不回写论文协议快照、不与 T1/T2/T3 confirmed metrics 混算、不删 T3 artifact。
3. 后续如继续实验，只在新命名入口下新增 run root，不恢复旧阶段编号入口。
4. 可另起任务处理仿真压力测试、检索任务提升或论文材料化工作。
