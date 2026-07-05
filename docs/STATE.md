# Chronaris 当前状态

更新时间：2026-07-05

## 一句话状态

仓库已经完成一次本地命名迁移：当前源码、脚本、测试和可引用产物入口已从阶段编号式命名改为“职责 + 日期 + 意图”的命名。当前工作没有重跑训练、没有改 confirmed metrics、没有推送远端。

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

1. 本轮迁移未 push；如需发布，先复核大范围 rename diff 和 LFS 状态。
2. 后续如继续实验，只在新命名入口下新增 run root，不恢复旧阶段编号入口。
3. 可另起任务处理仿真压力测试、检索任务提升或论文材料化工作。
