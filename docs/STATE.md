# Chronaris 当前状态

更新时间：2026-07-07

## 一句话状态

“融合表示流结构评价（E3）”第二轮 evaluator validation 与第三方来源审计已完成：`claspy 0.2.8` 与 `stumpy 1.14.1` 已在 `chronaris` 环境可 import；ClaSP 与 STUMPY 在合成数据和小规模 Dingxin dry run 均完成；MulT / ContiFormer 在当前 Dingxin artifact 中没有可复用融合表示流或可加载 checkpoint，Dingxin E3 暂保持 two-method validation；未训练、未改 confirmed metrics、未回写论文协议快照。

## 当前入口

- 代码：`src/chronaris/feature_export/`、`src/chronaris/modeling/`、`src/chronaris/evaluation/dingxin/`、`src/chronaris/evaluation/fusion_stream_structure/`、`src/chronaris/evaluation/public_datasets/`、`src/chronaris/evidence/`、`src/chronaris/runtime/`、`src/chronaris/llm_preprocessing/`。
- 脚本：`scripts/feature_export/`、`scripts/modeling/`、`scripts/evaluation/dingxin/`、`scripts/evaluation/fusion_stream_structure/`、`scripts/evaluation/public_datasets/`、`scripts/evidence/`、`scripts/runtime/`、`scripts/llm_preprocessing/`。
- 测试：`tests/feature_export/`、`tests/modeling/`、`tests/evaluation/dingxin/`、`tests/evaluation/fusion_stream_structure/`、`tests/evaluation/public_datasets/`、`tests/evidence/`、`tests/runtime/`、`tests/llm_preprocessing/`。
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
- 已完成 E3 第一批实现、第二轮 evaluator-enabled 验证和 MulT / ContiFormer 来源审计：合同/加载/预处理、ClaSP/CLaP gated wrapper、STUMPY Matrix Profile wrapper、结构指标、报告、CLI、合成数据 validation run、小规模 Dingxin validation run，以及 `docs/artifacts/runs/2026-07-07_fusion-stream-thirdparty-source-audit/`。

## 后续队列

1. CLaP 在当前短序列 validation 中未稳定形成多状态标签，已按 `clap_unavailable` 保留；后续若扩大样本或更换固定协议，必须另起 run root 并记录判据。
2. 小规模 Dingxin validation 当前只发现 `chronaris` 与 `naive_time_sync` 有可复用融合表示流；source audit 结论为 `C. no_reusable_sources`，`mult` / `contiformer` 缺少可复用融合表示流和可加载 checkpoint，已按 `method_unavailable` 记录。
3. E3 后续实验仍必须只在新 run root 产出，不回写论文协议快照，不与分类、回归和历史检索 confirmed metrics 混算，不删除历史检索 artifact。
4. 可另起任务处理仿真压力测试、历史检索任务提升或论文材料化工作。
