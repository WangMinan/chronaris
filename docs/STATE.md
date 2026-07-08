# Chronaris 当前状态

更新时间：2026-07-08

## 一句话状态

“融合表示流结构评价（E3）”已完成 Dingxin deep baseline representation export、四方法 Dingxin validation、结果审查和 Chronaris 受控候选优化：本轮只在 `docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/` 训练 MulT 与 ContiFormer，按回归任务（`T2_next_window_physiology_response`）+ leave-one-view-out + seed17 导出 held-out pooled embedding；四方法 run `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/` 已完成，`chronaris` / `naive_time_sync` / `mult` / `contiformer` 均可用；审查 run `docs/artifacts/runs/2026-07-08_e3-result-review/` 判定当前 E3 结果为 mixed，论文可用性为 D 类；受控优化 run `docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-dev/` 注册 8 个 Chronaris 候选，8 个 smoke 通过，Pareto 选择 `chr_v2_residual_delta_h64`，并在 `docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-confirm/` 完成 locked confirmation；Chronaris OOF 表示导出见 `docs/artifacts/runs/2026-07-08_chronaris-oof-representation-export/`。本轮未改 confirmed metrics、未回写论文协议快照、未删除历史检索 artifact。

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
- 已完成 E3 第一批实现、第二轮 evaluator-enabled 验证、MulT / ContiFormer 来源审计、deep baseline OOF 表示导出、四方法 Dingxin validation、结果审查和 Chronaris 受控候选优化：合同/加载/预处理、ClaSP/CLaP gated wrapper、STUMPY Matrix Profile wrapper、结构指标、报告、CLI、合成数据 validation run、小规模 Dingxin validation run、来源审计、`docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/`、`docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/`、`docs/artifacts/runs/2026-07-08_e3-result-review/`、`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-dev/`、`docs/artifacts/runs/2026-07-08_chronaris-controlled-optimization-confirm/` 和 `docs/artifacts/runs/2026-07-08_chronaris-oof-representation-export/`。

## 后续队列

1. CLaP 在当前 Dingxin 四方法短序列 validation 中仍未稳定形成多状态标签，已按 `clap_unavailable:12` 保留；后续若扩大样本或更换固定协议，必须另起 run root 并记录判据。
2. 四方法 E3 是结构评价补充证据，不替代分类任务和回归任务；审查结论为旧 E3 不适合作为正文优势证据，只可作为附录诊断或受控优化依据。
3. Chronaris 受控优化 locked candidate `chr_v2_residual_delta_h64` 在新 run root 中改善回归任务并给出一个 E3 motif 正向信号，但仍略弱于固定 MulT / ContiFormer 回归 baseline；论文中只建议作为补充或附录诊断，不直接写成正文主优势结论。
4. E3 后续实验仍必须只在新 run root 产出，不回写论文协议快照，不与分类、回归和历史检索 confirmed metrics 混算，不删除历史检索 artifact。
