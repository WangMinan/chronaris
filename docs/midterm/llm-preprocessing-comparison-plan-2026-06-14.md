# P21 LLM 预处理对比实验计划

更新时间：2026-06-14

本计划用于承接 P20 DeepSeek 在线时序数据预处理之后的下一轮实验。目标不是证明 LLM 替代人工标注，也不是把 LLM 变成核心因果模块；目标是用对比实验说明：把 LLM preprocessing context 接入现有 Stage I 数据融合管线后，在哪些环节带来了可审计、可解释、可复核的增量价值。

## 1. 实验问题

本轮要回答四个问题：

1. `LLM context` 接入 task builder 后，是否能在不改写 weak-label 标签值的前提下，补充字段语义、规则复核和 schema gap 说明。
2. `LLM semantic hints` 接入 semantic event support 后，是否能扩展可解释 query 覆盖，并改变或补充 view-level attribution。
3. `LLM runtime explanations` 是否能让 runtime replay 报告更完整地说明 prediction、semantic attribution、schema gap 和 weak-label boundary。
4. 小样本人工复核时，LLM 预处理是否减少人工查字段、查规则、解释缺失 schema 的整理成本。

## 2. 固定边界

- 使用 DeepSeek v4-pro 已生成的 P20 `llm_preprocessing_context`，默认不改成 OpenAI。
- LLM 输出只作为 preprocessing context / rule review / semantic hints / runtime explanation。
- 不覆盖 `risk_proxy / workload_proxy / event_replay_tag` 的 label value。
- 不把 weak-label 写成人工真值。
- 不把 canonical exact 写成 native exact。
- 不发送原始全量高频时序；如需要更多字段、窗口或 runtime cases，继续使用切片调用与本地 stable identifier merge。

## 3. 对比设计

| 条件 | 名称 | 控制变量 | 要比较的输出 |
| --- | --- | --- | --- |
| A0 | baseline | 不接 LLM context；使用当前 task entries 和内置 semantic query bank | 原始 task summary、semantic query coverage、runtime report completeness |
| A1 | llm_context | attach `llm_preprocessing_context` 到 Stage I task entries，标签值保持不变 | task entry context 覆盖率、weak-label rule review 覆盖率、label unchanged audit |
| A2 | llm_semantic_hints | 在内置 query bank 之外加入 whitelisted LLM semantic hints | query count、新增 query recipe、view ranking 变化、top attribution 变化 |
| A3 | llm_runtime_explanation | 给 runtime cases 增加 LLM explanation 层 | explanation completeness、schema gap note 覆盖、weak-label boundary 覆盖 |
| A4 | human_review_packet | 抽取字段/规则/schema gap 小样本，生成复核表 | 人工复核 item 数、LLM 建议可采纳/需复核/冲突分类、复核成本说明 |

## 4. 建议实现入口

新增或复用代码时保持当前目录组织：

- pipeline：`src/chronaris/pipelines/stage_i/llm/`
- CLI：`scripts/stage_i/llm/`
- tests：`tests/test_stage_i_llm_preprocessing.py` 或新增 `tests/test_stage_i_llm_comparison.py`

建议新增入口：

- `src/chronaris/pipelines/stage_i/llm/comparison.py`
- `src/chronaris/pipelines/stage_i/llm/comparison_reporting.py`
- `scripts/stage_i/llm/run_preprocessing_comparison.py`

默认输入：

- P20 context：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
- Stage I weak-label manifest：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- semantic support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- runtime schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- runtime semantic case table：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`

## 5. 必须落盘的工程产物

建议 run id：

`20260614T-stage-i-p21-llm-comparison-r1`

工程资产目录：

`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/`

必须包含：

- `llm_comparison_summary.json`
- `condition_manifest.json`
- `task_context_comparison.csv`
- `semantic_hint_comparison.csv`
- `runtime_explanation_comparison.csv`
- `human_review_packet.csv`
- `midterm_claims_payload.json`
- `progress.json`
- `run.log`

Stage I 工程报告：

`docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`

## 6. 必须落到 midterm 的结果

P21 不以工程报告结束。完成判据必须包括 `docs/midterm/` 下的写作级材料：

- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
- 更新 `docs/midterm/README.md`，把该 summary 加入当前入口。
- 更新 `docs/midterm/claims-matrix-2026-06-13.md`，只在 P21 真实跑完后新增或升级 “LLM 接入带来可解释性/复核效率增量” claim。
- 如发现新的边界，更新 `docs/midterm/boundaries-and-risks-2026-06-13.md`，保持 “LLM 不替代人工真值”。

`llm-preprocessing-comparison-summary-2026-06-14.md` 必须面向中期报告，而不是命令日志。建议结构：

1. 一句话结论。
2. 实验设置：A0-A4 条件、输入数据、样本范围。
3. 结果表：task context、semantic hints、runtime explanation、human review packet。
4. 可以写进中期报告的表述。
5. 不能写的表述。
6. 后续计划。

## 7. 验收标准

- 所有 A0-A4 条件都要有本地可追溯输出；如果某一条件无法完成，必须在 summary 中写成 `partial` 并说明原因。
- `label_unchanged=true` 必须由代码检查得出，不允许口头声明。
- LLM semantic hints 必须经过 recipe whitelist；不允许自由文本 prompt 进入融合模块。
- runtime explanation completeness 必须至少检查四项：`model_prediction`、`semantic_attribution`、`schema_gap_note`、`weak_label_boundary`。
- human review packet 只生成复核材料；没有人工填写前，不得写成人工验证已经完成。
- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md` 必须存在，且引用工程资产路径。
- `git diff --check` 通过。

## 8. 中期报告建议表述

P21 完成后可以考虑写：

> 在现有 Stage I weak-label 与 runtime 证据链上，本文引入 DeepSeek 在线 LLM 预处理作为可审计上下文层。对比实验保持 weak-label 标签值不变，将 LLM 输出限制为字段语义归一、规则复核、schema gap policy、语义 query hint 和 runtime explanation。结果用于评估 LLM 对数据预处理可解释性、规则复核效率和报告完整性的增量价值，而不作为人工真值或核心因果证据。

P21 完成前只能写：

> 已完成 P20 小样本真实 DeepSeek preprocessing context，下一步将通过 A0-A4 对比实验评估 LLM 接入对现有融合管线的增量价值。
