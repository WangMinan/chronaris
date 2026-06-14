# P21 LLM preprocessing 对比实验结果摘要

更新时间：2026-06-14

## 一句话结论

P21 已在现有 Stage I weak-label 与 runtime 证据链上完成 A0-A4 本地对比：A1 验证 `333` 条 task entry 接入 LLM context 后 `label_unchanged=true`；A2 仅通过 whitelist 将 semantic query coverage 从 `3` 扩到 `7`；A3 在 `4` 个已有 LLM explanation case 上补齐 prediction、semantic attribution、schema gap note 和 weak-label boundary；A4 生成 `15` 条人工复核 packet，但人工未填写前不写成验证完成。

## 实验设置

| 条件 | 含义 | 本轮实现口径 |
| --- | --- | --- |
| A0 baseline | 不接 LLM context | 使用原始 Stage I task entries、内置 3 条 semantic query 和 runtime case table |
| A1 llm_context | 接入 P20 context | 只 attach context 和 rule review，代码审计 label value / label name 不变 |
| A2 llm_semantic_hints | 接入 LLM hints | 仅允许 `coordination_gap / gap_plus_event / physiology_plus_gap / vehicle_plus_event` recipes |
| A3 llm_runtime_explanation | runtime 解释层 | 对比 runtime case table 与 P20 `runtime_llm_explanations.jsonl` 的四项完整性 |
| A4 human_review_packet | 人工复核材料 | 生成字段语义、weak-label 规则和 schema gap policy 小样本复核表 |

## 结果表

| 指标 | 结果 | 中期写作边界 |
| --- | ---: | --- |
| task entries | `333` | thesis weak-label entries，不是人工真值 |
| LLM context 覆盖率 | `1.000000` | 只新增 context，不改 label |
| label_unchanged | `true` | 由代码逐 entry 检查得出 |
| semantic query count | `3 -> 7` | whitelist 接入；未用 summary 伪造 ranking 重算 |
| runtime explained cases | `4/12` | 有界 P20 explanation subset |
| explained case completeness delta | `0.250000` | runtime explanation 完整性，不是专家复盘真值 |
| human review packet items | `15` | `human_review_completed=false` |

## 可以写进中期报告的表述

在不改写 `risk_proxy / workload_proxy / event_replay_tag` weak-label 值的前提下，DeepSeek P20 预处理 context 已通过 P21 对比实验接入 Stage I 证据链：它提供字段语义、规则复核、whitelisted semantic hints、schema gap policy 和 runtime explanation，主要增量体现在可审计上下文、解释完整性和人工复核材料组织。

## 不能写的表述

- 不能写成 DeepSeek 或 LLM 替代人工标注。
- 不能写成 LLM semantic hints 证明核心因果融合模块。
- 不能写成 human review 已完成；本轮只生成复核表。
- 不能写成 runtime native input 已经 exact；当前仍是 native aligned / canonical exact 边界。

## 后续计划

- 由人工填写 `human_review_packet.csv` 后，再统计可采纳、需复核和冲突项。
- 若要证明 semantic ranking/top attribution 变化，需要基于 Stage H tensor 重新运行带 LLM query specs 的 support，而不是从现有 summary 倒推。
- 继续保持 LLM 输出为 preprocessing context，不进入人工真值或核心因果证据层。

## 资产路径

- 工程 summary：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
- 条件 manifest：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/condition_manifest.json`
- task comparison：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/task_context_comparison.csv`
- semantic comparison：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/semantic_hint_comparison.csv`
- runtime comparison：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/runtime_explanation_comparison.csv`
- human review packet：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/human_review_packet.csv`
- Stage I 工程报告：`docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`
