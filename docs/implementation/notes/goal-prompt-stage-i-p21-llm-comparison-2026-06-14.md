# Goal Prompt: Stage I LLM 预处理对比 LLM Preprocessing Comparison

请在 `/home/wangminan/projects/chronaris` 仓库继续推进 Stage I LLM 预处理对比。开始前必须先读：

- `AGENTS.md`
- `docs/STATE.md`
- `docs/implementation/TASKS.md`
- `docs/artifacts/ARTIFACTS.md`
- `docs/midterm/README.md`
- `docs/midterm/llm-preprocessing-comparison-plan-2026-06-14.md`

## 目标

实现并真实运行 LLM 预处理对比：LLM preprocessing 融入 Stage I 数据融合管线的对比实验。最终必须生成工程资产和中期报告可直接引用的 `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`。

## 当前事实

- DeepSeek 在线时序数据预处理已完成真实切片 run：
  - `docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
  - `docs/artifacts/stage_i/stage-i-llm-preprocessing-20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced.md`
- DeepSeek 时序预处理 只能作为 preprocessing context / rule review / semantic hints / runtime explanation；不能写成人工真值、OpenAI 默认接入、原始全量高频时序外发或核心因果证据。
- 当前 Stage I 代码已按目录拆分：
  - LLM pipeline：`src/chronaris/pipelines/stage_i/llm/`
  - LLM CLI：`scripts/stage_i/llm/`
  - evidence：`src/chronaris/pipelines/stage_i/evidence/`
  - training：`src/chronaris/pipelines/stage_i/training/`

## 实施要求

新增或复用以下入口：

- `src/chronaris/pipelines/stage_i/llm/comparison.py`
- `src/chronaris/pipelines/stage_i/llm/comparison_reporting.py`
- `scripts/stage_i/llm/run_preprocessing_comparison.py`
- `tests/test_stage_i_llm_comparison.py`

默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## 实验条件

实现 A0-A4 对比：

- `A0 baseline`：不接 LLM context；复用当前 Stage I task entries 与内置 semantic query bank。
- `A1 llm_context`：attach DeepSeek 时序预处理 context 到 task entries；标签值必须保持不变，并输出 `label_unchanged=true` 的代码检查结果。
- `A2 llm_semantic_hints`：只通过 whitelisted recipes 接入 LLM semantic hints，对比 query coverage、view ranking、attribution。
- `A3 llm_runtime_explanation`：比较 runtime cases 有/无 LLM explanation 的报告完整性。
- `A4 human_review_packet`：生成字段/规则/schema gap 小样本人工复核表；人工未填写前只写成 review packet，不写成验证完成。

## 默认输入

- DeepSeek 时序预处理 context：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_context.json`
- task manifest：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- semantic support summary：`docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/support_summary.json`
- runtime schema contract：`docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json`
- runtime semantic case table：`docs/artifacts/assets/stage_i_thesis_figures/20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv`

## 必须输出

工程资产：

- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/condition_manifest.json`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/task_context_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/semantic_hint_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/runtime_explanation_comparison.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/human_review_packet.csv`
- `docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/midterm_claims_payload.json`
- `docs/artifacts/stage_i/stage-i-llm-comparison-20260614T-stage-i-p21-llm-comparison-r1.md`

中期写作资产：

- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`
- 更新 `docs/midterm/README.md`
- LLM 预处理对比 真实完成后再更新 `docs/midterm/claims-matrix-2026-06-13.md`
- 如暴露新边界，更新 `docs/midterm/boundaries-and-risks-2026-06-13.md`

## 验收

- A0-A4 全部有本地可追溯记录；无法完成的条件必须写成 `partial`。
- `label_unchanged=true` 由代码检查生成。
- LLM semantic hints 必须通过 recipe whitelist。
- runtime explanation completeness 至少覆盖 `model_prediction / semantic_attribution / schema_gap_note / weak_label_boundary`。
- `docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md` 必须面向中期报告写作，而不是命令日志。
- 运行相关单测。
- 运行 `git diff --check`。
- 不泄露 `docs/SECRETS.md` 中任何 secret。

## 最终回答要求

最终回复请给出：

- 实现了哪些代码入口。
- A0-A4 的真实运行结果摘要。
- 工程产物路径。
- `docs/midterm/` 写作结果路径。
- 测试命令与结果。
- 明确说明 LLM 输出仍不是人工真值或核心因果证据。
