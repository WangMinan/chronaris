# Stage I P20 DeepSeek 在线时序数据预处理计划

更新时间：2026-06-14

## 定位

本文件记录中期前新增的 DeepSeek 在线大模型时序数据预处理方案。当前提交只更新文档，不修改 `src/`、`scripts/`、`tests/`，也不产生新的实验资产。

P20 的目标是在现有 MySQL / InfluxDB 私有数据链路上增加一个可审计的在线 LLM 预处理层，使中期报告能够真实写入“大模型辅助时序数据预处理”内容。中期默认 provider 为 DeepSeek，不以 OpenAI 作为默认路径，原因是用户明确提出 OpenAI 可能存在信息安全顾虑。

## 现有数据基础

P20 不新建上游接收器，不重建入库链路，复用当前已经闭环的私有数据和 Stage H / Stage I 资产：

- MySQL：业务元数据、飞行批次日期、sortie / pilot / view 关系、飞机字段标签与 measurement metadata。
- InfluxDB：已经入库的生理流和飞机时序流，当前通过 Stage H 导出为双流窗口样本。
- 当前中期私有数据规模：`2` 个 sortie、`3` 个双流 view、`111` 个窗口样本。
- 当前主要 sortie：
  - `20251005_四01_ACT-4_云_J20_22#01`
  - `20251002_单01_ACT-8_翼云_J16_12#01`
- 当前 runtime 单 view smoke 输入：`37` 个窗口样本，native vehicle features 为 `965`，checkpoint 期望 `1930`。
- 当前 schema gap 已定位到 BUS measurement groups `BUS6000019110021` 到 `BUS6000019110026`。
- 当前刚体字段核验已确认 `BUS6000019110020.code1030/code1031/code1032` 可对应 pitch / yaw / roll 角度，但缺少成对角速度字段。

## 为什么中期优先 DeepSeek

1. 用户当前可用 DeepSeek v4-pro 在线 API，并明确要求中期先做 DeepSeek 版本。
2. DeepSeek API 与 OpenAI SDK 兼容，后续可保留 provider 抽象，但默认不发往 OpenAI。
3. 中期阶段需要一个能真实调用、能落盘、能解释输出边界的在线 LLM 能力，而不是只在报告里写“可探索”。
4. DeepSeek 接入应服务于预处理、字段语义、规则复核和结果解释，不替代现有双流连续对齐模型、物理约束和因果融合主线。

## P20 建议拆分

### P20-A：DeepSeek provider contract

目标：形成在线 LLM 调用的最小工程契约。

建议未来代码落点：

- `src/chronaris/llm/provider.py`
- `src/chronaris/llm/schemas.py`
- `src/chronaris/llm/prompts.py`
- `src/chronaris/pipelines/stage_i/llm/preprocessing.py`
- `scripts/stage_i/llm/run_preprocessing.py`

建议配置：

```bash
CHRONARIS_LLM_PROVIDER=deepseek
CHRONARIS_LLM_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=...
```

要求：

- API key 只从环境变量或远端 secret 读取，不写入 repo。
- 请求和响应必须落盘到 `docs/artifacts/assets/stage_i_llm_preprocessing/<run_id>/`。
- 每条请求必须记录 `run_id`、`provider`、`model`、prompt version、input hash、output schema version、token / latency / retry summary。
- 失败时输出 `llm_error_cases.json`，不能把在线失败包装成成功。

### P20-B：字段语义归一

目标：用 DeepSeek 读取 MySQL 字段标签、Stage H feature schema 和少量统计摘要，生成字段语义字典。

输入建议：

- MySQL 字段 label、measurement id、code id。
- Stage H / runtime schema 中的 physiology / vehicle feature names。
- 每个字段的窗口级统计摘要，如 count、min、max、std、delta、missing ratio。

默认不发送：

- 原始高频生理流完整数值。
- 原始长序列飞机流完整数值。
- 未脱敏的人员身份字段。

输出建议：

- `llm_field_semantics.jsonl`
- `field_semantic_dictionary.csv`
- `field_semantic_audit.md`

字段语义结构建议：

| 字段 | 说明 |
| --- | --- |
| stream_kind | `physiology` 或 `vehicle` |
| measurement_group | BUS group 或 physiology group |
| feature_name | Stage H / runtime feature name |
| llm_semantic_role | speed / attitude / vertical / physiological_load / signal_quality 等 |
| llm_unit_guess | 单位候选，不作为真实单位 |
| evidence | MySQL label、feature name、统计摘要 |
| confidence | 0 到 1 |
| needs_human_review | 是否需要人工确认 |

### P20-C：weak-label 规则复核

目标：让 DeepSeek 对当前 `risk_proxy / workload_proxy / event_replay_tag` 的规则构造给出候选解释、字段权重和冲突样本说明，再由程序化代码做对比。

输入建议：

- `stage_i_real_task_builders.py` 当前规则摘要。
- 111 个窗口样本的聚合统计，不发送完整原始序列。
- 当前 weak-label 输出分布、sample partition、event tag distribution。
- 当前 semantic support 的 query 和 event attribution 摘要。

输出建议：

- `llm_weak_label_review.jsonl`
- `weak_label_llm_comparison.csv`
- `weak_label_conflict_cases.csv`
- `stage-i-llm-weak-label-review-<run_id>.md`

评估指标：

| 指标 | 目的 |
| --- | --- |
| label_agreement_rate | DeepSeek 候选规则与当前规则的一致率 |
| conflict_count | 需要人工复核的窗口数 |
| high_confidence_conflict_count | 高置信冲突窗口数 |
| field_weight_shift | LLM 建议权重相对当前规则的变化 |
| downstream_delta | 用候选规则复跑小网格后的指标变化，若中期有预算再做 |

边界：

- DeepSeek 输出不是人工真值。
- LLM 建议规则必须通过 schema 校验、范围校验和人工抽检后才能进入训练。
- 中期可写“已规划或实现 LLM 规则复核层”，不能把它写成“真实标签问题已经解决”。

### P20-D：缺失与 schema gap 预处理建议

目标：围绕当前 runtime native aligned 但不是 exact 的问题，让 DeepSeek 生成可解释的 preprocessing policy 草案。

输入建议：

- `runtime_schema_contract.json` 的 native / canonical schema 对比。
- missing vehicle measurement group counts。
- 当前 `BUS6000019110021` 到 `BUS6000019110026` 缺失事实。
- `runtime_error_cases.json` 中的 expected failure 类型。

输出建议：

- `llm_schema_gap_policy.json`
- `schema_gap_preprocessing_policy.md`
- `runtime_llm_schema_gap_summary.csv`

可接受的策略类型：

- `drop_or_mask`：显式 mask 缺失 measurement group。
- `canonical_fill_nan`：保持 canonical payload exact，但缺失字段填 `NaN` 或 deterministic placeholder。
- `source_requery_required`：要求从 InfluxDB 重新采样缺失 BUS groups。
- `human_review_required`：字段语义或单位无法确认时进入人工复核。

禁止策略：

- 不允许 DeepSeek 直接编造缺失 BUS groups 的真实数值。
- 不允许把 canonical exact 写成原始 native exact。
- 不允许绕过 runtime schema contract。

### P20-E：runtime 与语义事件解释

目标：把 runtime prediction、semantic event attribution、weak-label task 和 schema gap 组织成窗口级自然语言解释，用于中期报告案例和答辩讲解。

输入建议：

- `runtime_inference_summary.json`
- `runtime_semantic_case.csv`
- `support_summary.json`
- `runtime_schema_contract.json`
- 当前 top view 的 12 个展示窗口。

输出建议：

- `runtime_llm_explanations.jsonl`
- `runtime_llm_case_table.csv`
- `stage-i-llm-runtime-explanation-<run_id>.md`

输出要求：

- 每条解释必须包含 source paths。
- 每条解释必须区分 `model_prediction`、`semantic_attribution`、`schema_gap_note`、`weak_label_boundary`。
- 不把解释写成专家复盘结论。

## 默认请求粒度

中期前建议使用三类 payload，而不是把原始全量数据直接发给在线模型：

1. `schema_card`：字段名、MySQL label、measurement group、统计范围、缺失率。
2. `window_summary_card`：窗口 offset、top changed fields、weak-label 当前输出、语义事件 attribution 摘要。
3. `runtime_case_card`：prediction、confidence、schema status、error case、source path。

若后续需要发送短片段原始序列，必须额外记录：

- 片段选择原因。
- 时间范围。
- 字段白名单。
- 脱敏策略。
- 是否经过人工批准。

## 中期报告可写口径

在完成 P20 代码和一次真实 DeepSeek 调用前，只能写：

> 已制定 DeepSeek 在线大模型辅助时序数据预处理方案，计划接入现有 MySQL / InfluxDB 派生的 Stage H schema、窗口统计和 runtime 证据，用于字段语义归一、weak-label 规则复核、schema gap 预处理建议和 runtime 结果解释。

在 P20 完成真实 run 后，可以写：

> 已实现基于 DeepSeek 的在线 LLM 预处理模块，并在当前 2 个 sortie、3 个双流 view、111 个窗口样本的 Stage H / Stage I 证据链上完成字段语义、weak-label 复核、schema gap 建议和 runtime 解释的落盘审计。

禁止写：

- “DeepSeek 已替代人工标注。”
- “DeepSeek 已解决因果推断。”
- “LLM 输出就是 ground truth。”
- “OpenAI API 已接入中期主线。”
- “原始全量 MySQL / InfluxDB 数据已直接发送给大模型。”

## 与现有文献的连接

- L01 / L03：支撑时序与自然语言语义对齐，可对应字段语义和 runtime case explanation。
- L02：支撑语言模型参与时序插补策略讨论，但本项目只允许 LLM 生成 preprocessing policy，不允许直接编造真实缺失值。
- L04：支撑 LLM agent 辅助表征学习，可作为后续扩展，不作为中期 P20 主线。
- L05：支撑自动标注和程序化标注函数，最适合对应 P20-C 的 weak-label 规则复核。

## 验收标准

P20 真正进入“已完成”需要满足：

1. 有 mock provider 测试，离线环境不依赖真实 DeepSeek API。
2. 有一次真实 DeepSeek 小样本 run，建议覆盖 3 个 view 和不超过 111 个窗口摘要。
3. 所有请求、响应、错误、成本和 prompt version 落盘。
4. 产物进入 `docs/artifacts/assets/stage_i_llm_preprocessing/<run_id>/`。
5. 报告进入 `docs/artifacts/stage_i/stage-i-llm-preprocessing-<run_id>.md`。
6. `docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/artifacts/ARTIFACTS.md` 和 `docs/midterm/claims-matrix-*.md` 同步更新。

## 当前状态

截至 2026-06-14，本文件只是 P20 计划和文档口径冻结。尚未更新代码，尚未调用 DeepSeek，尚未生成 `stage_i_llm_preprocessing` 运行产物。
