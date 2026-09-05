# Chronaris 中期边界与风险说明

> 历史写作快照：正文中的“当前”指文内原始更新日期；不替代[九月当前状态](../STATE.md)与[冻结实验复核](../review/stage/thesis-mainline/frozen-simulation-review-2026-09-04.md)。

更新时间：2026-06-14

本说明用于中期报告和答辩问答。目标是把当前工作讲清楚、讲稳，不把弱证据包装成强结论，也不因为边界存在就低估已经完成的工程和实验闭环。

## 1. 总边界

当前 Chronaris 中期阶段的强结论是：

- 已经完成从鼎新 MySQL / InfluxDB 数据到 Stage H 双流 view、Stage I weak-label 任务、物理/因果/语义/runtime 支撑证据的闭环。
- 已经形成 `docs/artifacts/` 下可追溯的报告、JSON、CSV、PNG 和 checkpoint。
- 已经把关键风险从口头说明转成 `partial_summary.json`、`runtime_schema_contract.json`、`runtime_error_cases.json` 和图表说明。

当前不能扩写成：

- 人工真值任务完全闭环。
- 全量飞行数据或跨机型泛化验证完成。
- 原始上游 runtime schema 已完全 exact。
- 完整旋转刚体约束已启用。
- public adapter 等价于论文鼎新真实双流主线的直接胜利。
- 把 DeepSeek 时序预处理/LLM 预处理对比 DeepSeek 在线 LLM 预处理与对比实验写成替代人工真值、核心因果证据或人工复核完成。

## 2. 证据层级边界

| evidence layer | 当前用途 | 可以怎么写 | 不能怎么写 |
| --- | --- | --- | --- |
| thesis_weak_label | `risk_proxy / workload_proxy / event_replay_tag` 论文任务原型 | “基于窗口统计构造 weak-label 任务，用于验证端到端建模和消融链路” | “人工标注风险/负荷/事件真值验证完成” |
| dingxin_weak_label | `chronaris_opt`、分类任务、回归任务和检索任务 鼎新真实数据弱监督任务基准 | “用于诊断模块组合、因果掩码、时间残差、任务头的贡献” | “论文最终任务已由 分类任务、回归任务和检索任务 直接证明” |
| public_adapter/calibration | UAB/NASA 公开数据支撑线 | “用于公开数据适配、校准和 transfer boundary 说明” | “公开数据证明鼎新真实航空双流主线 fully closed” |
| rigid_body_support | 物理一致性约束 | “translation + vertical 已启用，rotation 受字段限制保留 diagnostics” | “完整 6DoF 刚体旋转约束已经实现并验证” |
| semantic_support | 语义事件融合 support | “在 3 个双流 view 上形成 view-level ranking 和 attribution” | “专家语义事件复盘标注验证完成” |
| runtime_replay/service_contract | runtime replay、服务化 smoke、schema contract | “native aligned，canonical exact；错误样例与 contract 已固化” | “原始上游输入已 native exact schema” |
| llm_preprocessing_context/comparison | DeepSeek 在线时序数据预处理 context 与 LLM 预处理对比 A0-A4 对比 | “已实现 DeepSeek 在线 LLM 预处理模块，并完成小样本字段语义、weak-label 复核、schema gap policy、runtime 解释、切片整合和 A0-A4 对比落盘审计” | “DeepSeek 已替代人工标注、LLM 输出等同人工真值、证明核心因果结论或完成人工复核” |

## 3. 鼎新弱监督任务扫描 风险：live_influx sweep 的完成度

### 当前事实

- `20260613T-stage-i-p11-live-influx-r3-resume` 是稳定完成版。
- 稳定版 sample_source=`live_influx`，sample_count=`111`，task_entry_count=`333`，combination_count=`2`。
- 该稳定版复用 `20260613T-stage-i-p11-live-influx-r1` 中已经真实完成的两个 child run。
- 更大的尝试没有伪造成 completed；blocker 保留在 run log 和 partial summary。
- `20260613T-stage-i-p11-live-influx-r4-partial` 明确记录 status=`partial_blocked`、completed_child_runs、blocked_at_run_index=`3`。

### 风险

老师可能问：为什么不是更大网格？为什么 live_influx 只有两个组合？

### 回答口径

可以回答：

> 当前中期阶段优先保证真实链路可复现与证据诚实。live_influx 路线已经完成两个真实 child run，并通过 stable/resume 机制固化；更大网格的第 3 个 child run 以后出现 blocker，因此没有伪造成成功，而是保留为 partial_blocked 证据。中期报告把它作为 weak-label sweep 的真实运行边界，而不是完整大规模超参搜索结论。

不要回答：

> 4 个组合都跑完了。

后续计划：

- 若中期后继续扩展，先明确预算，再从当前 2 组合 stable resume 增量补 `rigid_body, cw=0, tlw=0.5, lag=None|3`。
- 保持 `--resume-existing` 与 partial summary，不覆盖现有 stable summary。

## 4. 运行时服务冒烟验证/稳定扫描与字段契约收口 风险：runtime schema aligned 而不是 native exact

### 当前事实

- Runtime smoke r2 contract 成功。
- native_feature_schema_status=`aligned`
- canonical_feature_schema_status=`exact`
- checkpoint 期望 physiology features=`12`，vehicle features=`1930`
- native runtime input 提供 physiology features=`12`，vehicle features=`965`
- native 缺少 `965` 个 vehicle features，集中在 6 个 BUS measurement groups：
  - `BUS6000019110021`
  - `BUS6000019110022`
  - `BUS6000019110023`
  - `BUS6000019110024`
  - `BUS6000019110025`
  - `BUS6000019110026`
- canonical payload 已补齐到 vehicle features=`1930`，达到 service contract exact。

### 风险

老师可能问：既然 native 不是 exact，runtime 是否可靠？是不是上游链路有问题？

### 回答口径

可以回答：

> 当前 runtime 已经能从 checkpoint 冷启动，并对 JSONL replay 样本输出 prediction 与错误样例。真实部署边界是 native input aligned，因为当前单 view replay 只携带一半 vehicle measurement groups；我们没有把它包装成 native exact。为部署接口补充了 canonical payload 与 runtime_schema_contract，因此服务层契约可以达到 exact。后续如果要收紧到 native exact，应补齐上游 view replay payload 的 vehicle measurement groups，而不是重建整个接收器或入库链路。

不要回答：

> 原始上游输入已经 exact。

后续计划：

- 继续保留 `runtime_schema_contract.json` 作为部署契约事实源。
- 若要 native exact，优先补齐 JSONL 采样器输出的 BUS6000019110021-0026 groups。
- 不在中期前重建历史接收器或原始大文件入仓链路。

## 5. 刚体旋转审计 风险：rotation disabled

### 当前事实

- Stage H feature 与 MySQL metadata 已加载。
- metadata measurement=`BUS6000019110020`
- field_count=`96`
- 已找到 pitch、roll、yaw 角度字段：
  - pitch=`BUS6000019110020.code1030`
  - roll=`BUS6000019110020.code1032`
  - yaw=`BUS6000019110020.code1031`
- 缺少 pitch_rate、roll_rate、yaw_rate。
- rotation_enabled=`false`
- rotation_status=`disabled`
- translation + vertical residuals 已启用。

### 风险

老师可能问：题目里强调物理一致性，为什么 rotation 没启用？

### 回答口径

可以回答：

> 当前物理约束不是空的，translation 与 vertical 已经真实启用并进入 rigid_body family。rotation 部分需要成对角度与角速度字段；现有 sortie 中虽然能找到 pitch/roll/yaw 角度，但缺少对应 rate 字段。因此中期阶段把 rotation 作为字段诊断保留，不用插值或伪造角速度来强行启用。

不要回答：

> rotation 已经完整实现并验证。

后续计划：

- 若后续发现可用角速度字段，在 rotation audit 基础上复跑 `minimal / full / rigid_body`。
- 若仍缺失，继续保持 disabled diagnostics，并在论文里说明数据字段限制。

## 6. Weak-label 风险：没有人工真值

### 当前事实

当前 thesis task 是：

- risk_weak_label
- workload_weak_label
- 事件回放标签

来源：

- risk_weak_label：vehicle intensity + physiology variation。
- workload_weak_label：physiology variation + vehicle intensity。
- event_replay_tag：derived event tag group pairing。

### 风险

老师可能问：这些标签是不是人工标注？能不能代表真实风险/负荷？

### 回答口径

可以回答：

> 当前阶段使用 weak-label 任务验证模型链路、对齐机制、融合机制和 runtime 输出，不把 weak-label 等价为人工真值。中期报告会把它写成 thesis weak-label evidence。后续如果要提高结论强度，需要引入专家复盘、人工标注或外部任务标签。

不要回答：

> 已完成人工标注风险/负荷/事件复盘。

后续计划：

- 中期后可设计小规模专家复盘表，先覆盖少量高风险窗口和关键事件片段。
- 优先把 weak-label 与人工复核样本做一致性分析，而不是立刻扩大弱标签规模。

## 7. Dingxin weak-label 风险：分类任务、回归任务和检索任务 的论文定位

### 当前事实

`chronaris_opt` 在 Dingxin weak-label benchmark 中表现强：

- 分类任务 macro_f1=`1.0`
- 回归任务 rmse=`201.4895651832178`
- 检索任务 top1_accuracy=`1.0`

但 分类任务、回归任务和检索任务 被明确标记为 dingxin_weak_label，不是 thesis weak-label task。

### 风险

老师可能问：这些漂亮指标能不能直接作为论文主要任务结果？

### 回答口径

可以回答：

> 分类任务、回归任务和检索任务 是鼎新真实数据弱监督任务基准，用于诊断 chronaris_opt 的组件价值和模块消融，不能直接替代论文主线的 risk/workload/event weak-label 任务。论文主线仍以 Stage H 鼎新真实双流样本上的 weak-label task、物理约束、语义融合和 runtime contract 为核心。

不要回答：

> 分类任务、回归任务和检索任务 就是论文最终任务。

后续计划：

- 把 分类任务、回归任务和检索任务 放在“辅助实验/消融分析/弱监督任务基准”章节。
- 正文主线避免用 Dingxin weak-label 指标单独支撑最终结论。

## 8. Public adapter 风险：公开数据不是鼎新真实双流

### 当前事实

公开支撑线包括：

- UAB workload dataset。
- NASA CSM。
- public adapter baseline。
- calibration baseline。
- public transfer boundary。

这些公开数据中的第二模态是 task/scenario context-derived second stream，不是 Chronaris 鼎新真实航电流。

### 风险

老师可能问：公开数据结果能不能证明方法泛化？

### 回答口径

可以回答：

> 公开数据用于验证 adapter 与 calibration 思路，并作为公开支撑线；它们的第二模态不是鼎新真实航电流，所以不能直接证明论文鼎新真实双流连续对齐主线 fully closed。我们在 transfer boundary 文档中已经把 dingxin_stage_h、UAB、NASA 的 modality_pair、labels 和 evidence_role 分开。

不要回答：

> UAB/NASA 证明鼎新真实航空双流方法已经公开泛化成功。

后续计划：

- 文献综述中用公开数据说明相关任务与 baseline。
- 论文实验中明确 public adapter 与 Dingxin mainline 的边界。

## 9. 语义事件融合风险：不是专家标注事件

### 当前事实

semantic support 覆盖：

- 3 个双流 view。
- 3 个 query：risk_weak_label、workload_weak_label、event_replay_tag。
- view-level ranking。
- 样本级 top event attribution。

### 风险

老师可能问：语义事件是不是专家定义的飞行事件？

### 回答口径

可以回答：

> 当前语义事件融合用于把模型输出与 weak-label query、事件 token、attention attribution 连接起来，形成可解释 support。它还不是专家复盘事件库或人工语义标签。

不要回答：

> 已完成专家语义事件标签体系。

后续计划：

- 中期后可选取若干代表窗口，结合飞行事件复盘材料做人工解释验证。
- 可把 current semantic support 作为专家标注前的候选片段筛选器。

## 10. 数据规模风险

### 当前事实

当前主要鼎新真实双流样本：

- sortie_count=`2`
- view_count=`3`
- sample_count=`111`

### 风险

老师可能问：规模是否足够？

### 回答口径

可以回答：

> 当前中期阶段重点是证明从数据接入、样本组织、双流建模、物理/因果/语义融合到 runtime 的闭环可运行，并形成可追溯证据。样本规模仍是后续扩展重点，不把当前结果写成大规模泛化结论。

不要回答：

> 当前结果已经代表全量数据分布。

后续计划：

- 中期后优先扩展更多 sortie/view。
- 在扩展前保持当前 evidence runner 与 schema contract，避免新增样本破坏可复现性。

## 11. DeepSeek 时序预处理/LLM 预处理对比 风险：DeepSeek 在线 LLM 预处理和对比实验不是人工真值

### 当前事实

- 已形成文档计划：`docs/implementation/notes/stage-i-deepseek-llm-preprocessing-plan-2026-06-14.md`。
- 已实现 DeepSeek v4-pro 在线 preprocessing pipeline，不默认使用 OpenAI。
- 已完成真实小样本切片 run：`docs/artifacts/assets/stage_i_llm_preprocessing/20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/llm_preprocessing_summary.json`。
- 当前 `request_count=8`、`error_count=0`、`field_semantic_count=24`、`weak_label_review_count=3`、`semantic_query_hint_count=4`、`runtime_explanation_count=4`。
- 当前 `prompt_version=stage_i_llm_preprocessing.agent_guardrails.v2`，本地 harness `final_invalid_task_count=0`，切片任务包括 `field_semantics / schema_gap_policy / runtime_explanations`。
- 输入来自现有 MySQL / InfluxDB 派生链路：
  - MySQL 字段 label、measurement metadata、sortie / view 元信息。
  - InfluxDB 派生的 Stage H 窗口统计摘要。
  - 当前 2 个 sortie、3 个双流 view、111 个窗口样本的 weak-label 和 runtime 证据。
- 当前没有发送原始全量高频时序，只发送 schema card、window summary card 和 runtime case card；较大的字段、schema gap、runtime case 会先切片，再在本地按 stable identifier 合并。
- LLM 预处理对比 已完成 A0-A4 本地对比实验：
  - A1 `333/333` 条 Stage I weak-label task entries attach DeepSeek 时序预处理 context，`label_changed_count=0`、`label_unchanged=true`。
  - A2 semantic query coverage 从 `3` 扩展到 `7`，新增 `4` 条 hints 均通过 recipe whitelist；没有从现有 summary 伪造 view ranking/top attribution 重算。
  - A3 `12` 条 runtime cases 中 `4` 条有 LLM explanation，解释子集四项完整性达到 `1.0`。
  - A4 生成 `15` 条 human review packet item，`human_review_completed=false`。
  - 工程入口：`docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json`。
  - 中期入口：`docs/midterm/llm-preprocessing-comparison-summary-2026-06-14.md`。

### 风险

老师可能问：LLM 是否已经接入？是否把原始数据发给了外部 API？能否替代人工标签？LLM 预处理对比 是否证明 LLM 对融合结果有因果贡献？

### 回答口径

可以回答：

> 当前已实现 DeepSeek 在线大模型辅助时序数据预处理模块，并在现有 Stage H / Stage I 证据链上完成小样本真实切片调用与 LLM 预处理对比 A0-A4 对比实验。请求、响应、prompt version、input hash、harness verdict、slicing summary、latency、retry、error summary、condition manifest 和 comparison CSV 已落盘审计。默认 payload 是 schema card、window summary card 和 runtime case card，没有直接发送原始全量高频时序；较大 payload 先分片，再由本地 harness 按 stable identifier 合并。LLM 输出只作为 preprocessing context、semantic hints、rule review、runtime explanation 和人工复核 packet，不作为人工真值或核心因果证据。

不要回答：

> DeepSeek 已经替代人工标注，已经解决标签可信度问题，或者 LLM 预处理对比 已经证明 LLM semantic hints 改善了核心因果融合结果。

后续计划：

- 中期后可扩展更多 schema/window/runtime cards，但继续使用切片调用与本地合并。
- 人工填写 `human_review_packet.csv` 后，再统计可采纳、需复核和冲突项。
- 若要证明 LLM semantic hints 改变 view ranking 或 top attribution，需要基于 Stage H tensor 重新运行带 LLM query specs 的 support。
- 继续把输出限制为字段语义归一、weak-label 复核、schema gap policy、runtime 解释和复核材料，不替代人工真值。

## 12. 中期报告中的推荐风险章节写法

可在“存在问题与下一步计划”中写：

> 当前工作已经完成人机异构时序数据从接入、标准化、双流建模、物理约束、因果融合、语义支撑到运行时验证的闭环，但仍存在五类边界。第一，风险、负荷和事件复盘任务目前采用 weak-label 构造，尚未引入专家人工标注；第二，公开 UAB/NASA 数据仅用于 adapter 和 calibration 支撑，不能等同于鼎新真实航空双流主线的泛化证明；第三，runtime 原生输入目前为 aligned schema，service contract 可通过 canonical payload 达到 exact，后续需补齐 native replay payload 的 vehicle measurement groups；第四，刚体旋转项受角速度字段缺失限制，当前保留为 diagnostics；第五，DeepSeek 在线 LLM 预处理已完成小样本真实切片接入和 A0-A4 对比实验，但输出仍是字段语义、规则复核、whitelisted semantic hints、schema gap policy、结果解释和人工复核 packet，不是人工真值或因果证明。后续将围绕人工复核样本、更多 sortie/view、native exact schema、可用角速度字段和带 LLM query specs 的 semantic support 复跑继续推进。

## 13. 答辩问答速记

| 问题 | 短答 |
| --- | --- |
| 你们有没有人工真值？ | 目前没有完整人工真值，当前是 weak-label thesis task；人工/专家复核是后续计划。 |
| live_influx 大网格有没有全跑完？ | 没有伪造成全跑完；两个真实 child run 形成 stable resume，更大尝试保留 partial blocker。 |
| runtime 是否 exact？ | native input 是 aligned；canonical service payload 是 exact。两者分开写。 |
| rotation 为什么 disabled？ | 缺少 pitch/roll/yaw rate 成对字段；translation + vertical 已启用。 |
| public 数据能证明什么？ | 证明 adapter/calibration 支撑线，不能证明鼎新真实双流主线 fully closed。 |
| Dingxin weak-label 指标能当论文主结果吗？ | 不能直接当主结果；可作为模块消融和弱监督任务基准。 |
| 样本量是否足够？ | 足够支撑中期闭环展示，不足以支撑大规模泛化结论。 |
| LLM 有没有已经接入？ | 已接入 DeepSeek v4-pro，完成小样本真实切片 run 和 LLM 预处理对比 A0-A4 对比；默认不走 OpenAI；输出是 preprocessing context、解释层和待复核材料，不是人工真值。 |
