# Chronaris 中期边界与风险说明

更新时间：2026-06-13

本说明用于中期报告和答辩问答。目标是把当前工作讲清楚、讲稳，不把弱证据包装成强结论，也不因为边界存在就低估已经完成的工程和实验闭环。

## 1. 总边界

当前 Chronaris 中期阶段的强结论是：

- 已经完成从私有 MySQL / InfluxDB 数据到 Stage H 双流 view、Stage I weak-label 任务、物理/因果/语义/runtime 支撑证据的闭环。
- 已经形成 `docs/artifacts/` 下可追溯的报告、JSON、CSV、PNG 和 checkpoint。
- 已经把关键风险从口头说明转成 `partial_summary.json`、`runtime_schema_contract.json`、`runtime_error_cases.json` 和图表说明。

当前不能扩写成：

- 人工真值任务完全闭环。
- 全量飞行数据或跨机型泛化验证完成。
- 原始上游 runtime schema 已完全 exact。
- 完整旋转刚体约束已启用。
- public adapter 等价于论文私有双流主线的直接胜利。

## 2. 证据层级边界

| evidence layer | 当前用途 | 可以怎么写 | 不能怎么写 |
| --- | --- | --- | --- |
| thesis_weak_label | `risk_proxy / workload_proxy / event_replay_tag` 论文任务原型 | “基于窗口统计构造 weak-label 任务，用于验证端到端建模和消融链路” | “人工标注风险/负荷/事件真值验证完成” |
| private_proxy | `chronaris_opt`、T1/T2/T3 私有代理 benchmark | “用于诊断模块组合、因果掩码、时间残差、任务头的贡献” | “论文最终任务已由 T1/T2/T3 直接证明” |
| public_adapter/calibration | UAB/NASA 公开数据支撑线 | “用于公开数据适配、校准和 transfer boundary 说明” | “公开数据证明私有航空双流主线 fully closed” |
| rigid_body_support | 物理一致性约束 | “translation + vertical 已启用，rotation 受字段限制保留 diagnostics” | “完整 6DoF 刚体旋转约束已经实现并验证” |
| semantic_support | 语义事件融合 support | “在 3 个双流 view 上形成 view-level ranking 和 attribution” | “专家语义事件复盘标注验证完成” |
| runtime_replay/service_contract | runtime replay、服务化 smoke、schema contract | “native aligned，canonical exact；错误样例与 contract 已固化” | “原始上游输入已 native exact schema” |

## 3. P11 风险：live_influx sweep 的完成度

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

## 4. P17/P18 风险：runtime schema aligned 而不是 native exact

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

## 5. P15 风险：rotation disabled

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

- `risk_proxy`
- `workload_proxy`
- `event_replay_tag`

来源：

- risk_proxy：vehicle intensity + physiology variation。
- workload_proxy：physiology variation + vehicle intensity。
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

## 7. Private proxy 风险：T1/T2/T3 的论文定位

### 当前事实

`chronaris_opt` 在 private proxy benchmark 中表现强：

- T1 macro_f1=`1.0`
- T2 rmse=`201.4895651832178`
- T3 top1_accuracy=`1.0`

但 T1/T2/T3 被明确标记为 `private_proxy`，不是 thesis weak-label task。

### 风险

老师可能问：这些漂亮指标能不能直接作为论文主要任务结果？

### 回答口径

可以回答：

> T1/T2/T3 是私有代理 benchmark，用于诊断 chronaris_opt 的组件价值和模块消融，不能直接替代论文主线的 risk/workload/event weak-label 任务。论文主线仍以 Stage H 私有双流样本上的 weak-label task、物理约束、语义融合和 runtime contract 为核心。

不要回答：

> T1/T2/T3 就是论文最终任务。

后续计划：

- 把 T1/T2/T3 放在“辅助实验/消融分析/代理 benchmark”章节。
- 正文主线避免用 private proxy 指标单独支撑最终结论。

## 8. Public adapter 风险：公开数据不是私有双流

### 当前事实

公开支撑线包括：

- UAB workload dataset。
- NASA CSM。
- public adapter baseline。
- calibration baseline。
- public transfer boundary。

这些公开数据中的第二模态是 task/scenario context proxy，不是 Chronaris 私有航电流。

### 风险

老师可能问：公开数据结果能不能证明方法泛化？

### 回答口径

可以回答：

> 公开数据用于验证 adapter 与 calibration 思路，并作为公开支撑线；它们的第二模态不是私有真实航电流，所以不能直接证明论文私有双流连续对齐主线 fully closed。我们在 transfer boundary 文档中已经把 private_stage_h、UAB、NASA 的 modality_pair、labels 和 evidence_role 分开。

不要回答：

> UAB/NASA 证明私有航空双流方法已经公开泛化成功。

后续计划：

- 文献综述中用公开数据说明相关任务与 baseline。
- 论文实验中明确 public adapter 与 private mainline 的边界。

## 9. 语义事件融合风险：不是专家标注事件

### 当前事实

semantic support 覆盖：

- 3 个双流 view。
- 3 个 query：risk_proxy、workload_proxy、event_replay_tag。
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

当前主要私有双流样本：

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

## 11. 中期报告中的推荐风险章节写法

可在“存在问题与下一步计划”中写：

> 当前工作已经完成人机异构时序数据从接入、标准化、双流建模、物理约束、因果融合、语义支撑到运行时验证的闭环，但仍存在四类边界。第一，风险、负荷和事件复盘任务目前采用 weak-label 构造，尚未引入专家人工标注；第二，公开 UAB/NASA 数据仅用于 adapter 和 calibration 支撑，不能等同于私有航空双流主线的泛化证明；第三，runtime 原生输入目前为 aligned schema，service contract 可通过 canonical payload 达到 exact，后续需补齐 native replay payload 的 vehicle measurement groups；第四，刚体旋转项受角速度字段缺失限制，当前保留为 diagnostics。后续将围绕人工复核样本、更多 sortie/view、native exact schema 和可用角速度字段继续推进。

## 12. 答辩问答速记

| 问题 | 短答 |
| --- | --- |
| 你们有没有人工真值？ | 目前没有完整人工真值，当前是 weak-label thesis task；人工/专家复核是后续计划。 |
| live_influx 大网格有没有全跑完？ | 没有伪造成全跑完；两个真实 child run 形成 stable resume，更大尝试保留 partial blocker。 |
| runtime 是否 exact？ | native input 是 aligned；canonical service payload 是 exact。两者分开写。 |
| rotation 为什么 disabled？ | 缺少 pitch/roll/yaw rate 成对字段；translation + vertical 已启用。 |
| public 数据能证明什么？ | 证明 adapter/calibration 支撑线，不能证明私有双流主线 fully closed。 |
| private proxy 指标能当论文主结果吗？ | 不能直接当主结果；可作为模块消融和代理 benchmark。 |
| 样本量是否足够？ | 足够支撑中期闭环展示，不足以支撑大规模泛化结论。 |
