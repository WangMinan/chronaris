# Stage I 论文主线重构详细编码计划

更新时间：2026-05-15

## 1. 计划原则

- 不新建顶层 `src/chronaris/stage_i/` 包，继续沿用 `dataset / features / models / pipelines / serving / evaluation` 现有布局。
- 不覆写 `Stage E/F/G/H` 历史 closure，只在其上增加更强主线。
- 所有新增路径都要补测试、报告 contract 和文档回写。

当前状态补记：

- `Phase A / Phase B / Phase C` 首轮代码与测试收敛已完成。
- 本文当前主要保留“文件级落点索引 + 后续未完成项”；并不表示 `Phase C` 仍未实现。

## 2. Phase A：主线边界校准

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/features/stage_i_sequences.py` | 把公开第二模态的论文口径统一为 `context_proxy`；保留 `task_context / scenario_context` 作为数据集内字段名，但在 metadata 中显式声明 `adapter/proxy` 语义。 | 导出的 `dataset_summary / sequence_schema / metadata` 不再把公开第二流写成真实航电流。 |
| `src/chronaris/pipelines/stage_i/common/deep_models.py` | 保留 `Stage G` 核心接口不动，但在 `ChronarisPublicFusionWrapper` 层补明确的 second-stream 语义边界，避免 thesis-facing 代码把它误称为 vehicle stream。 | public deep wrapper 的输入/输出说明与 `stage_i_sequences.py` 一致。 |
| `src/chronaris/pipelines/stage_i/public/mainline_report.py` | 把 `public opt closed` 重新表述为 `public adapter evidence`，显式区分 `public adapter`、`public fusion exploratory`、`private mainline`。 | 主报告不再把 `public opt` 直接写成 thesis dual-stream 主线闭环。 |
| `src/chronaris/pipelines/stage_i/public/opt_reporting.py` | 在 `UAB robust-prior`、`NASA round1` 等 report summary 中保留 adapter/calibration 边界。 | report wording 与 roadmap/gap 文档一致。 |
| `src/chronaris/pipelines/stage_i/private/benchmark_data.py` | 给 `T1/T2/T3` 增加 `proxy_task` 语义字段或独立 builder 标识。 | 任务 metadata 能区分 `proxy` 与后续 `thesis task`。 |
| `tests/test_stage_i_public_opt.py` | 新增 adapter/proxy metadata 与 report wording 合约测试。 | synthetic asset 下可校验新 metadata 键和值。 |
| `tests/test_stage_i_deep_pipeline.py` | 校验公开双流 wrapper 的第二模态语义不会回退成 thesis-facing vehicle stream 描述。 | public deep path contract 通过。 |

## 3. Phase B：统一骨干与 checkpoint 导出

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/pipelines/alignment_preview.py` | 抽出 shared backbone train/infer helper，避免 `Stage H export` 直接绑死 preview training。 | 训练与推理共享编码/导出逻辑，不再只能走 preview pipeline。 |
| `src/chronaris/pipelines/stage_i/training/backbone_train.py` | 新增统一 backbone 训练入口，负责 `E/F/G/H` 骨干训练、checkpoint、训练摘要。 | 可以从固定 config 训练并落盘可复用 checkpoint。 |
| `src/chronaris/pipelines/stage_h/export.py` | 新增 `checkpoint_path`、`inference_only`、`backbone_run_id` 等参数；保留 preview/research 路径但降为 secondary。 | 同一 checkpoint 可用于多 view 导出，不再 per-view 训练。 |
| `src/chronaris/features/stage_h_bundle.py` | 在 run manifest / bundle metadata 中补 `export_mode`、`checkpoint_path`、`backbone_run_id`、`train_config_digest`。 | 下游能识别 bundle 来自 preview 还是 frozen inference。 |
| `scripts/run_stage_h_export.py` | CLI 增加 frozen inference 入口和参数校验。 | CLI 可以切换 `preview_train` 与 `checkpoint_inference`。 |
| `scripts/stage_i/training/train_backbone.py` | 新增 backbone train CLI。 | 单命令可训练骨干并产出 checkpoint。 |
| `tests/test_stage_h_export.py` | 增加 checkpoint inference contract 测试。 | fake checkpoint / synthetic manifest 下通过。 |
| `tests/test_alignment_pipeline.py` | 回归训练/推理 shared path。 | 原 preview 路径不回归，新增 infer path 可通过。 |

## 4. Phase C：联合训练与 thesis task builder（首轮已完成）

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/models/alignment/task_heads.py` | 新增多任务头定义，覆盖 classification / regression / retrieval 与后续 thesis-task heads。 | 单文件集中管理 task heads 和 output contract。 |
| `src/chronaris/models/alignment/losses.py` | 在现有 objective 上增加 `task_loss` 汇总入口和权重配置。 | `StageEObjectiveBreakdown` 或并行 multitask breakdown 能输出 `L_task`。 |
| `src/chronaris/dataset/stage_i_real_task_builders.py` | 新增 thesis-task builder，先实现 `risk_proxy`、`workload_proxy`、`event_replay_tag`。 | 可以从现有 `Stage H / private` 资产构建真实任务近似标签。 |
| `src/chronaris/pipelines/stage_i/training/multitask_train.py` | 新增统一训练入口，打通 `L_recon + L_align + L_phy + L_causal + L_task`。 | 至少一组私有弱标签 thesis task 能端到端训练并导出结果。 |
| `src/chronaris/pipelines/stage_i/private/benchmark_data.py` | 把 `T1/T2/T3` 的构造逻辑下沉到 `proxy task builder`，与真实 thesis-task builder 拆开。 | `proxy` 与 `thesis task` contract 不再混在一个入口。 |
| `src/chronaris/pipelines/stage_i/private/benchmark.py` | 支持 benchmark 结果按 `proxy / thesis` 两层输出。 | 报告可单独说明代理任务和 thesis-task 证据。 |
| `src/chronaris/pipelines/stage_i/private/benchmark_models.py` | 允许共享 backbone + task head 的训练/评估路径，不再只消费导出特征表。 | deep/classical 路线可以和 multitask 路线并列比较。 |
| `tests/test_stage_i_private_optimization.py` | 保留旧 benchmark 回归，同时新增 `proxy/thesis task split` 合约测试。 | 历史 benchmark 不回归，新任务入口可验证。 |
| `tests/test_stage_i_multitask_train.py` | 新增 multitask smoke test。 | synthetic 数据上能完成一次前向、loss 汇总与落盘。 |

## 5. Phase D：Stage F 物理约束补强

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/models/alignment/physics_state_mapping.py` | 新增显式状态映射，把字段归到 `speed / acceleration / altitude / vertical_speed / attitude / angular_rate` 等状态向量。 | 字段映射失败与缺失状态都能明示。 |
| `src/chronaris/models/alignment/physics_residuals.py` | 新增显式残差计算函数，先覆盖当前字段能稳定支持的运动学子系统。 | 可以单测每类残差，不再把全部逻辑塞在 `physics.py`。 |
| `src/chronaris/models/alignment/physics.py` | 把 `minimal/full` 重构成 `minimal/weak_physics/rigid_body` 家族注册；旧 `full` 作为 `weak_physics` 保留兼容层。 | 旧配置不回归，新 `rigid_body` 可被选择。 |
| `src/chronaris/models/alignment/physics_features.py` | 补足状态映射辅助信息和字段覆盖诊断。 | report 可以说明当前 sortie 哪些残差被启用。 |
| `src/chronaris/models/alignment/losses.py` | 接入新的 physics family 与 breakdown 输出。 | `physics_components` 能分辨 weak vs rigid-body 残差。 |
| `tests/test_alignment_model_losses.py` | 新增 `rigid_body_family` 及其 fallback 测试。 | 不同 family 的数值路径都可回归。 |

## 6. Phase E：Stage G 语义事件融合补强

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/models/fusion/causal.py` | 保留最小因果注意力路径，同时把 richer event fusion 的公共张量接口稳定下来。 | 原 `G(min)` 不回归，新路径可复用相同输入 contract。 |
| `src/chronaris/models/fusion/semantic_event.py` | 新增 `SemanticQueryBank`、`EventTokenExtractor`、`CausalEventFusion`。 | 可输出 event token、query-to-event attention 与归因摘要。 |
| `src/chronaris/pipelines/stage_i/evidence/support_builders.py` | 支持导出语义事件级 support 工件。 | support 资产里出现 event token 和 query attribution。 |
| `src/chronaris/pipelines/stage_i/evidence/support.py` | 报告中增加语义事件级对照，而不是只写 attention heatmap。 | support 报告能区分时间步 attention 与事件级归因。 |
| `src/chronaris/pipelines/stage_i/evidence/anchors.py` | 关键工况导出补事件级解释。 | anchor 报告可以展示“哪个事件原型触发了关注”。 |
| `tests/test_stage_i_support.py` | 新增 event-level support contract 测试。 | support builder/reporting 不回归。 |
| `tests/test_stage_i_deep_pipeline.py` | 增加 semantic-event path 的最小 smoke test。 | 新 fusion path 可以前向并导出关键张量。 |

## 7. Phase F：runtime inference

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `src/chronaris/dataset/streaming_windows.py` | 新增流式时间基准与滑窗缓存工具。 | 可以按增量点流产出统一窗口。 |
| `src/chronaris/serving/runtime_inference.py` | 新增 checkpoint 推理主入口，负责接入、缓存、推理、解释输出。 | CLI replay 或 mock stream 可连续输出预测。 |
| `src/chronaris/serving/runtime_demo.py` | 保留离线报告角色，并显式说明 `not inference engine`。 | demo 与 inference 职责边界清晰。 |
| `scripts/stage_i/runtime/run_inference.py` | 新增 runtime inference CLI。 | 可对本地回放流执行增量推理。 |
| `scripts/stage_i/runtime/run_demo.py` | 更新帮助信息，强调其离线展示定位。 | CLI 文案与代码职责一致。 |
| `tests/test_runtime_inference.py` | 新增流式缓存与 checkpoint 推理 smoke test。 | mock stream 下能跑完整个增量闭环。 |

## 8. 文档回写文件

| 文件 | 动作 | 验收 |
| --- | --- | --- |
| `docs/implementation/notes/coding-roadmap.md` | 只保留阶段状态与当前主线，不再把旧 `public opt` 扩搜当下一步。 | 顶层阶段状态与现行工作包一致。 |
| `docs/implementation/notes/thesis-coding-gap.md` | 把“只剩图表整理”的旧判断改成“七类实现缺口 + 新优先级”。 | gap 文档与源码现状一致。 |
| `docs/artifacts/stage_i/README.md` | 区分 `historical closure / public adapter evidence / private proxy evidence / thesis mainline roadmap`。 | 报告索引不再混淆当前主线。 |
| `docs/README.md` | 更新 `planning` 目录索引和当前事实源。 | 顶层文档入口不再指向旧计划。 |
| `AGENTS.md` | 更新当前优先引用的事实源与 Stage I 默认工作方式。 | 协作约束与当前规划同步。 |

## 9. 推荐执行顺序

1. `Phase A/B/C` 当前仅做回归维护，不再重复扩旧 benchmark wording。
2. 优先推进 `Phase D` 的 physics 补强。
3. 然后推进 `Phase E` 的 semantic event 方法体。
4. 最后完成 `Phase F` 的 runtime inference。

## 10. 每阶段最小验证

- `Phase A`：`tests.test_stage_i_public_opt`、`tests.test_stage_i_deep_pipeline`
- `Phase B`：`tests.test_stage_h_export`、`tests.test_alignment_pipeline`
- `Phase C`：`tests.test_stage_i_private_optimization`、`tests.test_stage_i_multitask_train`
- `Phase D`：`tests.test_alignment_model_losses`
- `Phase E`：`tests.test_stage_i_support`、`tests.test_stage_i_deep_pipeline`
- `Phase F`：`tests.test_runtime_inference`
