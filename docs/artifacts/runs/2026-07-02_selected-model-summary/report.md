# task evaluation Optimized Model Summary - 20260702T-task-eval-optimized-model-summary-r4-v3-confirm20

- status: `completed`
- runtime_device: `cuda`
- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary`
- boundary: This package summarizes completed optimized 任务感知头优化、流角色融合与优化模型再评估 evidence with fixed 鼎新真实数据第三方模型对比、公开融合消融与跨证据矩阵 references; public rows remain context-derived second-stream evidence and historical artifacts are read-only.

## 任务感知头优化 Key Deltas
- 分类任务：机动强度分类 `balanced_accuracy`: 鼎新真实数据第三方模型对比=`0.333333` 任务感知头优化=`0.34188` delta=`0.00854701` status=`improved`
- 分类任务：机动强度分类 `macro_f1`: 鼎新真实数据第三方模型对比=`0.173333` 任务感知头优化=`0.216065` delta=`0.0427322` status=`improved`
- 分类任务：机动强度分类 `balanced_accuracy`: 鼎新真实数据第三方模型对比=`0.333333` 任务感知头优化=`0.344729` delta=`0.011396` status=`improved`
- 分类任务：机动强度分类 `macro_f1`: 鼎新真实数据第三方模型对比=`0.173333` 任务感知头优化=`0.187489` delta=`0.0141557` status=`improved`
- 回归任务：下一窗口生理响应 `mae`: 鼎新真实数据第三方模型对比=`992.524` 任务感知头优化=`317.41` delta=`675.114` status=`improved`
- 回归任务：下一窗口生理响应 `nrmse`: 鼎新真实数据第三方模型对比=`11.3163` 任务感知头优化=`2.05032` delta=`9.26601` status=`improved`
- 回归任务：下一窗口生理响应 `rmse`: 鼎新真实数据第三方模型对比=`1111.98` 任务感知头优化=`415.361` delta=`696.618` status=`improved`
- 回归任务：下一窗口生理响应 `mae`: 鼎新真实数据第三方模型对比=`762.558` 任务感知头优化=`276.504` delta=`486.054` status=`improved`
- 回归任务：下一窗口生理响应 `nrmse`: 鼎新真实数据第三方模型对比=`8.78099` 任务感知头优化=`2.25945` delta=`6.52154` status=`improved`
- 回归任务：下一窗口生理响应 `rmse`: 鼎新真实数据第三方模型对比=`838.121` 任务感知头优化=`346.827` delta=`491.294` status=`improved`
- 检索任务：配对飞行员窗口检索 `mrr`: 鼎新真实数据第三方模型对比=`0.119948` 任务感知头优化=`0.117939` delta=`-0.00200877` status=`regressed`
- 检索任务：配对飞行员窗口检索 `top1`: 鼎新真实数据第三方模型对比=`0.027027` 任务感知头优化=`0.0315315` delta=`0.0045045` status=`improved`
- 检索任务：配对飞行员窗口检索 `top3`: 鼎新真实数据第三方模型对比=`0.0945946` 任务感知头优化=`0.0855856` delta=`-0.00900901` status=`regressed`
- 检索任务：配对飞行员窗口检索 `top5`: 鼎新真实数据第三方模型对比=`0.148649` 任务感知头优化=`0.13964` delta=`-0.00900901` status=`regressed`

## 流角色融合 Stream-role Gates
- dingxin_feature_export role=`real_vehicle` route=`causal_lagged_vehicle_to_physio` lag=`0.869754` context=`0.101672` vehicle=`0.857364` causal=`0.876729`
- `nasa_csm` role=`scenario_context_proxy` route=`adaptive_context_gate` lag=`0.115751` context=`0.752347` vehicle=`0.155897` causal=`0.121497`
- `uab_workload_dataset` role=`task_context_proxy` route=`adaptive_context_gate` lag=`0.119853` context=`0.755867` vehicle=`0.158099` causal=`0.13321`
- `nasa_csm` role=`scenario_context_proxy` route=`causal_lagged_vehicle_to_physio` lag=`0.865751` context=`0.102347` vehicle=`0.861355` causal=`0.871497`
- `uab_workload_dataset` role=`task_context_proxy` route=`context_adapter_only` lag=`0.0479411` context=`0.855867` vehicle=`0.0451711` causal=`0.0532838`

## GPU Runtime
- `P34` device=`cuda` gpu=`NVIDIA GeForce RTX 4090` cache=`auto` batch=`2048` amp=`bf16` compile=`off` max_mem_gb=`0.0217443` samples_per_sec=`14.144` util_pct=`NA`
- `P35` device=`cuda` gpu=`NVIDIA GeForce RTX 4090` cache=`auto` batch=`NA` amp=`bf16` compile=`off` max_mem_gb=`4.6633` samples_per_sec=`36.1538` util_pct=`1`

## Outputs
- summary: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/optimized_model_summary.csv`
- key metrics: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/key_metric_summary.csv`
- gates: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/stream_role_gate_summary.csv`
- gpu: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/gpu_runtime_summary.csv`
- claim boundary: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/claim_boundary_summary.csv`
- resume commands: `/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-07-02_selected-model-summary/resume_commands.txt`
