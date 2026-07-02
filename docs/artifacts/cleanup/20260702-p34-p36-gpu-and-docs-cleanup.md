# 2026-07-02 P34-P36 GPU profiling and docs cleanup

本记录只覆盖本轮 P34/P35/P36 后处理，不改变 P34/P35/P36 已确认指标，也不覆盖 P30/P31/P32 reference metrics。

## GPU profiling

- profiling root：`docs/artifacts/assets/stage_i_gpu_parallel_profile/20260702T-stage-i-gpu-parallel-profile-r1/nasa_csm_v3_like_public_fusion/`
- protocol：NASA public context-proxy prepared data，`chronaris_public_fusion`，CUDA，1 epoch / 1 fold，`tensor_cache=auto`，`amp=bf16`，`checkpoint_policy=off`。
- result：`runtime_device=cuda`，CUDA tensor cache 生效；单折 cache 约 `0.115 GB`。
- auto batch selection：`benchmark_only` 与 `loft_only` 选 `2048`，`combined` 选 `1024`。
- probe memory pressure：最大约 `0.209`，低于 `0.88` limit；说明当前小/中 batch 仍偏保守，后续长跑可继续使用 `2048/1024/512/...` candidates，并优先减少 checkpoint / dense prediction IO。

## Code changes for future runs

- 新增 `checkpoint_policy`：`off` / `last` / `epoch_and_fold`。
- public deep baseline、P31 public fusion ablation、P30 private third-party comparison、P34 task-head wrapper、P35 stream-role wrapper 和对应 CLI 都支持该参数。
- 默认策略为 `last`：保留最近恢复点，但不再默认写每 fold checkpoint 矩阵；profiling 或临时 screen 可以用 `off`。
- `.gitignore` 新增 docs artifact 下 checkpoint / model binary / dense prediction / partial CSV 规则。

## Figure refresh

已基于现有 CSV/JSON 重绘以下柱状图，柱顶/柱端增加短数值标签：

- P34：`stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/`
- P35：`stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/`
- P35 nested public/private confirm aggregate figures。
- P36：`stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/`
- optimized model summary：`stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/`

## Docs cleanup

- 删除本轮过渡资产：
  - P34 `r1`、`r2-confirm-screen`
  - P35 `r1`、`r2-confirm-screen`、`r3-confirm20`
  - P36 `r1`、`r2-confirm-screen`、`r3-confirm20`
  - optimized summary `r1`、`r2-confirm-screen`、`r3-confirm20`
- 删除已由 P31 GPUOPT r1 接管且不再被当前文档入口引用的 `stage_i_public_fusion_ablation/20260701T-stage-i-public-fusion-ablation-r1/`。
- P35 public nested confirm 删除 per-candidate `confirm/` 子目录、dense predictions、partial CSV 和 checkpoint；保留 aggregate metrics、figures、summary、manifest、progress、trimmed log。
- 压缩训练曲线：
  - P35 top-level `training_curves.csv`：约 `76.1 MB -> 7.3 MB`
  - P35 public nested `training_curves.csv`：约 `74.5 MB -> 6.6 MB`
  - P31 GPUOPT `training_curves.csv`：约 `27.9 MB -> 4.4 MB`
- 裁剪大日志：
  - P35 public nested `run.log`：约 `35.8 MB -> 0.34 MB`
  - P31 GPUOPT `run.log`：约 `21.0 MB -> 0.33 MB`
- dense prediction CSV 已从 git 追踪中移除，保留 summaries / metrics / figures；现存副本备份到远程开发机仓库外路径：`/home/wangminan/projects/chronaris-local-artifacts/dense-predictions-pruned-20260702/`。
- 重复的 weak-label `thesis_task_manifest.jsonl` 副本哈希一致；git 中只保留 canonical：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`。
- 当前 git 历史内 15 个 `.pt` checkpoint 已备份到远程开发机仓库外路径：`/home/wangminan/projects/chronaris-local-artifacts/checkpoints-history-20260702/`；提交后使用 `git filter-repo` 从 git 历史和 HEAD 移除 checkpoint binary。
