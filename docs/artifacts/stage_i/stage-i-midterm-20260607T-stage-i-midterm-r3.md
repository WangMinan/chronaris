# Stage I Midterm Evidence - 20260607T-stage-i-midterm-r3

## 1. 总判断

- 当前鼎新真实数据主线：`chronaris_opt`，`private_optimality_supported=True`。
- 当前公开主线：`public opt closed`。
- NASA public fusion confirm：`macro-F1=0.3558`，相对 `0.40` 门槛结论为 `negative evidence`。
- UAB fairness confirm：`n_back=4.6541 (hist=4.6541), heat_the_chair=1.4568 (hist=1.4568)`。

## 2. 字体与图件

- 字体策略：`using CJK font WenQuanYi Zen Hei`
- 图件根目录：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots`
- `thesis_chain_status`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/thesis_chain_status.png`
- `private_opt_task_metrics`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/private_opt_task_metrics.png`
- `public_mainline_metrics`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/public_mainline_metrics.png`
- `support_ablation`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/support_ablation_metrics.png`
- `anchor_window_scores`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/anchor_window_scores.png`
- `runtime_progress_timeline`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/plots/runtime_progress_timeline.png`

## 3. 指标摘录

| section | scope | metric_name | variant | value | delta | source_path | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dingxin | 分类任务：机动强度分类 | macro_f1 | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
| Dingxin | 分类任务：机动强度分类 | macro_f1 | chronaris_opt_no_causal_mask | 0.173333 | 0.826667 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
| Dingxin | 回归任务：下一窗口生理响应 | rmse | chronaris_opt | 201.489565 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
| Dingxin | 回归任务：下一窗口生理响应 | rmse | chronaris_opt_no_causal_mask | 313.232477 | 111.742912 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
| Dingxin | 检索任务：配对飞行员窗口检索 | top1_accuracy | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
| Dingxin | 检索任务：配对飞行员窗口检索 | top1_accuracy | chronaris_opt_no_causal_mask | 0.027027 | 0.972973 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
| support | alignment | mean_projection_cosine | e_baseline | 0.760173 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json |  |
| support | alignment | mean_projection_cosine | f_full | 0.699514 | -0.060659 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json | delta_f_minus_e |
| support | causal | mean_top_contribution_score | g_min | 2.609973 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json |  |
| support | causal | delta_mean_top_contribution_score | vehicle_delta_suppressed | -2.399792 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_support/20260506T120000Z-stage-i-support/support_summary.json | strongest ablation |
| public | uab/n_back | rmse | ridge_residual | 4.610314 | 0.043777 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_mainline/20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/public_mainline_summary.json | best_deep=contiformer |
| public | uab/heat_the_chair | rmse | target_prior_median | 1.433140 | 0.023618 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_mainline/20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/public_mainline_summary.json | best_deep=contiformer |

## 4. Cleanup Audit

- outdated_report_candidates：`2`
- unreferenced_code_candidates：`0`
- cleanup_audit_path：`/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/cleanup_audit.json`

## 5. 说明

- 本报告只整编当前仓库已验证事实与本轮 confirm，不改写 Stage E/F/G/H contract。
- UAB `target_prior_median` 仍只写成 `uab_public_adapter` / calibration baseline，不包装成融合模块本体胜利。
- 若后续环境补强了 `fonts-wqy-zenhei` 或 `fonts-noto-cjk`，可直接重跑本入口重导中文标题图件。
