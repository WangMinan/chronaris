# Chronaris v2 锁定后公开数据适配确认

本次确认只运行一项由 v2 锁定配置映射得到的公开数据适配配置，没有使用公开标签选择模型。NASA 与 UAB 结果均汇总 seeds 17、29、43 的完整留一受试者折，第二输入流仅作上下文构造，不等同于鼎新航电流。

## 三随机种子结果

- NASA 综合认知状态分类 Macro-F1：`0.5664 ± 0.0085`；公开参考为 `0.4550`。
- UAB N-back 工作负荷回归 RMSE：`5.0962 ± 0.0578`；公开参考为 `4.6103`。
- UAB 椅背加热任务回归 RMSE：`1.4556 ± 0.0202`；公开参考为 `1.4331`。

NASA 公开适配形成明确改善；UAB N-back 仍落后于公开参考与 ContiFormer，椅背加热任务与公开参考及 ContiFormer 接近。该结果只作为锁定后的公开数据适配证据，不改变 Chronaris v2 未晋级和论文主模型保持 v1 的结论。

## 图件

- 锁定配置说明：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_refresh_screen_leaderboard.png`
- 公开适配与既有方法比较：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_refresh_confirm_vs_baselines.png`
- 方向归一变化热图：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_refresh_delta_heatmap.png`
- 锁定配置参数：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_config_sensitivity.png`
- 训练损失曲线：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_training_curves_best.png`
- NASA 综合分类混淆矩阵：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_best_confusion_nasa_combined.png`
- 相对既有方法的优平劣汇总：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fig_public_fusion_win_summary.png`

## 可恢复入口

- 配置：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fusion_refresh_config.json`
- 折级指标：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/fold_metrics.csv`
- 训练曲线：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/training_curves.csv`
- 证据清单：`/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-13_chronaris-v2-public-adapter-confirmation/evidence_manifest.json`
