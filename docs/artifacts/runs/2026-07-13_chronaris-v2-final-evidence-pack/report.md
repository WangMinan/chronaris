# Chronaris v2 锁定复验与差距证据包

结论：v2 不晋级；论文主模型保持为 `chronaris_v1`。
未通过的预声明主指标：鼎新机动强度分类 Macro-F1, 鼎新高生理响应识别 AUPRC, 鼎新连续生理响应 RMSE, 仿真负荷分类 Macro-F1, 仿真负荷回归 RMSE, 仿真机动分段 Macro-F1。
鼎新结果属于固定数据同协议复验；独立仿真族、公开数据适配和正式消融均在配置锁定后运行，结果不回流当前开发轮次。
六主指标图：`docs/artifacts/runs/2026-07-13_chronaris-v2-final-evidence-pack/figures/six_primary_metrics.png`；压力斜率图：`docs/artifacts/runs/2026-07-13_chronaris-v2-final-evidence-pack/figures/stress_slopes.png`。
时间机制图：`docs/artifacts/runs/2026-07-13_chronaris-v2-final-evidence-pack/figures/timing_mechanism.png`；正式消融图：`docs/artifacts/runs/2026-07-13_chronaris-v2-final-evidence-pack/figures/formal_ablation.png`。
公开数据适配汇总包含 6 个数据集—随机种子条目，第二输入流仅作上下文构造。
