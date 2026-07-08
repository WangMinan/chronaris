# Next Prompt: Chronaris Controlled Optimization After E3 Review

你是 GPT 5.5 xhigh，运行在远程开发机上。请在 chronaris 仓库 `implement/fusion-stream-structure-20260707` 分支上执行下一轮 Chronaris 受控候选优化。先阅读 `docs/artifacts/runs/2026-07-08_e3-result-review/` 下所有文件，特别是 `paper_use_decision.md`、`e3_method_metric_summary.csv`、`t1_t2_e3_consistency_table.csv` 和 `risk_and_boundary.md`。

## 目标

在不修改已确认指标、不覆盖历史 run、不重训 MulT / ContiFormer、不改 E3 evaluator 参数的前提下，探索 Chronaris candidate，使 T1、T2、E3 三类证据形成 Pareto 改善或至少暴露可复盘的失败原因。

## 固定 Baselines

- 不重新训练 MulT / ContiFormer。
- 不修改 `docs/artifacts/runs/2026-07-07_deep-baseline-representation-export/`。
- 不修改 `docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/`。
- 不修改 E3 evaluator 参数：m-grid、tol、PCA/预处理、composite 权重全部固定。
- 不回写 `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/`、`result_matrix_long.csv`、`experiment_registry.csv`、`claim_boundary_table.csv`。

## 固定 Evaluation

- 新 Chronaris candidates 只能写入新的 `docs/artifacts/runs/YYYY-MM-DD_chronaris-controlled-optimization-*` run root。
- T1/T2 confirmed metrics 只读不改。
- E3 validation 不覆盖旧 run；candidate E3 必须写新 run root。
- 最终只允许一次 locked confirmation，不能把 dev sweep 最佳值直接写成论文结论。

## Candidate Optimization Targets

同时看：

- T1：macro-F1 / balanced accuracy 不得明显退化。
- T2：RMSE / MAE / NRMSE 不得明显退化，优先修复当前 Chronaris 弱于 MulT / ContiFormer 的回归差距。
- E3：至少改善 `cross_view_segment_stability`、`event_alignment_score`、`discord_maneuver_overlap`、`motif_event_consistency` 或可用的 fragment replay 指标之一；不能只刷 composite。
- 使用 Pareto 筛选，不使用单一总分刷榜。

## Allowed Chronaris Candidate Directions

- `fusion_output_mode`。
- causal lag window。
- vehicle contribution gate。
- physics residual weight。
- temporal smoothness / representation stability regularization。
- contrastive / retrieval auxiliary loss weight。
- dropout / weight decay。
- pooling strategy。
- representation normalization。
- OOF Chronaris representation export 口径，使其与 MulT / ContiFormer 的 T2-trained OOF embedding 更可比。

## Prohibited

- 不改 label。
- 不改 split。
- 不改 E3 evaluator。
- 不删除不利指标或失败 candidate。
- 不用 test fold 标签训练 test 表示。
- 不把 dev sweep 最佳结果直接写成论文结果。
- 不覆盖历史 artifact。
- 不回写 confirmed metrics。

## Required Outputs

新 run root 下必须产出：

- `candidate_registry.csv`
- `candidate_t1_t2_metrics.csv`
- `candidate_e3_metrics.csv`
- `pareto_selection.md`
- `locked_confirmation_run.md`
- `failure_cases.md`
- `paper_boundary.md`
- `evidence_manifest.json`

## Completion Gate

运行相关测试、`compileall`、`git diff --check`，并显式确认：training scope、confirmed metrics unchanged、thesis protocol snapshot unchanged、old E3 run unchanged、old deep baseline embedding unchanged。
