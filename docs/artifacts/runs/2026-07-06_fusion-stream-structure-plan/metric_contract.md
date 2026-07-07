# 指标合同 (Metric Contract) — Fusion Stream Structure Evaluation (E3)

更新时间：2026-07-06
状态：规划合同。所有指标在后续编码阶段的 `metrics.py` 实现，并写入 E3 专属 long 表，`evidence_quadrant = fusion_stream_structure`。

## 0. 总则

- E3 指标**单独成表**，不并入既有 `dingxin_model_comparison` / `dingxin_component_ablation` 的 leaderboard，不与 T1/T2/T3 confirmed metrics 混算 delta。
- E3 不产出单一 winner。主表只给结构指标对比 + 案例图。
- 所有 post-hoc 指标（与弱事件边界、机动代理、生理波动代理比对）只用于**评价结构输出的合理性**，不用于训练。
- 指标方向、是否可入论文主表、案例图/附录归属逐条标注。
- 结果不理想的处理见 §5。

## 1. ClaSPy / ClaSP-CLaP 指标

| 指标 | 定义 | 方向 | 论文主表? | 备注 |
| --- | --- | --- | --- | --- |
| `cp_tolerance_hit_rate` | ClaSP change points 在弱事件边界 `±tol`（默认 `tol = max(1, round(0.02T))` 窗口）内的命中率 | 越高越好 | 是（主表） | post-hoc，对照 `weak_event_boundary` |
| `segment_event_purity` | 每个 ClaSP 段内主导机动代理标签的纯度（加权平均） | 越高越好 | 是（主表） | post-hoc，对照 `maneuver_proxy_label` |
| `state_count` | CLaP 检测到的状态数 | 中性（记录） | 否（案例/附录） | 过多/过少都视为结构不稳 |
| `transition_entropy` | 状态转移图的转移熵 | 中性（记录） | 否（附录） | 衡量转移结构复杂度 |
| `cross_view_segment_stability` | 同一 sortie 跨 view 的 change point 集合的归一化一致性（如基于二值边界序列的 F1） | 越高越好 | 是（主表） | 不依赖弱标签，纯结构 |
| `clap_state_replay_consistency` | 同一 sortie 用不同 view 拟合 CLaP 后，状态标签序列的对齐一致性 | 越高越好 | 否（附录） | 纯结构，可复盘性证据 |

输出对象（per `(method, sortie, view)`）：

- `change_points`：change point 索引列表（映射回 `window_id`/`time`）。
- `state_sequence`：长度 T 的状态标签序列。
- `state_transition_graph`：`(states, transitions)`（来自 `AgglomerativeCLaPDetection().predict(sparse=True)`）。

ClaSP/CLaP 选择策略（实现时固化）：

- 主路径：`BinaryClaSPSegmentation()` 取 change points → `AgglomerativeCLaPDetection()` 取 state sequence 与 transition graph。
- 不强制 `n_cps`；若需对照弱标签做有偏评估，另起 `clasp_guided` 列并显式标注“使用了弱标签 hint，仅作对照”，不计入主无监督结论。

## 2. STUMPY / Matrix Profile 指标

| 指标 | 定义 | 方向 | 论文主表? | 备注 |
| --- | --- | --- | --- | --- |
| `motif_event_consistency` | top-k motif pair 落入相同机动代理区间的比例 | 越高越好 | 是（主表） | post-hoc |
| `discord_maneuver_overlap` | top-k discord segment 与高机动区间的时间重合率（IoU） | 越高越好 | 是（主表） | post-hoc |
| `discord_physio_overlap` | top-k discord segment 与高生理波动区间的重合率 | 越高越好 | 否（附录） | post-hoc |
| `nn_segment_cross_view_consistency` | 同 sortie 跨 view 的 nearest-neighbor segment 配对一致性 | 越高越好 | 是（主表） | 纯结构，直接服务于“可复盘” |
| `fluss_clasp_agreement` | FLUSS regime 边界与 ClaSP change point 的 `±tol` 一致率 | 越高越好 | 否（附录） | 两套无监督分割互验 |
| `mp_discord_isolation` | discord 的 matrix profile 距离相对全流分布的 z-score | 越大越异常 | 否（附录） | 异常片段强度 |
| `snippet_coverage` | `stumpy.snippets` 代表片段对全流的覆盖比例 | 越高越好 | 否（附录） | 可摘要性 |

输出对象（per `(method, sortie, view)`）：

- `matrix_profile` + `matrix_profile_indices`（来自 `mstump` 或 `stump`）。
- `motif_pair`：`(idx_a, idx_b, distance)`，映射回 `window_id`/`time` 区间。
- `discord_segment`：`(start_idx, end_idx, distance)`。
- `nearest_neighbor_segment`：`(query_idx, nn_idx, distance)`。
- `fluss_regimes`：regime 边界索引列表。

多维 vs 一维选择（实现时固化）：

- 默认多维：`stumpy.mstump(X.T, m=m)`，`X.T` 形状 `(d, T)`，返回 `(mp, indices)`。
- 当 `d'` 仍较高或需要稳定 motif 时，额外在第一主成分上跑 `stumpy.stump` 作对照，并记录两种结果的一致性。
- 子序列长度 `m` 的候选：优先从 `{window_len, event_duration_windows, maneuver_interval_windows}` 构造候选网格，再用 pan matrix profile / ClaSP score 选定；默认 `m ∈ {max(3, round(0.05T)), round(0.1T), round(0.2T)}`，受 `m < T - slack` 约束。

## 3. 跨方法结构对比（E3 主表用）

这些是 E3 论文主表真正要呈现的“四方法结构对比”：

| 指标 | 定义 | 方向 | 主表? |
| --- | --- | --- | --- |
| `structure_stability_score` | ClaSP `cross_view_segment_stability` 与 STUMPY `nn_segment_cross_view_consistency` 的加权汇总 | 越高越好 | 是 |
| `replay_consistency_score` | `clap_state_replay_consistency` 与 motif 跨 view 一致性的加权汇总 | 越高越好 | 是 |
| `event_alignment_score` | `cp_tolerance_hit_rate`、`segment_event_purity`、`motif_event_consistency`、`discord_maneuver_overlap` 的加权汇总 | 越高越好 | 是 |
| `fragment_replay_recall` | 片段级事件复盘：给定一个 query 片段，跨 method 能否检索到同事件片段（升级自旧 T3） | 越高越好 | 是 |

加权权重在 manifest 固定并记录；不得为刷分调权。四类 score 只做“方法间结构对比”，不写成“Chronaris 优于 X”的绝对结论。

## 4. TICC 备选指标（第一批不实现）

仅记录未来若实现时的指标合同：

| 指标 | 定义 | 备注 |
| --- | --- | --- |
| `ticc_state_stability` | TICC state assignment 在跨 view/重抽样下的稳定性 | 备选 |
| `ticc_dependency_sparsity` | 每个 state 的 Toeplitz 逆协方差网络稀疏度 | 备选，解释跨维依赖 |
| `ticc_clap_state_agreement` | TICC state 与 CLaP state 的归一化互信息 | 备选，验证状态层一致性 |

边界：TICC 若实现，必须单独说明输入、依赖（`ticc` 包）、超参（平滑性、cluster 数）、随机种子与图表，且不得新增过重实验内容；其结果只作附录增强，不进 E3 主表。

## 5. 结果不理想的处理

- **如实记录**：所有指标（含差值）写进 E3 long 表与 manifest，不删除、不四舍五入美化。
- **不调参刷分**：不得为提升某方法某指标而反复调 `m`、`tol`、降维维度、权重。候选网格在合同里固定。
- **不写绝对优越结论**：E3 叙事只允许“四方法结构对比 + 案例图”，不允许“Chronaris 在 E3 上全面优于 MulT/ContiFormer”这类断言。
- **失败也要呈现**：某方法在某 sortie 上 `too_short` 或 `contract_violation` 时，主表保留该行并标 `status`，不得静默丢弃。
- **与 confirmed metrics 隔离**：E3 指标不回写 `result_matrix_long.csv` / `experiment_registry.csv` / `claim_boundary_table.csv`，不改既有 claim boundary。
