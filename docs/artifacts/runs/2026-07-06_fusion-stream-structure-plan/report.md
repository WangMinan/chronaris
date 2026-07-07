# 融合表示流结构评价开发计划 (Fusion Stream Structure Evaluation Plan)

更新时间：2026-07-06
计划类型：开发计划（planning only），不是实验结果，不包含任何 confirmed metrics。

## 1. 背景

chronaris 当前服务于硕士论文“航空人机异构时序数据的连续对齐与语义融合方法研究”。仓库主线是：内部双流连续潜态建模，外部输出统一的融合表示流。下游模型统一以融合特征向量作为可消费主输入。

现有代理任务评价（`docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/result_matrix_long.csv`）已经固化了三类代理任务：

- `T1_maneuver_intensity_class`：机动强度分类（任务可用性）。
- `T2_next_window_physiology_response`：下一窗口生理响应回归（任务可用性）。
- `T3_paired_pilot_window_retrieval`：配对飞行员窗口检索（任务可用性，弱信号、mixed，不可夸大）。

这三种任务都回答“融合向量在监督/检索任务上是否可用”，但没有回答一个更结构性、与论文“连续对齐与语义融合”主题直接相关的问题：**四类方法（chronaris、mult、contiformer、naive_time_sync）导出的融合表示流，是否在时间上形成连续、稳定、可复盘的人机状态轨迹？**

本计划新增一层评价，命名为 **E3：融合表示流结构评价（fusion_stream_structure evaluation）**。E3 是无监督的结构性诊断，不是更强的分类器、回归器或普通检索器。

## 2. 本轮范围（只做计划，不编码不训练）

本轮（2026-07-06）只做以下事情：

1. 读取并理解当前仓库状态、目录结构、任务队列、产物索引和论文协议快照。
2. 制定 fusion_stream_structure evaluation 的详细开发计划。
3. 明确 E3 与 T1/T2/T3 的关系，避免重复实验。
4. 固化输入合同、输出合同、指标合同、文件布局、测试策略和验收标准。
5. 更新 docs 索引，使后续较一般的执行模型可以按计划做长程编码。

本轮明确不做：

- 不写任何 `src/`、`scripts/`、`tests/` 实现代码，不创建空代码占位。
- 不安装 claspy/stumpy/ticc，不修改 `pyproject.toml` 或 requirements。
- 不启动训练，不重跑 T1/T2/T3 或任何既有评价。
- 不改任何 confirmed metrics，不改论文协议快照。
- 不把外部仓库源码 vendor 进 chronaris。

本轮读取的关键仓库文件见 `evidence_manifest.json`。

## 3. E3 与 T1/T2/T3 的关系（避免重复实验）

| 评价层 | 问题 | 输入粒度 | 是否监督 | 与论文关系 |
| --- | --- | --- | --- | --- |
| T1 | 融合向量能否判别机动强度 | 单窗口向量 | 弱监督分类 | 任务可用性 |
| T2 | 融合向量能否回归下一窗口生理响应 | 单窗口向量 | 弱监督回归 | 任务可用性 |
| T3 | 融合向量能否检索配对窗口 | 单窗口向量 | 弱监督检索 | 任务可用性（弱信号、mixed） |
| **E3（新增）** | **融合表示流在时间上是否形成连续、稳定、可复盘的状态轨迹** | **连续片段（sortie/view 级 T×d 流）** | **无监督结构诊断** | **直接对应“连续对齐与语义融合”主题** |

关键边界：

- E3 **不替代** T1/T2。T1/T2 仍是任务可用性评价，继续保留。
- E3 **吸收并升级** T3 的检索动机：把“单窗口配对检索”升级为“片段级事件复盘检索”。T3 既有 artifact 不删除、不覆盖，仅在叙事上降级为历史检索诊断 / 片段级复盘的前置参考。
- E3 与 T1/T2/T3 **不共享指标表**。E3 的结果单独成表，并显式标注 `evidence_quadrant = fusion_stream_structure`，不并入既有 `dingxin_model_comparison` / `dingxin_component_ablation` 的 leaderboard，也不与既有 confirmed metrics 混算 delta。
- 弱事件边界、机动代理区间、生理波动代理区间在 E3 中**只用于输出后评价（post-hoc evaluation）**，不进入 ClaSPy/STUMPY 的无监督训练。

推荐执行路线（结论）：

- 保留 T1/T2。
- 将 T3 降级为历史检索诊断或片段级复盘的前置参考。
- 新增 E3：fusion stream structure evaluation。
- E3 第一批实现：ClaSPy / ClaSP-CLaP（状态分段与状态转移）+ STUMPY / Matrix Profile（motif、discord、片段复盘）。
- TICC 暂列备选增强项，第一批不实现。

## 4. 为什么不选这些方案

### 4.1 为什么 MiniRocketMultivariate / TCN 回归器不适合本目标

MiniRocketMultivariate 与 TCN 回归器都是**强监督的特征/回归 learner**。它们的价值在于“给定标签，把表征压到更可分”。E3 的目标是**无监督地诊断融合表示流是否结构良好**，不是再训练一个更强的下游模型。

- 用它们会再次把评价绑定到弱标签，和 T1/T2 重复。
- 它们输出的是预测，不是 change points / state sequence / motif，无法回答“连续、稳定、可复盘的轨迹”这一结构性问题。
- 会引入新的训练随机性与超参，污染既有的“固定 reference”边界。

结论：不纳入 E3 第一批。

### 4.2 为什么 TranAD 只适合作为异常片段备选，不作为主方案

TranAD 是基于 transformer 重构的异常检测模型，理论上能给出异常分数曲线，可用于“异常片段”定位。但：

- TranAD 是**需要训练的重构模型**，会引入训练/验证 split 与随机性，与 E3 “无监督、无训练、可复盘”的目标相悖。
- 它的输出是 per-point 异常分数，不是 state sequence / transition graph / motif pair，结构信息不足。
- 它与既有深度模型训练栈耦合度高，工程成本与解释成本都高于 ClaSPy/STUMPY。

结论：TranAD 仅在后续“异常片段复盘”需要时作为备选，不进入第一批，且必须在 plan 里单独说明它需要训练、不能与 E3 无监督主流程混为一谈。

### 4.3 为什么选择 ClaSPy + STUMPY

ClaSPy 与 STUMPY 都是**参数无关或弱参数、无需训练标签、输出可直接解释**的结构分析工具，正好匹配 E3 的目标：

- **ClaSPy / ClaSP / CLaP**（ermshaua/claspy）：时间序列分割 + 状态检测。
  - `BinaryClaSPSegmentation`：参数无关，输入 1D 或 2D numpy，直接输出 change points；README 明确指出多变量场景下“只提供必要的维度”以保性能。
  - `AgglomerativeCLaPDetection`（CLaP）：参数无关，输出 state sequence；`predict(sparse=True)` 返回 `(states, transitions)`，并可绘制状态转移图。
  - 正好回答“融合流是否形成清晰的人机状态段、状态之间是否有可解释的转移结构”。
  - 引用：ClaSP (DMKD 2023)、ClaSS (VLDB 2024)、CLaP (VLDB 2025)。安装：`python -m pip install claspy`。
- **STUMPY / Matrix Profile**（stumpy-dev/stumpy）：片段级 motif / discord / 语义分割。
  - `stumpy.mstump(ts, m=...)`：多维 Matrix Profile，输入 shape `(d, T)`（每行一个维度），返回 `(matrix_profile, matrix_profile_indices)`；适合直接消费多维融合流。
  - `stumpy.stump(ts, m=...)`：1D Matrix Profile，适合对融合流做低维投影（如第一主成分）后使用。
  - `stumpy.fluss(matrix_profile_distances, L, n_regimes, excl_factor)`：语义分割（FLUSS），返回 `(correct_arc_curve, regime_locations)`，与 ClaSP 的分割互为对照。
  - motif = matrix profile 距离最小的子序列对；discord = 距离最大的子序列（异常/新颖片段）。
  - `stumpy.snippets`：长序列摘要 / 代表片段。
  - 正好回答“融合轨迹里是否存在重复 motif、异常 discord、可复盘的代表片段”，把旧 T3 单窗口检索升级为片段级事件复盘。
  - 引用：Law 2019, JOSS 4(39), 1504；及 Matrix Profile I/II/VI/VIII/XII 等系列论文。安装：`python -m pip install stumpy` 或 `conda install -c conda-forge stumpy`。依赖 NumPy/Numba/SciPy，Python 3.10+。

外部仓库使用边界（必须遵守）：

- 只把 ClaSPy 和 STUMPY 作为 **evaluator**，消费各方法导出的 `fusion_feature_*`。
- 不写成 chronaris 的核心模型组件。
- 不把它们的输出直接写成人工专家标签。
- 不声称它们可以替代 T1/T2。
- 如果依赖安装存在冲突，按 fallback：先实现 contract、synthetic tests、dry-run manifest 和结构化错误报告，再延后真实 evaluator 执行。

### 4.4 为什么 TICC 暂列备选

TICC（Toeplitz Inverse Covariance-Based Clustering）能学习 state assignment 与每个 state 的依赖网络，理论上适合解释融合表示中的跨维依赖结构。但：

- 工程成本高（需要调节平滑性、cluster 数、重复拟合）。
- 解释成本高（依赖网络在论文里需要专门的图与叙述）。
- 与 ClaSP/CLaP 在“状态序列”层面功能重叠，第一批先用 ClaSP-CLaP 覆盖状态层即可。
- 会显著新增实验内容，违背“E3 只补充、不扩张”的边界。

结论：TICC 仅作为备选增强项写入计划（见 `metric_contract.md` §4 与 `implementation_plan.md` §7），第一批不实现，未来若实现必须单独说明输入/依赖/指标/图表，并避免新增过重实验内容。

## 5. 代码结构建议（本轮不创建，仅规划）

建议目录结构（评估后采纳，与现有 `evaluation/dingxin`、`evaluation/public_datasets` 的职责目录风格一致）：

```
src/chronaris/evaluation/fusion_stream_structure/
    __init__.py
    contracts.py            # 输入/输出/指标 dataclass 与校验
    dataset_loader.py       # 把多方法 fusion_feature_* 重组为 T×d 流，映射 window_id/time
    preprocessing.py        # 缺失/常量/低方差列过滤、标准化、PCA/UMAP 轻量降维
    clasp_segmentation.py   # ClaSP change points + CLaP state sequence/transition graph
    stumpy_motif_discord.py # mstump/stump motif、discord、FLUSS、snippets、片段复盘
    ticc_optional.py        # TICC 备选（第一批仅占位接口，不实现）
    metrics.py              # 结构指标计算（hit rate、purity、stability、consistency…）
    reports.py              # 状态时间轴图、transition graph、片段复盘图、manifest 写出

scripts/evaluation/fusion_stream_structure/
    run_fusion_stream_structure_benchmark.py   # CLI 入口，产 run root

tests/evaluation/fusion_stream_structure/
    test_contracts.py
    test_preprocessing.py
    test_metrics.py
    test_report_outputs.py
```

说明：本轮**不创建**上述任何 `src/scripts/tests` 文件。后续编码阶段才创建，且必须先过 `acceptance_checklist.md`。`ticc_optional.py` 在第一批只允许提供一个“未实现/抛 NotImplementedError 或返回结构化 unavailable”的最小接口，不引入 TICC 依赖。

## 6. 实验输入输出合同（摘要）

完整合同见 `input_contract.md` 与 `metric_contract.md`。摘要：

- 主输入列：`method_name, sortie_id, view_id, window_id, time, fusion_feature_1..fusion_feature_d`。
- `method_name` 取值至少覆盖：`chronaris`（等价现有 `chronaris_full`）、`mult`、`contiformer`、`naive_time_sync`。
- 分组：先按 `(method_name, sortie_id, view_id)` 分组，组内按 `time`（或 `window_id`）升序，得到一条 T×d 流。
- 公平性：四类方法**只用** `fusion_feature_*`；Chronaris 不得额外使用 `phys_latent_*`、`av_latent_*`、`alignment_score`、`physics_residual`；双流 sidecar 只作解释/附录，不作主排名输入。
- 输出：per `(method, sortie, view)` 的 change points、state sequence、transition graph、motif pair、discord segment、nearest-neighbor segment、结构指标 long 表、状态时间轴图、transition graph 图、片段复盘图，以及一个 `evidence_manifest.json`。
- 标签/代理字段（弱事件边界、机动代理区间、生理波动代理区间）只用于 post-hoc evaluation。

## 7. 指标与图表设计（摘要）

完整合同见 `metric_contract.md`。摘要：

- ClaSPy 指标：change-point tolerance hit rate、segment-event purity、cross-view segment stability、状态数与转移熵。
- STUMPY 指标：motif 与事件区间一致性、discord 与高机动/高生理波动窗口重合率、跨 view/sortie nearest-neighbor consistency、FLUSS regime 与 ClaSP change point 的一致性。
- TICC 备选指标：state assignment 稳定性、依赖网络稀疏度（仅备选）。
- 指标方向、能否进论文主表、案例图/附录归属在 `metric_contract.md` 逐条标注。
- 结果不理想的处理：如实记录、不调参刷分、不写“Chronaris 在 E3 上优于 X”的夸大结论；E3 主表只给结构指标对比与案例图，不给单一 winner。

## 8. 后续长程编码任务拆解（摘要）

完整拆解见 `implementation_plan.md`。阶段划分：

1. contracts + dataset_loader + preprocessing + 测试（不依赖外部库）。
2. clasp_segmentation + metrics + reports + 测试（依赖 claspy，带 fallback）。
3. stumpy_motif_discord + metrics + reports + 测试（依赖 stumpy，带 fallback）。
4. CLI 脚本 + dry-run + 小规模鼎新 fusion stream dry run。
5. （备选）ticc_optional 接口与最小实现。

每一步必须：只在新 run root 下产出、不改 confirmed metrics、过 `acceptance_checklist.md`、不产生 stage/final 命名。

## 9. 风险与验收标准（摘要）

风险：

- **序列过短**：单个 `(sortie, view)` 的窗口数 T 可能远小于 ClaSP/STUMPY 示例（数千点）。需在 preprocessing/合同里定义最小 T 阈值，过短则降级为“仅 discord/仅 FLUSS”或跨 view 拼接并显式标注。
- **维度过高 / 低方差列**：fusion_feature_d 可能较高。需低方差列过滤 + 标准化 + 轻量降维（PCA，必要时 UMAP），并对降维保持可复盘（记录保留维度与解释方差比）。
- **依赖冲突**：claspy/stumpy 与现有 conda env `chronaris` 的 NumPy/Numba/SciPy 版本可能冲突。fallback 见上。
- **公平性漂移**：长程编码中容易把 sidecar 偷偷接进主输入。需在 contracts 里强制校验。

验收标准见 `acceptance_checklist.md`。

## 10. 引用

- ClaSP: Ermshaus, Schäfer, Leser. *ClaSP: parameter-free time series segmentation.* DMKD 37, 1262–1300 (2023).
- ClaSS: Ermshaus, Schäfer, Leser. *Raising the ClaSS of Streaming Time Series Segmentation.* PVLDB 17(8), 1953–1966 (2024).
- CLaP: Ermshaus, Schäfer, Leser. *CLaP – State Detection from Time Series.* PVLDB 19(1), 70–83 (2025).
- STUMPY: Law, S.M. *STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining.* JOSS 4(39), 1504 (2019).
- Matrix Profile 系列：Yeh et al. (2016) MP I；Zhu et al. (2016) MP II；Yeh et al. (2017) MP VI（多维 motif）；Gharghabi et al. (2017) MP VIII（FLUSS 语义分割）；Gharghabi et al. (2018) MP XII（MPdist）。
