# 输入合同 (Input Contract) — Fusion Stream Structure Evaluation (E3)

更新时间：2026-07-06
状态：规划合同，后续编码阶段据此实现 `contracts.py` / `dataset_loader.py` / `preprocessing.py`。

## 1. 主输入行长式合同

E3 的主输入是一张“长表”，每行表示一个窗口的融合向量。列定义：

| 列名 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `method_name` | str | 必需 | 导出该融合向量的方法。取值见 §3。 |
| `sortie_id` | str | 必需 | 架次标识。 |
| `view_id` | str | 必需 | 视角/视图标识（与既有 feature export 的 view_id 一致）。 |
| `window_id` | str/int | 必需 | 窗口唯一标识（等价既有 `sample_id` / `raw_sample_id` 语义）。 |
| `time` | numeric | 必需 | 窗口时间戳或序号，用于组内排序。优先用 `start_offset_ms`；无则用 `window_index`。 |
| `fusion_feature_1..fusion_feature_d` | float | 必需 | 融合向量各维。列名统一为 `fusion_feature_<i>`，`i` 从 1 起。 |

## 2. 可选 / post-hoc 列（只用于评价，不进无监督训练）

| 列名 | 类型 | 用途 | 边界 |
| --- | --- | --- | --- |
| `maneuver_proxy_label` | str/float | 机动代理标签（来自 T1 构造） | 仅 post-hoc：hit rate、purity、重合率 |
| `maneuver_event_interval` | bool/区间 | 机动代理事件区间 | 仅 post-hoc |
| `physio_fluctuation_interval` | bool/区间 | 生理波动代理区间 | 仅 post-hoc |
| `weak_event_boundary` | int/区间 | 弱事件边界（change point 参考） | 仅 post-hoc，tolerance hit rate |
| `pilot_id` | int | 飞行员标识 | 仅用于分组/案例图，不进无监督训练 |
| `sample_partition` | str | 既有划分标记 | 仅用于跨 view/sortie consistency 分组 |

**强制约束**：上述可选列不得作为 ClaSPy/STUMPY 的输入特征，不得参与无监督拟合。它们只在对结构输出（change points、states、motif、discord）做事后比对时使用。

## 3. `method_name` 取值

E3 主表至少覆盖以下四类方法，且四类**必须使用相同导出口径的 `fusion_feature_*`**：

| `method_name`（E3 合同） | 既有仓库对应变体 | 说明 |
| --- | --- | --- |
| `chronaris` | `chronaris_full`（thirdparty_comparison.py） | Chronaris 完整模型融合流 |
| `mult` | `mult` | MulT 融合流 |
| `contiformer` | `contiformer` | ContiFormer 融合流 |
| `naive_time_sync` | `naive_time_sync`（→ `naive_sync`） | 朴素时间同步基线融合流 |

实现备注：

- 合同层接受 `chronaris` 与 `chronaris_full` 两个别名，统一归一为 `chronaris`。
- 若某方法在既有导出中尚未产出 `fusion_feature_*` 长表，E3 在该 method 上**降级为 unavailable**并写结构化错误，不得用 sidecar 或投影向量冒充融合流。

## 4. 排序与分组方式

1. 按 `(method_name, sortie_id, view_id)` 分组，每组对应一条流。
2. 组内按 `time` 升序（`time` 缺失时回退 `window_id` / `window_index`）。
3. 排序后得到矩阵 `X ∈ R^{T × d}`，其中：
   - `T` = 该组窗口数（流的长度）。
   - `d` = `fusion_feature_*` 维数。
   - 行序与 `window_id` / `time` 一一对应，用于把结构输出（change point 索引、state 序列、motif/discord 区间）映射回原始窗口。
4. ClaSPy 的多变量输入直接用 `X.T`（ClaSPy 接受 2D numpy，行/列即维度/时间，与 STUMPY `mstump` 的 `(d, T)` 约定一致；实现时按各自 API 转置并记录在 manifest）。

## 5. fusion_feature_* 预处理合同（在 `preprocessing.py` 实现）

按以下顺序处理，**每一步都要在 manifest 记录**（保留/删除的列、参数、解释方差比），保证可复盘：

1. **缺失值**：
   - 整列全缺：删除该列并记录。
   - 部分缺失：按列做组内（同一条流）均值填充；若填充后仍不稳定则置 0 并记录；不得跨 method 借填。
2. **常量列 / 低方差列**：方差低于阈值（默认 `1e-8`，可配）的列删除并记录。
3. **标准化**：组内 z-score（按每条流独立标准化），避免不同 sortie 量纲污染。记录均值/方差是否可逆。
4. **降维（仅在 `d` 较高或 T 较少时触发）**：
   - 默认 PCA，保留累计解释方差比 ≥ 阈值（默认 0.95）的主成分；记录保留维度 `d'` 与解释方差比。
   - UMAP 仅作为可选非默认项；使用时必须固定 `random_state` 并记录。
   - ClaSPy README 明确：“只提供必要的维度”以保性能，因此降维是推荐步骤而非可选。
5. **极小 T 保护**：若 `T < min_T`（默认 `min_T = 30`，可配），该流标记 `too_short`：
   - 不跑 ClaSP（至少不做 CLaP 状态检测）。
   - 仅在 `T ≥ m + slack` 时跑 STUMPY motif/discord。
   - manifest 显式记录降级原因。

## 6. 公平性硬约束（与论文边界一致）

1. 四类方法的 E3 主输入**只用** `fusion_feature_*`。
2. **禁止**让 Chronaris 额外使用 `phys_latent_*`、`av_latent_*`、`alignment_score`、`physics_residual`、`partial_data` sidecar、`diag_attention_entropy`、`diag_top_event_concentration`、`diag_event_mask_interference` 等内部诊断列作为结构评价输入。这些列只能出现在附录解释材料里。
3. Chronaris 的双流 sidecar 只作为解释/附录材料，不作为主排名输入。
4. 生理/航电是否都被利用，应通过统一 input variant / modality ablation 协议验证：`both_stream`、`phys_only`、`av_only`。
   - 若现有导出不支持 `phys_only` / `av_only`，本轮**只列入开发计划**，不强行改数据。
   - 实现时：若某 variant 不可得，E3 对该 variant 写 `variant_unavailable` 结构化错误，不得用拼接/投影伪造。

## 7. 来源对接（既有 feature export）

E3 的 `dataset_loader` 应从既有产物按需重组，不新建上游数据：

- 既有 feature export 产物（`docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/`、`...-f-allwindow-clean/`、`docs/artifacts/runs/2026-04-27_feature-export-closure/`）的 per-view window manifest（jsonl）含 `sample_id`、`sortie_id`、`view_id`、`pilot_id`、`window_index`、`start_offset_ms`、`end_offset_ms`、`sample_partition`、`physiology_feature_stats`、`vehicle_feature_stats`。
- 第三方对比（`src/chronaris/evaluation/dingxin/pipelines/thirdparty_comparison.py`）已为 `chronaris_full`/`mult`/`contiformer`/`naive_time_sync`/`classical_baseline` 定义变体到特征帧的映射。
- E3 的职责是：把上述**逐窗口融合向量**按 `(method, sortie, view)` 重组为连续 T×d 流，再交给 ClaSPy/STUMPY。E3 不重新训练任何模型，不重新导出 feature export。

## 8. 校验失败行为

`contracts.py` 必须在加载时校验：

- 必需列存在且非全空。
- `method_name` 取值在允许集合内（或显式注册新方法）。
- 每条 `(method, sortie, view)` 流的 `time` 单调（排序后）。
- `fusion_feature_*` 列数一致（同一 method 内 d 一致；跨 method 允许不同 d，但分别记录）。
- 未出现被禁用的 sidecar 列被当作主输入。

任一校验失败：抛出结构化 `ContractError`，并在 run manifest 写 `status = contract_violation` + 具体原因，不静默继续。
