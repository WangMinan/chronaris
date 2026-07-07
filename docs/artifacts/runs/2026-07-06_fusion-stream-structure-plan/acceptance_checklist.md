# 验收清单 (Acceptance Checklist) — Fusion Stream Structure Evaluation (E3)

更新时间：2026-07-06
适用对象：后续编码阶段。每完成一个阶段，对应项必须全部通过后方可推进。

## 1. 合同与预处理

- [ ] `contracts.py` 对必需列缺失、`method_name` 非法、sidecar 误用、`time` 非单调均抛 `ContractError` 并写 `status = contract_violation`。
- [ ] `dataset_loader.py` 按 `(method, sortie, view)` 分组、组内按 `time` 升序，行序与 `window_id`/`time` 一一对应。
- [ ] `preprocessing.py` 完成缺失/常量/低方差过滤、组内 z-score、PCA（默认），并写 preprocessing manifest（保留维度、解释方差比、删除列）。
- [ ] `min_T` 保护生效：`T < min_T` 的流标 `too_short` 并降级。
- [ ] 公平性硬约束：未出现 sidecar（`phys_latent_*`/`av_latent_*`/`alignment_score`/`physics_residual`/`partial_data`/`diag_*`）被当作主输入。
- [ ] `test_contracts.py` 通过。
- [ ] `test_preprocessing.py` 通过。

## 2. 合成序列测试

- [ ] 合成多变量序列（注入已知 change points、重复 motif、异常 discord）上：
  - [ ] `cp_tolerance_hit_rate` 随注入边界单调上升。
  - [ ] motif pair 命中注入的重复模式。
  - [ ] discord segment 命中注入的异常段。
  - [ ] `cross_view_segment_stability` 在复制流上为 1。
- [ ] claspy/stumpy 不可用时，合成测试走 fallback / `skipped` 且不崩。

## 3. no-training dry run

- [ ] 合成 T×d 流跑通全链路（loader→preprocess→clasp→stumpy→metrics→reports）。
- [ ] 全程不触发任何模型训练，不依赖真实鼎新数据。
- [ ] run manifest `training_invoked = false`。

## 4. 小规模鼎新 fusion stream dry run

- [ ] 取 1–2 个 sortie/view、四方法（`chronaris`/`mult`/`contiformer`/`naive_time_sync`）、小 T 跑通。
- [ ] 四方法均只用 `fusion_feature_*`；某方法不可得时写 `method_unavailable` 结构化错误。
- [ ] change points / state sequence / motif / discord 均能映射回 `window_id`/`time`。

## 5. 输出与图件

- [ ] E3 long 表存在，`evidence_quadrant = fusion_stream_structure`，`claim_boundary` 固定。
- [ ] `evidence_manifest.json` 完整：记录读取的 artifact、产出的文件、是否训练、外部库可用性。
- [ ] 图件路径存在：状态时间轴图、transition graph 图、片段复盘图（或降级占位图 + manifest 记录）。
- [ ] 不产生 `stage`/`final` 命名；run root 为 `YYYY-MM-DD_intent`。

## 6. 不污染既有证据

- [ ] 未修改 `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/` 下任何文件。
- [ ] 未回写 `result_matrix_long.csv` / `experiment_registry.csv` / `claim_boundary_table.csv`。
- [ ] 未改任何 confirmed metrics。
- [ ] 未删除既有 T3 artifact（T3 仅叙事降级）。

## 7. 仓库卫生

- [ ] `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q` 通过（新增测试 pass，既有测试不退化）。
- [ ] `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall src scripts tests` 通过。
- [ ] `git diff --check` 通过（无空白错误）。
- [ ] `git status` 干净（仅预期文件）。
- [ ] 未在未评估兼容性时修改 `pyproject.toml` / requirements；外部库以 gated import + conda env 安装。
