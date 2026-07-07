# 实施计划 (Implementation Plan) — Fusion Stream Structure Evaluation (E3)

更新时间：2026-07-06
状态：供后续执行模型按阶段做长程编码的拆解。本轮不编码。

## 0. 总原则

- 每个阶段都在**新 run root** `docs/artifacts/runs/<YYYY-MM-DD>_fusion-stream-structure-*/` 下产出，不污染既有 run。
- 不改 confirmed metrics：不写 `docs/artifacts/runs/2026-07-03_thesis-protocol-snapshot/` 下任何文件，不回写 `result_matrix_long.csv` / `experiment_registry.csv` / `claim_boundary_table.csv`。
- 不产生 `stage` / `final` 命名（沿用 `YYYY-MM-DD_intent` 命名）。
- 每阶段必须过 `acceptance_checklist.md` 对应项。
- 外部库（claspy/stumpy）一律 **gated import**（try/except），不可用时走 fallback，不阻塞 contract/synthetic/dry-run。

## 1. 阶段一：contracts + dataset_loader + preprocessing

目标：把多方法 `fusion_feature_*` 重组为 T×d 流，完成预处理，不依赖任何外部库。

新增文件：

- `src/chronaris/evaluation/fusion_stream_structure/__init__.py`
- `src/chronaris/evaluation/fusion_stream_structure/contracts.py`
  - `FusionStreamRecord`、`FusionStreamContract`、`ContractError`。
  - 校验逻辑见 `input_contract.md` §8。
- `src/chronaris/evaluation/fusion_stream_structure/dataset_loader.py`
  - `load_fusion_streams(...)` → 按 `(method, sortie, view)` 分组、组内按 `time` 排序，返回 `dict[(method, sortie, view), FusionStreamRecord]`。
  - 对接既有 feature export window manifest（见 `input_contract.md` §7），不重新训练/导出。
- `src/chronaris/evaluation/fusion_stream_structure/preprocessing.py`
  - 缺失/常量/低方差列过滤、组内 z-score、PCA（默认）/UMAP（可选）、`min_T` 保护。
  - 全程写 preprocessing manifest（保留维度、解释方差比、删除列）。

新增测试：

- `tests/evaluation/fusion_stream_structure/__init__.py`
- `tests/evaluation/fusion_stream_structure/test_contracts.py`：必需列缺失、`method_name` 非法、sidecar 列误用、`time` 非单调均抛 `ContractError`。
- `tests/evaluation/fusion_stream_structure/test_preprocessing.py`：低方差列剔除、标准化可逆性、PCA 解释方差比记录、`too_short` 标记。

需读取的既有 artifact：

- `docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/`、`...-f-allwindow-clean/`、`docs/artifacts/runs/2026-04-27_feature-export-closure/`（window manifest 结构）。
- `src/chronaris/evaluation/dingxin/pipelines/benchmark_data.py`（`load_aligned_private_records`、`base_feature_row`、`build_*_feature_frame` 变体映射）。
- `src/chronaris/evaluation/dingxin/pipelines/thirdparty_comparison.py`（method→variant 映射）。

验收：阶段一全部测试通过；`compileall` 通过；不引入外部依赖。

## 2. 阶段二：clasp_segmentation + metrics + reports（依赖 claspy，带 fallback）

新增文件：

- `src/chronaris/evaluation/fusion_stream_structure/clasp_segmentation.py`
  - gated import：`try: from claspy.segmentation import BinaryClaSPSegmentation; from claspy.state_detection import AgglomerativeCLaPDetection`。
  - 不可用时：返回 `status = claspy_unavailable` + 结构化错误，不抛到 CLI 崩溃。
  - 输出 change points、state sequence、transition graph，映射回 `window_id`/`time`。
- `src/chronaris/evaluation/fusion_stream_structure/metrics.py`（先实现 ClaSP 部分）
  - `cp_tolerance_hit_rate`、`segment_event_purity`、`state_count`、`transition_entropy`、`cross_view_segment_stability`、`clap_state_replay_consistency`。
- `src/chronaris/evaluation/fusion_stream_structure/reports.py`
  - 状态时间轴图、transition graph 图（ ClaSPy 自带 `plot` 优先；不可用时用 matplotlib 自绘占位并记录降级）。

新增/扩展测试：

- `tests/evaluation/fusion_stream_structure/test_metrics.py`：合成多变量序列（带已知 change points）上 hit rate/purity 单调性；cross-view stability 在复制流上为 1。
- `tests/evaluation/fusion_stream_structure/test_report_outputs.py`：图件路径存在、manifest 字段完整；claspy 不可用路径走 fallback 不崩。

依赖安装方式（评估后建议，**本轮不改 pyproject**）：

- 先在 conda env `chronaris` 内试装：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pip install claspy`。
- 验证 NumPy/Numba/SciPy 兼容；若冲突，保持 gated import + fallback，并在 manifest 记录“真实 evaluator 延后”。
- 是否写进 `pyproject.toml` 的 `[project.optional-dependencies]`（如 `e3 = ["claspy", "stumpy"]`）留给后续编码阶段单独评审决定；本轮不动。

## 3. 阶段三：stumpy_motif_discord + metrics + reports（依赖 stumpy，带 fallback）

新增文件：

- `src/chronaris/evaluation/fusion_stream_structure/stumpy_motif_discord.py`
  - gated import：`try: import stumpy`。
  - `mstump`（多维，默认）+ `stump`（第一主成分，对照）。
  - motif pair、discord segment、nearest-neighbor segment、FLUSS regime、snippets。
  - `m` 候选网格见 `metric_contract.md` §2。
- `metrics.py` 扩展 STUMPY 部分：`motif_event_consistency`、`discord_maneuver_overlap`、`discord_physio_overlap`、`nn_segment_cross_view_consistency`、`fluss_clasp_agreement`、`mp_discord_isolation`、`snippet_coverage`。
- `reports.py` 扩展：片段复盘图（query 片段 ↔ nearest-neighbor 片段，叠机动/生理代理区间）。

测试扩展：合成序列上 motif 命中注入重复模式、discord 命中注入异常段；fallback 路径。

依赖安装：`pip install stumpy` 或 `conda install -c conda-forge stumpy`，同样 gated + 兼容性评估。

## 4. 阶段四：CLI + dry-run + 小规模鼎新 dry run

新增文件：

- `scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py`
  - 参数：`--fusion-stream-manifest`（或既有 feature export run manifest 路径）、`--methods`、`--output-root`、`--report-root`、`--m-grid`、`--tol`、`--min-T`、`--device`（保持与既有 dingxin 脚本风格一致：默认 `auto`）。
  - 默认 `--output-root=docs/artifacts/runs`，run id 用 `YYYYmmddTHHMM%SZ-fusion-stream-structure`。
  - 产出 E3 long 表、manifest、图件。

执行（后续阶段，不在本轮）：

- no-training dry run：合成 T×d 流，验证全链路不依赖真实数据、不训练。
- 小规模鼎新 fusion stream dry run：取 1–2 个 sortie/view、四方法、小 T，验证 contract + 结构输出 + manifest。
- 不跑全量、不跑 GPU 实验。

## 5. 阶段五（备选）：ticc_optional

- `src/chronaris/evaluation/fusion_stream_structure/ticc_optional.py`：第一批仅提供最小接口（返回 `status = ticc_not_implemented` 或结构化 unavailable），不引入 `ticc` 依赖。
- 真正实现需单独 plan，且只进附录。

## 6. 文件改动一览（后续编码阶段）

| 文件 | 动作 | 阶段 |
| --- | --- | --- |
| `src/chronaris/evaluation/fusion_stream_structure/*.py` | 新增 9 个模块 | 1–5 |
| `scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py` | 新增 | 4 |
| `tests/evaluation/fusion_stream_structure/*.py` | 新增 4 个测试 | 1–3 |
| `docs/artifacts/ARTIFACTS.md` | 追加 E3 run（实验产出时） | 4 |
| `docs/implementation/TASKS.md` | 推进队列 | 4 |
| `pyproject.toml` | 仅在依赖兼容性确认后，按 optional-dependencies 追加（单独评审） | 2/3 |

## 7. 依赖不可用时的 fallback 顺序

1. gated import 失败 → 该 evaluator 标 `*_unavailable`，写结构化错误到 manifest。
2. contract / dataset_loader / preprocessing / metrics 的纯逻辑测试照常通过（不依赖外部库）。
3. 合成序列测试在无外部库时用轻量占位（自绘 matrix profile 近似或跳过并标 `skipped`），保证 CI 绿。
4. 真实 evaluator 执行延后到依赖装好后再跑，**绝不**为了跑通而 vendor 第三方源码或伪造指标。

## 8. 如何避免长程任务中途污染 confirmed metrics

- E3 的所有输出只写到 E3 专属 run root 与 E3 long 表，禁止写既有 `2026-07-03_thesis-protocol-snapshot/`、`2026-07-02_*` run。
- E3 long 表的 `evidence_quadrant` 固定为 `fusion_stream_structure`，`claim_boundary` 固定为“无监督结构诊断，非专家真值，不与 T1/T2/T3 leaderboard 混算”。
- 任何把 E3 指标回写 `result_matrix_long.csv` 的代码路径视为违规，code review 必拦。
- 阶段切换前跑 `git diff --check` + `compileall` + 受影响 `pytest`，确认未误改既有 artifact。
