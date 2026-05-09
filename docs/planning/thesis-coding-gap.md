# 毕业论文编码缺口评估

更新时间：2026-05-08

## 1. 目的

这份文档只回答一件事：

- 结合 `docs/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx` 与当前仓库/报告现状，为了支撑毕业论文，编码层面还差什么

它不重写阶段历史 closure，也不把 proxy 标签包装成人工真值。

## 2. 当前已经完成到什么程度

对照选题报告里的三段主线：

1. `基于连续时间动力学的异步数据对齐`
2. `基于语义解耦的非对称因果融合`
3. `融合管线的系统级实现与效能验证`

当前仓库已经可以成立的结论是：

- `E / F / G(min) / H` 已完成真实数据链路、导出 contract 与测试闭环。
- `Stage I Phase 0 + Phase 1 + Phase 2 + Phase 3` 的历史公开 benchmark closure 已完成。
- `chronaris_opt` 已在鼎新私有 proxy benchmark 的 `T1 / T2 / T3` 三任务上达到当前对照矩阵最优，并完成 package 固化。
- `alignment support`、`causal support`、`fixed six-path ablation` 三份论文证据 support 已落盘。
- thesis-facing `runtime/demo` 与 `anchor` 入口已正式落盘。
- 当前公开主线已从 `NASA closed, UAB partial` 提升为 `public opt closed`，最新统一口径见 `docs/reports/stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`。

因此，当前真正没完成的已经不是“E/F/G/H 还没做出来”，而是论文面向的最后一层证据表述、图表整理和可选公平性确认。

## 3. 最新 UAB 公开主线结论

### 3.1 2026-05-08 torch `heat_specialist`

GPU-first、heat-only 路线已完成真实 `full LOSO`：

- run：`20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1`
- runtime_device：`cuda`
- winner：`heat_residual_correction__lr0p0003__wd0p0001`
- `heat_the_chair RMSE=1.4630 / MAE=1.1594`

结论：

- 该路线在 `MAE` 上优于 legacy `physiology_persistence`，但 `RMSE` 仍高于 `1.4568` gate。
- 它作为 near-positive evidence 保留，不单独 promote。

### 3.2 2026-05-08 sklearn robust-prior adapter

本轮新增 fold-safe 的 UAB heat adapter，并只跑 `heat_the_chair`：

- run：`20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1`
- artifact：`docs/reports/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/`
- report：`docs/reports/stage_i/stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
- selected_subsets：`heat_the_chair`
- best_head：`target_prior_median`
- `heat_the_chair RMSE=1.4331 / MAE=1.0740`

该 adapter 的论文口径必须保持克制：

- 它是 UAB public adapter / calibration baseline。
- 它只使用每个 LOSO outer fold 的训练 subject 标签中位数，不读取测试 subject 标签。
- 它不能写成双流连续对齐或非对称因果融合本体的直接胜利。
- 它说明 UAB `heat_the_chair` 的剩余卡点主要是 subject-level 主观标签泛化与统计先验上限，而不是 E/F/G/H contract 缺失。

### 3.3 统一 public mainline

最新统一报告：

- report：`docs/reports/stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
- artifact：`docs/reports/assets/stage_i_public_mainline/20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/public_mainline_summary.json`
- status：`public opt closed`

当前 best-of：

| group | best source | best head | metric |
| --- | --- | --- | --- |
| `n_back` | `legacy_public_opt` | `ridge_residual` | `RMSE=4.6103 / MAE=3.8043` |
| `heat_the_chair` | `uab_public_adapter` | `target_prior_median` | `RMSE=1.4331 / MAE=1.0740` |
| `NASA combined` | `public opt round 1` | `balanced_logistic_context` | `macro-F1=0.4550 / balanced_accuracy=0.5591` |

注意：

- UAB 两组都已 clean win。
- `n_back` 的领先幅度仍处于 near-tie 区间，若论文要做最严格公平确认，可补一次 frozen `ContiFormer` confirm rerun。
- `chronaris_public_fusion` 仍是 secondary exploratory branch；当前 `NASA combined macro-F1=0.3348 < 0.40`，不进入主线。

## 4. 当前剩余编码工作

### 4.1 必须完成

1. 文档 truth-source 同步
   - `docs/planning/coding-roadmap.md`
   - `docs/reports/stage_i/README.md`
   - `tests/README.md`
2. 论文证据整理
   - 把 `chronaris_opt` 私有主线、Stage I support、公开 `public opt closed` 分开写。
   - 不把 UAB robust prior 包装成因果融合模块本体的胜利。
3. 图表与表格清洗
   - 汇总 private benchmark、support matrix、public mainline、runtime/demo、anchor。
   - 处理 Matplotlib 中文缺字 warning 后再导出论文图件。

### 4.2 可选完成

1. `UAB ContiFormer` fairness confirm rerun
   - 目的只是不让 `n_back` near-tie 被质疑。
   - 不改变当前 public mainline 已 closed 的事实。
2. `chronaris_public_fusion` NASA-first confirm
   - 仅当继续探索 secondary branch 时执行。
   - 若 `combined macro-F1 <= 0.40`，停止该支线。

### 4.3 不建议继续扩

1. 不继续盲目扩大 UAB torch / sklearn 候选空间。
2. 不继续增加公开数据集来拖大 Stage I 边界。
3. 不把私有 `proxy / weak-label` 任务写成真实 `G-LOC` 或人工真值最优。
4. 不为了论文措辞去重写上游 receiver / 入库链路。

## 5. 推荐收束顺序

1. 冻结当前公开主线：
   - `public opt closed`
   - `n_back = legacy_public_opt / ridge_residual`
   - `heat_the_chair = uab_public_adapter / target_prior_median`
   - `NASA = public opt round 1 / balanced_logistic_context`
2. 冻结私有主线：
   - `chronaris_opt` 仍是鼎新私有任务验证主线。
3. 进入论文整编：
   - 方法链路写 E/F/G/H。
   - 私有最优性写 `chronaris_opt`。
   - 公开数据写 UAB/NASA 可复现实证与 adapter/calibration 边界。

## 6. 本轮核查与测试

本轮新增或更新的真实工件：

- `docs/reports/stage_i/stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
- `docs/reports/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_summary.json`
- `docs/reports/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_predictions.csv`
- `docs/reports/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/run.log`
- `docs/reports/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/progress.json`
- `docs/reports/stage_i/stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
- `docs/reports/assets/stage_i_public_mainline/20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1/public_mainline_summary.json`

本轮已运行并通过的相关测试：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt tests.test_stage_i_public_opt_aggregation
```

真实运行命令使用同一个 `chronaris` 解释器完成，且 `progress.json` 显示本轮 sklearn adapter 只执行了 `selected_subsets=["heat_the_chair"]`。
