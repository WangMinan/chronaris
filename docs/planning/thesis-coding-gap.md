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
- 当前公开主线仍然是 `NASA closed, UAB partial`，最新统一口径见 `docs/reports/stage_i/stage-i-public-mainline-20260508T091000Z-stage-i-public-mainline-uab-heat-specialist-r1.md`。

因此，当前真正没完成的已经不是“E/F/G/H 还没做出来”，而是论文面向的最后一层编码收束。

## 3. 本轮 UAB 迭代结论

本轮围绕 `UAB subjective` 连续做了四轮真实实跑，并同步改了 `torch` 主线代码：

- `r2 = torch mainline`
  - `full + residual_only`
  - `mean_top2`
  - 结果：`n_back RMSE=5.0619`，`heat_the_chair RMSE=1.6211`
- `r3 = torch + session_mean_broadcast`
  - 结果：`n_back RMSE=4.9450`，`heat_the_chair RMSE=1.6204`
  - 结论：只对 `n_back` 有有限改善，对 `heat_the_chair` 几乎无效
- `r4 = torch + session_pooled_broadcast + physiology_only shortlist`
  - 结果：`n_back RMSE=6.6951`，`heat_the_chair RMSE=1.7997`
  - 结论：session-level pooled supervision 在当前实现下整体退化
- `r5 = torch + session_pooled_broadcast + physiology_scalar_only`
  - 结果：`n_back RMSE=6.6874`，`heat_the_chair RMSE=1.8651`
  - 结论：更接近 `physiology_persistence` 的 scalar profile 仍未把 `heat_the_chair` 拉到 gate 附近

本轮新增/验证过的代码方向：

- `torch` 路线支持 `prediction_aggregation_policy=session_mean_broadcast`
- `torch` 路线支持 `supervision_granularity=session_pooled_broadcast`
- `torch` 候选空间新增 `physiology_only / physiology_scalar_only`
- `torch` 候选家族新增 `linear_huber`
- full LOSO shortlist 不再只看整体均值，已支持保留组内 winner

但当前 best-of 事实没有改变：

- `n_back` 最优仍是旧 `legacy_public_opt`：`4.6103`
- `heat_the_chair` 最优仍是旧 `legacy_public_opt / physiology_persistence`：`1.4567586`
- 因此当前主线结论继续保持：`NASA closed, UAB partial`

### 3.2 2026-05-08 `heat_specialist` 真实 full LOSO 结果

按新的 GPU-first、heat-only 路线又补了一轮真实 `full LOSO`：

- run：`20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1`
- runtime_device：`cuda`
- shortlist 已显式保住：
  - overall RMSE winner
  - overall MAE winner
  - `heat_affine_calibrated_blend`
- screen winner：`heat_residual_correction__lr0p0003__wd0p0001`
- full LOSO best：
  - `heat_the_chair RMSE=1.4630`
  - `heat_the_chair MAE=1.1594`

这轮结果说明：

- 新的 `heat_residual_correction` 已经把 heat-only torch 路线推进到非常接近 legacy best-of 的位置。
- 它在 `MAE` 上已经优于 legacy `physiology_persistence`，但 `RMSE` 仍高于 `1.4568` gate。
- 因此统一公开主线报告仍然必须冻结为 `NASA closed, UAB partial`，不能把这轮写成公开主线闭环。

### 3.1 2026-05-08 可观测与 GPU 防误跑修正

昨晚的 UAB `sklearn --head-catalog uab_hybrid` 长任务出现断连，当前只留下空 artifact 目录，没有有效 `summary / predictions / report`，不能作为论文证据或主线事实引用。

本轮已把后续公开主线改成：

- 长任务默认落盘 `run.log` 与 `progress.json`，并在 CLI 打印 `start / dataset / candidate / subset / fold / metric / output path` 级进度。
- UAB torch 与 `public_fusion_screen` 增加 `require_cuda` 防护；CLI 默认要求 CUDA，只有显式 `--allow-cpu-debug` 才允许 CPU fallback。
- UAB `sklearn --head-catalog uab_hybrid` 归为 CPU-heavy historical reproduction；CLI 必须显式 `--allow-cpu-heavy-sklearn` 才允许运行。
- UAB 下一轮优化入口转向 `heat_the_chair` 专项 `heat_specialist`：`selected_subsets=("heat_the_chair",)`，低维生理 profile、残差修正与训练折内 affine calibration/blend，不再重复训练 `n_back`。
- NASA 保持 `attention_state` 任务定义不变；新增 prepared asset contract 校验和 `processing_diagnostics.json`，防止旧 `event_code / objective_label_text` context 泄漏资产进入 public-opt/public-fusion。

## 4. 仍未完成的编码工作

### 4.1 必须继续处理的项

1. 文档 truth-source 同步
   - `coding-roadmap.md`
   - `docs/reports/stage_i/README.md`
   - unified public mainline report
   这些文档都需要统一到“最新 UAB 迭代已经验证，但 best-of 仍未超出 legacy near-tie”的口径。

2. `UAB clean win` 仍未闭合
   - 若论文只需要“公开数据存在可复现实证，且 NASA 主线已经闭合”，当前可以成立。
   - 若论文要写成“Chronaris 在公开数据上全面优于 `MulT / ContiFormer` 且当前统一 UAB 主线也已闭合”，则编码工作仍未结束。

### 4.2 当前更合理的下一条代码主线

本轮已经证明：

- 继续盲目扩大 `torch` 候选搜索，不会自然把 `heat_the_chair` 推到 `1.4568` 附近。
- `session_mean_broadcast` 只能改善 `n_back`，不能有效解决 `heat_the_chair`。
- `session_pooled_broadcast` 与纯 `physiology_only / physiology_scalar_only` 反而会把 `n_back` 拉坏。

这轮 `heat_specialist` 真实 run 之后，下一条更合理的代码主线已经不再是“继续扩 UAB 候选”，而是：

1. 冻结当前 UAB 公开主线
   - `n_back` 仍引用 legacy `ridge_residual`
   - `heat_the_chair` 仍引用 legacy `physiology_persistence`
   - 新 `heat_specialist` 作为“已验证但未 promote”的最新负/近正证据保留
2. 若最终论文需要最严格公平表述，再补一次 `UAB ContiFormer` confirm rerun
   - 当前 `heat_the_chair` 仍是 near-tie 问题，不是“大幅落后”问题
3. 其余精力转入论文证据整编与图表整理，而不是继续公开训练扩搜

### 4.3 当前不建议继续扩的项

1. 不继续把 `session_pooled_broadcast` 当作默认主线扩大搜索。
2. 不继续盲目增加更多公开数据集来拖大 Stage I 边界。
3. 不把私有 `proxy / weak-label` 任务写成真实 `G-LOC` 或人工真值最优。
4. 不为了论文措辞去重写上游 receiver / 入库链路。

## 5. 推荐的编码收束顺序

1. 冻结当前最新统一口径：
   - `NASA closed`
   - `UAB partial`
   - `legacy_public_opt` 仍是 UAB best-of truth source
2. 将 `20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1` 作为最新 UAB heat-only 真实证据保留。
3. 冻结当前 `public mainline`，转入论文整编与图表整理；除非后续明确决定补 `ContiFormer fairness confirm`，否则不再继续 UAB 候选扩搜。

## 6. 当前判断

当前仓库已经足够支撑论文的核心方法链路：

- 连续对齐
- 物理约束
- 非对称因果融合
- 标准化导出
- 真实 sortie case study
- 私有 proxy 最优性
- 公开数据的可复现实证

但如果把要求提高到下面这两个版本，则编码工作还不能算结束：

1. `公开数据上全面优于 MulT / ContiFormer`
2. `公开 quantitative 主线已经完全 closed`

换句话说，当前剩余工作主要是 thesis-facing closure，而不是底层模型还没搭起来。

## 7. 本轮核查与测试

本轮新增或更新的真实工件：

- `docs/reports/stage_i/archive/public_history/stage-i-public-opt-20260507T133000Z-stage-i-public-opt-uab-torch-mainline-r2.md`
- `docs/reports/stage_i/stage-i-public-opt-20260507T134500Z-stage-i-public-opt-uab-torch-sessionmean-r3.md`
- `docs/reports/stage_i/archive/public_history/stage-i-public-opt-20260507T141500Z-stage-i-public-opt-uab-torch-sessionpooled-r4.md`
- `docs/reports/stage_i/stage-i-public-opt-20260507T142500Z-stage-i-public-opt-uab-torch-sessionpooled-scalar-r5.md`
- `docs/reports/stage_i/stage-i-public-opt-20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1.md`
- `docs/reports/stage_i/stage-i-public-mainline-20260508T091000Z-stage-i-public-mainline-uab-heat-specialist-r1.md`

本轮已运行并通过的相关测试：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt_aggregation tests.test_stage_i_deep_pipeline
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
nvidia-smi
```

CUDA 核查结果：`cuda_available=True`，`device_name=NVIDIA GeForce RTX 4090`；`nvidia-smi` 显示同一张 RTX 4090 可见，但当前有外部 `python3.10` 进程占用显存和算力。

当前残余技术风险：

- `stage_i_metrics.py` 出图时仍有 Matplotlib 中文缺字 warning；不影响指标，但论文图件出图前最好处理。
