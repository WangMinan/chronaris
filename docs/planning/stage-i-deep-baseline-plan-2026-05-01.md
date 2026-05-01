# Stage I Deep Baseline Plan

更新时间：2026-05-01

## 1. 定位

本文件只负责 Stage I 收口后的增强实验，不回退修改已冻结的 `Phase 0 + Phase 1 + Phase 2 + Phase 3` 主线事实。

当前增强实验沿下面顺序推进：

1. `Stage H real sortie`
2. `UAB window_v2`
3. `NASA attention_state`

## 2. 已验证状态

### 已完成

1. 第三方前置准备已落盘：
   - `docs/planning/stage-i-third-party-baseline-prep-2026-04-30.md`
2. 已新增 sequence contract 与统一 bundle：
   - `task_manifest.jsonl`
   - `sequence_bundle.npz`
   - `sequence_schema.json`
   - `dataset_summary.json`
3. 已新增三条 CLI：
   - `scripts/prepare_stage_i_sequences.py`
   - `scripts/run_stage_i_deep_baseline.py`
   - `scripts/run_stage_i_deep_comparison.py`
4. 已新增 `MulT` 双模态 wrapper 与最小 `ContiFormer / PhysioPro` 子集接入。
5. 已新增 `tests/test_stage_i_deep_pipeline.py`，并通过：
   - `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover -s tests -p 'test_stage_i_deep_pipeline.py'`
   - `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline tests.test_stage_i_case_study`

### 第一批真实 sortie 结果

- 已基于 `docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json` 生成 `stage_h_case` sequence 资产：
  - `docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/stage_h_case_sequences/`
- 已完成 `MulT + ContiFormer` 第一批真实 sortie comparison：
  - 机器资产根目录：`docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/comparison/`
  - 主报告：`docs/reports/stage-i-real-sortie-deep-comparison-2026-05-01.md`

### 第二批公开数据 full LOSO

1. 已完成 `UAB window_v2` sequence 资产：
   - `docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/uab_sequences/`
2. 已完成 `NASA attention_state` sequence 资产：
   - `docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/nasa_sequences/`
3. 已完成一轮统一 deep comparison probe：
   - `docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/comparison_probe_all/`
   - `docs/reports/stage-i-deep-comparison-probe-2026-05-01.md`
4. 已完成第二批 full LOSO：
   - `docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/`
   - `docs/reports/stage-i-deep-comparison-full-loso-2026-05-01.md`

## 3. 当前边界

### 已实跑

- `Stage H real_sortie_v1`
- `3 views / 24 sequence samples / 16-step bi-modal projection sequence`
- `MulT` 与 `ContiFormer` 各一次真实 sortie smoke comparison
- `UAB window_v2` sequence 导出：`34748` 条 sequence sample
- `NASA attention_state` sequence 导出：`16609` 条 sequence sample
- `Stage H -> UAB -> NASA` unified comparison probe（`max-folds=1`）
- `UAB / NASA` 双模型 full LOSO 已完成
- `full LOSO` 验证配置：`epochs=1 / batch_size=256 / hidden_dim=32 / num_heads=2 / layers=1`

### 尚未实跑

- 更大 epoch 或更重 hidden size 的调参批次
- 面向论文终稿的最终叙事整理与图表精选

## 4. 下一步

1. 如果需要继续冲指标，优先只对 `ContiFormer + UAB subjective` 做小范围调参，而不是重开全链路。
2. 如果目标转向论文定稿，直接消费 `full LOSO` 资产与主报告整理终稿图表和文字。
3. 若不追加调参，当前第二批增强实验已经具备收口条件。
