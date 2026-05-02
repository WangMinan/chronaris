# Stage I 私有双流 benchmark 执行计划

更新时间：2026-05-02

## 1. 目标

当前新增的最高优先级不是继续扩公开数据，而是：

1. 在私有双流 sortie 上导出 `all-window / all partitions` 资产
2. 在同一批私有窗口上验证 `naive_sync -> E -> F -> G(min) -> G(no causal mask)` 模块增益
3. 在 T1/T2/T3 三个私有 proxy 任务上比较优化后的 `chronaris_opt` 与 `naive_sync / E / F / G(min) / no-mask / MulT / ContiFormer`

本计划对应的优化候选已经完成 full LOSO 实跑。当前状态只应表述为：

- `chronaris_opt` 在鼎新私有 proxy benchmark 的 T1/T2/T3 三任务上达到当前对照矩阵全面最优
- `private_optimality_supported=True`
- 这些标签仍是 proxy / 弱标签，不应写成人工真值

## 2. 已接入代码入口

### Stage H all-window 导出 contract

- `src/chronaris/pipelines/alignment_preview.py`
  - `intermediate_partition` 已支持 `all`
  - `intermediate_sample_limit=None` 时可导出全部样本
- `src/chronaris/pipelines/stage_h_export.py`
  - view 执行结果会保留 `sample_partition_by_id`
- `src/chronaris/pipelines/stage_h_export_helpers.py`
  - `window_manifest.jsonl` 已新增 `sample_partition`
  - `feature_bundle.npz` 已新增：
    - `sample_ids`
    - `sample_partitions`
    - `physiology_reference_hidden`
    - `vehicle_reference_hidden`
  - 每个 view 会额外导出 `raw_window_summary.jsonl`

### 私有任务与 benchmark 管线

- 任务 contract：`src/chronaris/dataset/stage_i_private_contracts.py`
- 主入口：`src/chronaris/pipelines/stage_i_private_benchmark.py`
- 优化候选与任务头：`src/chronaris/pipelines/stage_i_private_optimization.py`
- 导出报告：
  - `private-alignment-support-<run_id>.md`
  - `private-causal-fusion-support-<run_id>.md`
  - `private-optimality-summary-<run_id>.md`
  - `private-optimization-summary-<run_id>.md`
  - `optimized_candidate_summary.json`
  - `optimized_candidate_metrics.csv`

## 3. 当前固定任务

### T1 maneuver_intensity_class

- 来源：`raw_window_summary.jsonl` 里的车辆窗口统计
- 策略：按候选机动字段的 `abs(delta) + std + span` 聚合打分，再按全体分位数离散为 `low / medium / high`

### T2 next_window_physiology_response

- 来源：同 view 下一窗口的生理统计
- 策略：用下一窗口生理字段变化强度构造回归标签

### T3 paired_pilot_window_retrieval

- 来源：同 sortie、同 `window_index` 的双 pilot 配对
- 策略：在同 sortie 的对侧 pilot 候选集中做 top-1 / MRR 检索

## 4. 当前对照矩阵

- `naive_sync`
  - 使用 raw-window 聚合特征
- `E baseline`
  - 使用无 physics 的 reference projection 聚合特征
- `F full`
  - 使用 full physics 的 reference projection 聚合特征
- `G min`
  - 基于 F hidden state 离线重建因果融合表示
- `G no causal mask`
  - 基于 F hidden state 离线重建无因果掩码表示
- `chronaris_opt`
  - 基于 F hidden state 的 lag-aware causal fusion、raw-window residual 与任务感知轻量头
- `chronaris_opt_no_causal_mask`
  - 同一优化候选关闭 causal mask 的配对诊断对照
- `classical`
  - `window_v2` 线性分类 / 回归 baseline
- `MulT / ContiFormer`
  - 使用 F full 的双流 reference projection sequence 作为外部深基线输入

## 5. 建议执行顺序

1. 先用 `StageHExportConfig.preview_config.intermediate_partition='all'` 和 `intermediate_sample_limit=None` 重跑 E / F all-window 资产。
2. 再执行私有 benchmark 编排入口。
3. 只在三份私有报告和资产都落盘后，再回写“是否证明最优”的状态文本。

## 6. 建议命令

以下命令仍应显式使用 `chronaris` 解释器：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_h_export tests.test_stage_i_deep_pipeline tests.test_stage_i_private_optimization
```

真实 benchmark 实跑时，应在调用脚本里构造：

- 一个 `E all-window` run manifest
- 一个 `F all-window` run manifest
- 一个 `StageIPrivateBenchmarkConfig`

再调用：

```python
from chronaris.pipelines import (
    StageIPrivateBenchmarkConfig,
    run_stage_i_private_benchmark,
)
```

## 7. 当前状态边界

当前仓库状态是：

- `all-window private benchmark scaffolding 已接入`
- `synthetic / contract 回归已通过`
- `E/F all-window clean run` 已完成，`E` / `F` 均导出 `3` 个 view，且 `sample_manifest` 对齐
- `private benchmark smoke` 已完成并已被 full run 覆盖，docs/assets 中不再保留中间 smoke 产物
- `private benchmark full` 已完成：`docs/reports/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/`
- 私有 benchmark 的最终结论是：
  - `private_optimality_supported = True`
  - T1：`chronaris_opt` macro-F1 `1.000000`、balanced accuracy `1.000000`
  - T2：`chronaris_opt` RMSE `201.489565`、MAE `113.851926`
  - T3：`chronaris_opt` top-1 `1.000000`、MRR `1.000000`
  - `chronaris_opt` 在 T1/T2/T3 均优于 `chronaris_opt_no_causal_mask`
  - `criterion_details` 全为 `true`

因此当前可以说“已在鼎新私有 proxy benchmark 上证明 `chronaris_opt` 三任务全面最优”。该结论不外推为人工真值或真实飞行风险标签最优。
