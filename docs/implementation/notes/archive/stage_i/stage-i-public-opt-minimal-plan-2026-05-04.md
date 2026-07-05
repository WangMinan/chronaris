# Stage I Public Opt 最小可跑计划

更新时间：2026-05-04

> 说明：本文件已归档为 `chronaris public opt` 启动期最小方案快照。  
> 当前现行入口改为 `docs/implementation/notes/stage-i-thesis-mainline-roadmap-2026-05-15.md` 与 `docs/implementation/notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md`。

## 1. 目标

本计划的目标不是重新设计 Stage I，而是把 `chronaris_opt` 的核心思路迁到现有公开数据 `sequence contract` 上，做出一个：

1. 不破坏既有 `task_manifest.jsonl / sequence_bundle.npz / sequence_schema.json / dataset_summary.json` 的最小实现
2. 能直接在当前 `chronaris` 仓库内复用 `LOSO + report + prediction artifact` 路径的最小可跑版本
3. 优先服务 `UAB subjective regression`

当前默认命名：

- `chronaris public opt`

## 2. 最小范围

### 本轮纳入

1. 数据集：
   - `UAB workload dataset`
2. 主任务：
   - `subjective regression`
3. 复用入口：
   - `src/chronaris/pipelines/stage_i/stage_i_sequence_preparation.py`
   - `src/chronaris/features/stage_i_sequences.py`
   - `src/chronaris/pipelines/stage_i/stage_i_baseline_models.py`
   - `src/chronaris/evaluation/stage_i_metrics.py`

### 本轮不纳入

1. `NASA attention_state`
2. 新增顶层包
3. 重开 `Stage H` 导出 contract
4. 替换现有 `MulT / ContiFormer` 报告
5. 复杂大模型调参

## 3. 设计原则

`chronaris public opt` 只迁移思路，不强行复制鼎新真实数据 benchmark 的所有实现细节。

本轮保留三条核心：

1. 非对称双流：
   - `physiology` 作为主流
   - `task_context / scenario_context` 作为辅流
2. 轻量 residual/context 特征：
   - 从现有 `sequence bundle` 与 `entry.context_payload` 中派生
   - 不依赖鼎新 `raw_window_summary.jsonl`
3. 任务感知轻量 head：
   - 优先 regression
   - 先求可跑和稳定，再谈更重模型

## 4. 推荐实现路径

### Phase A: 公共 opt 特征帧

新增一个公共 opt 的 frame builder，输入为：

- `StageISequenceEntry[]`
- `StageISequenceBundle`

输出为：

- `sample_id`
- `split_group`
- `subset_id`
- `feature_values`
- `target metadata`

最小特征组建议：

1. `physiology pooled sequence stats`
   - mean / std / min / max / delta
2. `context modality pooled stats`
   - 对第二模态做同样 pooled stats
3. `time-axis context`
   - sequence length
   - effective valid ratio
   - window fraction
4. `public residual proxies`
   - physiology end-start delta weak-label
   - physiology short-term variability weak-label
   - context intensity weak-label

建议文件：

- `src/chronaris/pipelines/stage_i/stage_i_public_opt_data.py`

### Phase B: 轻量任务头

先只做 `UAB subjective regression`。

推荐保留两个 head：

1. `physiology_persistence`
   - 作为回归下限与 fallback
2. `ridge_residual`
   - 输入使用公共 residual/context 特征

输出：

- `predictions.csv`
- `summary.json`
- `report.md`

建议文件：

- `src/chronaris/pipelines/stage_i/stage_i_public_opt.py`

### Phase C: CLI

新增一条最小 CLI：

- `scripts/run_stage_i_public_opt.py`

职责：

1. 读取已准备好的 sequence assets
2. 执行 `UAB subjective regression`
3. 落盘 summary / report / predictions

## 5. 推荐目录落点

### 代码

- `src/chronaris/pipelines/stage_i/stage_i_public_opt_data.py`
- `src/chronaris/pipelines/stage_i/stage_i_public_opt.py`

### 脚本

- `scripts/run_stage_i_public_opt.py`

### 测试

- 并入 `tests/test_stage_i_deep_pipeline.py`
- 或新增 `tests/test_stage_i_public_opt.py`

本轮更推荐：

- 新增 `tests/test_stage_i_public_opt.py`

原因：

- 公共 opt 的目标是新主线尝试，不应把 `MulT / ContiFormer` 的历史 deep baseline 测试再缠得更重

## 6. 最小测试集

至少覆盖：

1. synthetic UAB prepared asset 可生成 public opt frame
2. regression 两个 head 都能在 LOSO 下跑通
3. report / summary / predictions 都会落盘
4. 非有限值输出可被兜底处理

推荐命令：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt
```

若接入现有 preparation/deep suite 联合回归，再补：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline tests.test_stage_i_deep_pipeline tests.test_stage_i_public_opt
```

## 7. 首轮成功标准

本轮最小成功标准不是“立刻超过所有 baseline”，而是：

1. `chronaris public opt` 路径跑通
2. `UAB subjective regression` summary / report / predictions 落盘
3. 结果能直接和现有：
   - classical baseline
   - `ContiFormer`
   - `MulT`
   做同口径比较

额外加分项：

1. `n_back` 上接近或超过当前 `ContiFormer`
2. `heat_the_chair` 上明显优于 classical baseline

## 8. 第二步扩展

如果最小版本跑通，下一步按下面顺序扩：

1. 把同一套 frame/head 抽象到 `NASA attention_state`
2. 再决定是否需要把公共 opt 扩成：
   - 更强的 asymmetric cross-attention wrapper
   - 更重的 unified public comparison

## 9. 下一轮实现顺序

建议严格按下面顺序推进：

1. 写 `stage_i_public_opt_data.py`
2. 写 `stage_i_public_opt.py`
3. 写 `tests/test_stage_i_public_opt.py`
4. 写 `scripts/run_stage_i_public_opt.py`
5. 用 synthetic UAB 回归跑通
6. 再用真实 prepared UAB assets 跑第一轮结果
