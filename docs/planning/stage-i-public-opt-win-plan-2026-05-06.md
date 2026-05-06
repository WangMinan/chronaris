# Stage I public opt 超越 MulT / ContiFormer 执行计划

更新时间：2026-05-06

## 1. 目标

本计划只回答一件事：

- 让当前 `chronaris public opt` 在公开数据上优于现有 `MulT / ContiFormer` full LOSO 对照

当前固定验收口径：

1. `UAB subjective`
   - `n_back` 与 `heat_the_chair` 两组都要赢
   - 主指标：`RMSE`
   - 同分裁决：`MAE`
2. `NASA attention_state`
   - 主 gate 只看 `combined`
   - 主指标：`macro-F1`
   - 同分裁决：`balanced_accuracy`
   - `benchmark_only / loft_only` 保留为诊断列，不作为主 gate

当前固定对照口径：

- 默认冻结 `docs/reports/stage_i/stage-i-deep-comparison-full-loso-2026-05-01.md` 中的 `MulT / ContiFormer` 历史结果
- 只有在新结果“接近但不稳”时，才重跑一次对应 deep baseline 做公平确认
- 本轮不再以 `classical baseline` 作为前进门槛

## 2. 执行顺序

默认资源策略：

- 不再继续扩大 CPU 侧 `sklearn` 网格搜索
- 高成本新增训练默认优先迁到 `torch/GPU`
- `chronaris_public_fusion` 是当前公开主线的 GPU 优先承接入口
- 当前运行时注意：本机 `nvidia-smi` 可见 `RTX 4090`，但 `chronaris` 环境内的 `torch.cuda.is_available()` 仍可能不稳定，需要把 CUDA runtime 可用性作为进入 GPU 主线前的第一检查项

### Round 1：强化现有 public opt

保留当前公开 sequence contract，不改 `Stage H` contract，不引入新顶层包。

固定改动：

1. 扩展特征帧
   - 保留现有全序列 pooled stats
   - 新增 `early / middle / late` 三段强度与变化特征
   - 新增主流/辅流强度差、变化差、有效率差等跨模态标量
   - 新增辅流 jump / peak / change density 代理特征
2. 扩展 `public opt` head family
   - `UAB subjective`
     - `physiology_persistence`
     - `ridge_residual_cv`
     - `elasticnet_residual`
     - `huber_residual`
   - `NASA attention_state`
     - `physiology_margin_balanced_logistic`
     - `balanced_logistic_context`
     - `balanced_linear_svc_context`
3. 固定输出
   - `predictions.csv`
   - `summary.json`
   - `report.md`
   - `winning_margin_vs_deep`
   - `needs_deep_rerun`
4. 资源边界
   - Round 1 允许保留少量 CPU 线性模型搜索
   - 若单轮成本明显过高，不再追加更大的 CPU 搜索，而是提前转入 `chronaris_public_fusion`

### Round 2：只在 Round 1 未过 gate 时继续强化 public opt

固定策略：

1. 只保留 Round 1 中每个数据集最好的 `2` 条头
2. 在相同 head family 上切换 `feature_profile`
   - `full`
   - `physiology_only`
   - `context_only`
   - `residual_only`
3. 启用轻量集成
   - `UAB`：top-2 mean ensemble
   - `NASA`：top-2 vote ensemble
4. Round 2 结束后再次判定：
   - `UAB` 两组都赢
   - `NASA combined` 稳定赢

### Round 3：只在前两轮失败时切到原始 Chronaris 公开化

如果 Round 2 后 `NASA combined` 仍未稳定超过两条 deep baseline，则新增一条公开版 Chronaris 深模型：

- 名称固定：`chronaris_public_fusion`

固定结构：

1. 两个公开模态各自投影到共享 `hidden_dim`
2. 各自经过轻量 masked temporal encoder
3. 使用现有 `Stage G causal fusion`
4. 使用 `pooled_with_residual` 风格的 fused representation
5. 接任务头：
   - `UAB` regression
   - `NASA` classification

固定搜索顺序：

1. 先在 `NASA combined` 上粗筛
2. 只保留前 `3` 组配置跑 full LOSO
3. 再回测 `UAB subjective`

默认执行设备：

- `device=cuda`
- 当前高成本筛选优先放到 GPU，不再继续叠加 CPU 线性搜索预算

## 3. 配置与接口增量

### public opt

`StageIPublicOptConfig` 增加：

- `feature_profile`
- `head_catalog`
- `train_balance_policy`
- `ensemble_policy`
- `winner_margin_policy`

`public_opt_summary.json` 增加：

- `feature_profile`
- `head_catalog`
- `train_balance_policy`
- `ensemble_policy`
- `winner_margin_vs_deep`
- `needs_deep_rerun`

### deep baseline

`StageIDeepBaselineConfig` 增加：

- `model_name=chronaris_public_fusion`
- `fusion_event_bias_weight`
- `fusion_lag_window_points`
- `fusion_normalize_states`

CLI 同步支持：

- `scripts/run_stage_i_public_opt.py`
- `scripts/run_stage_i_deep_baseline.py`
- `scripts/run_stage_i_deep_comparison.py`

## 4. 判定与重跑规则

只有在以下情况才重跑 `MulT / ContiFormer`：

1. `UAB`
   - 新 `public opt` 相对最好 deep baseline 的 `RMSE` 领先幅度 `< max(0.02, 1% relative)`
2. `NASA combined`
   - 新 `public opt` 相对最好 deep baseline 的 `macro-F1` 领先幅度 `< 0.005`

否则默认直接引用历史 `MulT / ContiFormer` full LOSO 对照，不再给 deep baseline 新预算。

## 5. 测试与实跑要求

最小测试：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_deep_pipeline
```

当前推荐实跑顺序：

1. `UAB public opt` Round 1
2. `NASA public opt` Round 1
3. 若未过 gate，执行 Round 2
4. 若仍未过 gate，再执行 `chronaris_public_fusion`

## 6. 本轮不做的事

1. 不新增新的第三方模型家族
2. 不回退修改已冻结的 `Stage I closure` 历史事实
3. 不把 `classical baseline` 重新拉回主 gate
4. 不改 `Stage H` export contract
