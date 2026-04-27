# Stage G Closure - 2026-04-22

## 1. 结论

阶段 G 最小原型已完成，满足进入阶段 H 的开发条件。

依据：

1. 已实现最小非对称因果融合模块，遵守选题报告中“当前生理状态作为 Query、历史航电事件作为 Key/Value、非对称因果掩码切断逆向信息流”的设计约束。
2. 已使用真实 InfluxDB + MySQL 元数据完成 `F baseline` 与 `F+G(min)` 对比实跑。
3. `F baseline` 与 `F+G(min)` 的默认阈值模板评估均为 `PASS`。
4. 阶段 G 相关单测、pipeline 测试与全量 runtime 测试通过。

## 2. 收口实跑记录

主报告：

- `docs/reports/alignment-preview-stage-g-min-closure-2026-04-22.md`

核心配置：

- sortie：`20251005_四01_ACT-4_云_J20_22#01`
- physiology measurements：`eeg`、`spo2`
- vehicle measurement：`BUS6000019110020`
- input normalization：`zscore_train`
- physics family：`full`
- physics mode：`feature_first_with_latent_fallback`
- causal fusion state source：`hidden`
- MySQL metadata：`loaded`，字段映射数 `96`
- seed：`20260421`

核心结果：

| 配置 | train total | validation total | test total | test physics total | threshold verdict |
| --- | ---: | ---: | ---: | ---: | :---: |
| `F baseline` | `1.839854` | `1.672444` | `1.769787` | `1.539611` | `PASS` |
| `F+G(min)` | `1.839854` | `1.672444` | `1.769787` | `1.539611` | `PASS` |

阶段 G 注意力与事件贡献摘要：

| metric | value |
| --- | ---: |
| exported fusion samples | `3` |
| reference point count | `16` |
| state dim | `32` |
| fused dim | `96` |
| mean attention entropy | `0.931446` |
| mean max attention | `0.225623` |
| mean causal option count | `8.500000` |
| mean top event score | `1.000000` |
| mean top contribution score | `2.609973` |

## 3. 实现边界

本轮完成的是阶段 G 在当前真实数据范围内的最小因果融合原型：

- 飞机侧：从阶段 F 对齐后的连续潜态参考网格中抽取相邻变化量作为事件级 salience。
- 融合侧：以生理参考状态为 Query，以历史飞机参考状态为 Key/Value，使用时间因果掩码禁止未来航电状态泄露。
- 输出侧：导出融合表示、注意力权重、车辆事件分数、样本级最高贡献事件与注意力热力图。

注意：

- 当前仍是单架次 overlap-focused preview 收口，不等价于多架次泛化验证。
- 当前 G(min) 是无下游标签的最小融合与解释接口；阶段 H v1 已在后续完成标准化特征导出入口，但阶段 I 下游任务验证尚未开始。
- 当前 G(min) 没有宣称完成完整多头语义查询训练层；本轮只先固化与选题报告一致的单向因果计算图、历史事件 Key/Value 约束和解释产物。
- 后续消融应至少覆盖 `E`、`E+F(full)`、`E+F(full)+G(min)`、`E+G(no physics)`、`F+G(no causal mask)`。

## 4. 产物与入口

脚本入口：

- `CHRONARIS_MYSQL_HOST=127.0.0.1 python scripts/run_stage_e_relative_preview.py --input-normalization-mode zscore_train --physics-constraint-family full --compare-with-causal-fusion-baseline --report-path docs/reports/alignment-preview-stage-g-min-closure-2026-04-22.md`

关键产物：

- 对比主报告：`docs/reports/alignment-preview-stage-g-min-closure-2026-04-22.md`
- F baseline 细粒度资产：`docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-f-baseline/`
- G(min) 细粒度资产：`docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/`
- 诊断 JSON/CSV：`docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/`
- 因果注意力热力图：`docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/causal_attention_heatmap.png`
- 模型 checkpoint：`docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/alignment_model_checkpoint.pt`

说明：顶层单配置子报告 Markdown 已清理，阶段 G(min) 事实以本 closure 文档和对比主报告为准。

## 5. 测试闭环

执行命令：

- `CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 python -m unittest tests.test_alignment_pipeline`
- `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 python -m unittest discover -s tests -p 'test_*.py'`

结果：

- 定向测试：`Ran 9 tests ... OK`
- 全量测试：`Ran 73 tests ... OK`
