# Stage F Closure - 2026-04-22

## 1. 结论

阶段 F 已完成，满足进入阶段 G 的开发条件。

依据：

1. 已实现组件级完整物理约束族，并保持阶段 E baseline 默认关闭物理约束。
2. 已使用真实 InfluxDB + MySQL 元数据完成 `E baseline` 与 `E+F(full)` 对比实跑。
3. `E baseline` 与 `E+F(full)` 的默认阈值模板评估均为 `PASS`。
4. 阶段 F 相关单测、pipeline 测试与全量 runtime 测试通过。

## 2. 收口实跑记录

主报告：

- `docs/reports/alignment-preview-stage-f-closure-2026-04-22.md`

核心配置：

- sortie：`20251005_四01_ACT-4_云_J20_22#01`
- physiology measurements：`eeg`、`spo2`
- vehicle measurement：`BUS6000019110020`
- input normalization：`zscore_train`
- physics family：`full`
- physics mode：`feature_first_with_latent_fallback`
- MySQL metadata：`loaded`，字段映射数 `96`
- seed：`20260421`

核心结果：

| 配置 | train total | validation total | test total | test physics total | threshold verdict |
| --- | ---: | ---: | ---: | ---: | :---: |
| `E baseline` | `1.559523` | `1.460789` | `1.561467` | `0.000000` | `PASS` |
| `E+F(full)` | `1.839854` | `1.672444` | `1.769787` | `1.539611` | `PASS` |

`E+F(full)` 测试集物理约束组件：

| component | value |
| --- | ---: |
| `vehicle_semantic` | `0.752882` |
| `vehicle_smoothness` | `0.046221` |
| `vehicle_envelope` | `0.000000` |
| `vehicle_latent` | `0.070477` |
| `physiology_smoothness` | `0.457301` |
| `physiology_envelope` | `0.000000` |
| `physiology_pairwise` | `0.042706` |
| `physiology_spo2_delta` | `0.000000` |
| `physiology_latent` | `0.170024` |

## 3. 实现边界

本轮完成的是阶段 F 在当前真实数据范围内的完整约束族：

- 飞机侧：RealBus 字段语义映射、语义运动学残差、通用平滑、训练分布包络、潜态 fallback。
- 生理侧：EEG 平滑、EEG 对称通道一致性、`spo2` 突变约束、训练分布包络、潜态 fallback。
- 无法由当前字段触发的组件保留为 `0.000000`，不伪造业务语义。

注意：

- 当前仍是单架次 overlap-focused preview 收口，不等价于阶段 D 批量数据集工程。
- 阶段 G 因果融合尚未启动。

## 4. 产物与入口

脚本入口：

- `CHRONARIS_MYSQL_HOST=127.0.0.1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_e_relative_preview.py --input-normalization-mode zscore_train --physics-constraint-family full --compare-with-physics-baseline --report-path docs/reports/alignment-preview-stage-f-closure-2026-04-22.md`

关键产物：

- 对比主报告：`docs/reports/alignment-preview-stage-f-closure-2026-04-22.md`
- baseline 细粒度资产：`docs/reports/assets/alignment-preview-stage-f-closure-2026-04-22-e-baseline/`
- F(full) 细粒度资产：`docs/reports/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full/`
- 诊断 JSON/CSV 与可视化图像：同上 assets 目录
- 模型 checkpoint：`docs/reports/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full/alignment_model_checkpoint.pt`

说明：顶层单配置子报告 Markdown 已清理，阶段 F 事实以本 closure 文档和对比主报告为准。

## 5. 测试闭环

执行命令：

- `CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_alignment_model_losses tests.test_alignment_pipeline`
- `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover -s tests -p 'test_*.py'`

结果：

- 定向测试：`Ran 19 tests ... OK`
- 全量测试：`Ran 71 tests ... OK`
