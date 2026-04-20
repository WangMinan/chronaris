# Implementation Summary - 2026-04-20

## 1. 本轮目标

本轮按“先文档整理，再阶段 E 编码与测试”推进，目标是：

1. 清理并补全文档索引，减少跨会话定位成本
2. 针对上一轮 `vehicle reconstruction loss` 量级主导问题，完成一轮可验证的代码修正
3. 在当前机器可用环境下完成最小回归验证并记录限制
4. 补齐样本级中间态诊断能力并固化实验产物格式

## 2. Round 1：docs 整理记录

本轮文档整理完成了以下内容：

1. 更新 `docs/README.md` 为中文索引并统一为相对路径链接
2. 合并 `2026-04-15` 与 `2026-04-17` 两份历史 summary 为单一历史文档
3. 补齐 `reports` 分类中的 `alignment-preview-20251005-act4-j20-22.md` 入口
4. 补齐“选题报告与基金申请书”原始文档索引

本轮整理后，`docs` 目录可直接作为阶段 E 交接入口使用，无需依赖本地盘符路径。

## 3. Round 2：阶段 E 编码与测试记录

### 3.1 本轮编码变更

围绕“vehicle 重构损失量级过大”的问题，本轮新增了最小尺度归一化能力：

1. `src/chronaris/models/alignment/losses.py`
   - 为重构损失新增 `mode` 参数，支持 `mse` 与 `relative_mse`
   - `relative_mse` 以目标值均方作为尺度进行归一化，降低跨流量纲差异
   - `build_stage_e_objective` 新增 `reconstruction_mode` 与 `reconstruction_scale_epsilon` 参数
   - 将 `prototype` 依赖改为 `TYPE_CHECKING`，消除不必要运行时强耦合
2. `src/chronaris/pipelines/alignment_preview.py`
   - `AlignmentPreviewConfig` 新增：
     - `reconstruction_loss_mode`（默认 `relative_mse`）
     - `reconstruction_scale_epsilon`
   - 训练目标构建时透传上述参数

### 3.2 测试与验证

新增测试：

- `tests/test_alignment_loss_scaling.py`
  - 验证 `relative_mse` 能显著削弱跨流尺度差异对重构损失的主导效应

补充测试：

- `tests/test_alignment_losses.py`
  - 增加 `relative_mse` 场景下的损失行为断言

本机执行结果：

1. `python -m unittest tests.test_alignment_loss_scaling tests.test_alignment_losses tests.test_alignment_preview_pipeline tests.test_alignment_experiment_pipeline`
2. 结果：`Ran 4 tests in 0.002s, OK (skipped=3)`
3. 说明：当时本机缺少 `torchdiffeq`，三组 `torch runtime` 测试仍按既有策略跳过

## 4. Round 3：服务器 WSL 环境实跑记录

### 4.1 运行时与依赖确认

在服务器 WSL 中已完成：

1. `chronaris` conda 环境补装 `torchdiffeq 0.2.5`
2. `CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1` 下的 Stage E 相关测试全通过

验证命令：

- `conda run -n chronaris env CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 python -m unittest tests.test_alignment_loss_scaling tests.test_alignment_losses tests.test_alignment_preview_pipeline tests.test_alignment_experiment_pipeline`

结果：

- `Ran 8 tests in 1.296s, OK`

### 4.2 relative-mse 真实回归

本轮新增脚本：

- `scripts/run_stage_e_relative_preview.py`

脚本作用：

1. 复用 overlap-focused 配置构建 `E0` 样本
2. 以 `relative_mse` 作为重构目标运行 Stage E preview 训练
3. 导出报告并输出 JSON 摘要

本轮实跑输出：

- 报告文件：`docs/reports/alignment-preview-20251005-act4-j20-22-relative-mse-2026-04-20.md`
- runtime device：`cuda`
- sample count：`25`
- split：`15 / 5 / 5`

关键指标（final）：

1. train total：`2.032757`
2. validation total：`2.027239`
3. test total：`2.027239`
4. train alignment：`0.032758`
5. validation/test alignment：`0.027244`

中间态导出：

1. partition：`test`
2. exported sample count：`3`
3. reference point count：`16`

### 4.3 结果解释

与上一轮 `vehicle reconstruction ≈ 1e26` 的回归结果相比，本轮结论是：

1. `relative_mse` 已把跨流尺度差异对总损失的主导显著压平
2. 当前 `physiology / vehicle reconstruction` 都稳定在约 `1.0`
3. 总损失回到可比较量级，后续可以更清晰观察 alignment 与中间态变化

## 5. Round 4：可视化产出补充

为支持“实跑时同步产图”，本轮补充了脚本可视化能力：

1. `scripts/run_stage_e_relative_preview.py` 默认在每次实验后生成图片到：
   - `docs/reports/assets/<report-stem>/`
2. 生成图片后，会自动将图片链接追加到报告末尾 `Visual Artifacts` 区域
3. 当前默认输出 4 张图：
   - `train_validation_total_loss.png`
   - `train_validation_alignment_loss.png`
   - `reconstruction_stream_loss.png`
   - `reference_projection_cosine.png`

本轮已验证：上述图片已随同报告一并生成，并可在 Markdown 中直接渲染查看。

## 6. Round 5：样本级中间态诊断编码

本轮继续按 roadmap 第 1 优先级推进，新增了样本级诊断模块：

1. `src/chronaris/evaluation/alignment_diagnostics.py`
   - 输出 sample-level 的投影诊断指标：
     - mean/min/max projection cosine
     - projection L2 gap
     - projection L2 ratio（vehicle/physiology）
   - 支持渲染 Markdown 诊断区块
2. `scripts/run_stage_e_relative_preview.py`
   - 自动将诊断区块插入实验报告
   - 自动输出诊断产物：
     - `projection_diagnostics_summary.json`
     - `projection_diagnostics_samples.csv`
3. `tests/test_alignment_diagnostics.py`
   - 覆盖诊断摘要计算与 Markdown 渲染行为

验证结果：

1. 本机：`python -m unittest tests.test_alignment_diagnostics` 通过
2. 服务器环境：
   - `CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1` 相关测试共 `10` 项通过
   - 实跑后报告已包含 `Sample-Level Projection Diagnostics` 区块
   - 诊断 JSON/CSV 已写入对应 `assets` 目录

## 7. 当前阶段 E 状态更新

相对于上一轮，本轮完成后阶段 E 的结论更新为：

1. 已完成“损失尺度归一化 + 真实回归验证”闭环
2. 服务器环境下 runtime 测试与真实训练链路均已跑通
3. 实验流程已经具备“数值指标 + 诊断指标 + 图像产出”三通道输出能力
4. 下一轮应转向输入归一化评估与跨样本差异性诊断

## 8. 相关文档

- [coding-roadmap.md](coding-roadmap.md)
- [stage-e-execution-plan.md](stage-e-execution-plan.md)
- [implementation-summary-history-2026-04-15-to-2026-04-17.md](implementation-summary-history-2026-04-15-to-2026-04-17.md)
- [alignment-preview-20251005-act4-j20-22.md](../reports/alignment-preview-20251005-act4-j20-22.md)
- [alignment-preview-20251005-act4-j20-22-relative-mse-2026-04-20.md](../reports/alignment-preview-20251005-act4-j20-22-relative-mse-2026-04-20.md)
