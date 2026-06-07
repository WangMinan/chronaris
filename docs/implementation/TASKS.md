# Chronaris 当前任务

更新时间：2026-06-07

## 文档定位

本文件是当前唯一主动执行入口。总路线、阶段结构、当前任务队列和默认工作方式统一维护在这里；历史计划与阶段笔记保留在 `notes/`。

## 总路线

1. 读取指定架次的人机多源数据及元信息。
2. 建立统一 schema、统一时间参考和统一样本组织。
3. 实现双流连续潜态建模。
4. 实现物理一致性约束时间对齐。
5. 实现因果掩码跨模态融合。
6. 输出标准化融合特征与中间态接口。
7. 面向典型任务开展对比、消融和案例验证。

## 阶段结构

- `Stage A/B/C`：已完成。
- `Stage D`：后置，保留为后续数据集工程化工作。
- `Stage E0/E/F/G(min)/H`：已完成并收口，作为历史基线与后续依赖。
- `Stage I`：
  - `Phase A/B/C`：已接上统一骨干、真实 weak-label 联合训练、private/thesis 分层资产和中期证据包。
  - `Phase D`：刚体运动物理约束补强。
  - `Phase E`：语义事件融合补强。
  - `Phase F`：runtime inference。

## 默认工作方式

每轮实现默认按下面顺序收敛：

1. 目标锁定。
2. 代码实现。
3. 测试闭环。
4. 文档回写。
5. 冗余清理。

运行 Python 脚本、测试、基准或阶段命令前，默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## 当前 P0：冻结 Phase C 工作区并提交

目标：把当前 Stage I thesis mainline Phase C 从“已测试但未提交的工作区”固化为可追溯主线。

必须处理：

- 复查 `git status --short` 中全部未提交项，区分三类改动：
  - Phase C 代码与测试。
  - 历史路径清理、导航合并和文档入口重写造成的文本刷新。
  - 与本阶段无关的临时改动。
- 保留并复查新增文件：
  - `src/chronaris/models/alignment/task_heads.py`
  - `src/chronaris/dataset/stage_i_real_task_builders.py`
  - `src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`
  - `tests/test_stage_i_multitask_train.py`
- 保留并复查相关修改：
  - `src/chronaris/models/alignment/losses.py`
  - `src/chronaris/models/alignment/__init__.py`
  - `src/chronaris/dataset/__init__.py`
  - `src/chronaris/pipelines/__init__.py`
  - `src/chronaris/pipelines/stage_i/stage_i_private_benchmark.py`
  - `src/chronaris/pipelines/stage_i/stage_i_private_benchmark_data.py`
  - `src/chronaris/pipelines/stage_i/stage_i_private_benchmark_models.py`
  - `tests/test_alignment_model_losses.py`
  - `tests/test_stage_i_private_optimization.py`
- 修正或记录新增代码默认输出路径，统一落到 `docs/artifacts/assets/...`。
- 补跑并保留结果：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_multitask_train tests.test_stage_i_private_optimization tests.test_alignment_model_losses
```

当前最近一次结果：`Ran 10 tests in 5.827s`，`OK (skipped=2)`。

退出条件：

- Phase C 范围清楚。
- 关键测试通过。
- 提交前文档事实与代码事实一致。

## 已完成 P1：补真实资产上的 Stage I multitask 联合训练证据

结果：Phase C 已经不再停留在合成烟测，而是在真实 Stage H all-window clean 资产上形成了可引用证据。

前置编码任务：

- 已新增 `scripts/run_stage_i_multitask_train.py`。
- 已复用 `collect_stage_i_backbone_samples()` 路线并补 `view_id::raw_window_sample_id` sample-id contract，生成真实 `E0ExperimentSample`。
- 已复用 `load_aligned_private_records()` 与 `build_stage_i_real_task_payload()` 构造 weak-label task entries。
- 已修正 workload proxy 的归一化尺度，并把 event replay pair 调整为优先近邻配对，避免验证/测试分区丢失 retrieval 监督。
- 默认输出已切到 `docs/artifacts/assets/stage_i_multitask/<run_id>/`。

建议真实资产入口：

- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
- `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`

本轮输出：

- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_checkpoint.pt`
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
- `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl`
- `docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`

报告边界：

- 已明确写成 `thesis weak-label evidence`。
- 已保持“不写成人工真值任务”的边界。
- 当前产物按 `mainline closure evidence` 使用，不宣称人工真值最优。

## 已完成 P2：刷新 private benchmark 分层资产

结果：已用当前 Phase C 代码重跑 private benchmark，使历史 `T1/T2/T3` proxy evidence 与 thesis weak-label evidence 在资产层明确拆开。

建议命令形态：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/run_stage_i_private_benchmark.py \
  --e-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json \
  --f-run-manifest docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json \
  --output-root docs/artifacts/assets/stage_i_private \
  --report-root docs/artifacts \
  --enable-optimized-chronaris \
  --export-optimized-package
```

本轮验收：

- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_summary.json`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_manifest.jsonl`。
- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_summary.json`。
- `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence` 和 `evidence_layers.thesis_task_evidence`。
- CLI 已打印 thesis task manifest / summary 路径。

## 当前 P3：补 Stage F 刚体运动物理约束

目标：把已有弱物理约束整理成可选择、可诊断、可单测的 `rigid_body` physics family。

建议落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `src/chronaris/models/alignment/physics_residuals.py`
- `src/chronaris/models/alignment/physics.py`
- `src/chronaris/models/alignment/losses.py`
- `tests/test_alignment_model_losses.py`

验收：

- 旧 `full` / weak physics 配置不回归。
- 新 `rigid_body` family 可被显式选择。
- 分项损失能说明哪些刚体运动残差启用、哪些因字段缺失 fallback。

## 当前 P4：补 Stage G 语义事件融合

目标：从 `G(min)` 时间步注意力升级到语义查询、事件 token 与事件级归因。

建议落点：

- `src/chronaris/models/fusion/semantic_event.py`
- `src/chronaris/models/fusion/causal.py`
- `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
- `src/chronaris/pipelines/stage_i/stage_i_support.py`
- `tests/test_stage_i_support.py`

验收：

- 输出 event token。
- 输出 query-to-event attention。
- 输出事件级归因摘要。
- support 报告能区分“时间步注意力”和“事件级归因”。

## 当前 P5：补 runtime inference

目标：新增真正的离线回放或准实时推理入口，避免继续把 `runtime_demo.py` 误当实时推理引擎。

建议落点：

- `src/chronaris/dataset/streaming_windows.py`
- `src/chronaris/serving/runtime_inference.py`
- `scripts/run_stage_i_runtime_inference.py`
- `tests/test_runtime_inference.py`

验收：

- mock stream 或本地回放流可以增量产出窗口。
- 可以加载 checkpoint 做风险/负荷/事件预测。
- 可以输出 attention / event attribution 解释。

## 中期前最小收敛顺序

1. 提交 P0。
2. 直接引用最新 Phase C 真实联合训练证据与 private benchmark 分层资产。
3. 直接引用最新中期证据包：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`。
4. 中期后把主要研发投入切到 P3 / P4 / P5。

## 中期前降级处理

下面几类工作不是永久排除，而是中期答辩前不抢占主线投入；后续毕业论文整理和系统封装阶段，可以按边界清楚、证据分层、可复现的方式适当纳入。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前不再扩搜；论文封装阶段可作为公开 adapter baseline / calibration baseline 的补充材料。
- NASA/UAB 公开数据适配器结果：中期前不改写成论文双流本体闭环；论文中可作为 public adapter evidence，用于说明方法在公开代理数据上的迁移与校准边界。
- `chronaris_opt` 与 `T1/T2/T3`：中期前不写成人工真值 thesis task fully closed；论文中可作为 private proxy benchmark evidence，用于支撑表示学习、对齐和因果融合增益。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。

## 历史计划入口

- [notes/coding-roadmap.md](notes/coding-roadmap.md)
- [notes/stage-i-thesis-mainline-roadmap-2026-05-15.md](notes/stage-i-thesis-mainline-roadmap-2026-05-15.md)
- [notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md](notes/stage-i-thesis-mainline-coding-plan-2026-05-15.md)
- [notes/thesis-coding-gap.md](notes/thesis-coding-gap.md)
- [notes/iteration-playbook.md](notes/iteration-playbook.md)
