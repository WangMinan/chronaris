# Chronaris 当前状态

更新时间：2026-06-07

## 一句话状态

项目已经具备中期答辩可用的历史实验资产与最新主线补充证据：真实链路 `Stage E/F/G(min)/H`、Stage I 历史公开 benchmark、`chronaris_opt` 私有代理证据、public adapter 支撑线、Phase C 真实 Stage H multitask 联合训练证据和重建后的中期证据包都已形成。当前剩余工作已基本收敛到两个方向：一是把 Phase C 工作区与新增产物选择性提交；二是把 `Stage F rigid_body / Stage G semantic event fusion / runtime inference` 作为中期后的远期实现。

## 当前阶段

- 阶段 A/B/C：已完成。
- 阶段 E0：已完成 preview 路径。
- 阶段 E/F/G(min)：已完成真实链路收口，作为历史基线保留。
- 阶段 H：已完成标准化特征导出收口，`validation` profile 可稳定导出 3 个双流 view。
- 阶段 I 历史公开 benchmark：`Phase 0/1/2/3` 已完成并收口。
- 阶段 I thesis mainline：
  - `Phase A/B` 已经进入 git 历史，主线边界校准、统一骨干训练入口、checkpoint inference export contract 已具备。
  - `Phase C` 当前已在工作区完成真实资产闭环，内容包括统一任务头、任务监督损失、因果正则接入、`risk_proxy / workload_proxy / event_replay_tag` weak-label thesis task builder、`stage_i_multitask_train` 联合训练入口，以及 private benchmark 中 `proxy_evidence / thesis_task_evidence` 分层。真实产物已补齐，但仍未提交到 git 历史。
  - `Phase D/E/F` 尚未实现，分别对应刚体运动物理约束补强、语义事件融合补强和 runtime inference。

## Git 与工作区核对

- 当前 `HEAD=f0b58fc`，提交信息为 `restructure docs for ai coding`，主要是把事实源整理到 `docs/implementation`、`docs/artifacts`、`docs/requirements`，并保留 `docs/planning`、`docs/reports` 等兼容入口。
- 上一个关键实现提交 `2055dec` 覆盖 Stage I thesis mainline `Phase A/B`：public adapter/proxy 边界、backbone train、Stage H checkpoint inference contract 和相关测试。
- 当前工作区仍有未提交改动：
  - Phase C 代码与测试：`src/chronaris/models/alignment/task_heads.py`、`src/chronaris/dataset/stage_i_real_task_builders.py`、`src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`、`tests/test_stage_i_multitask_train.py`，以及相关 `__init__`、loss、private benchmark 和测试改动。
  - Phase C 新增 runner 与真实产物：`scripts/run_stage_i_multitask_train.py`、`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/`、`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/`、`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/`，以及对应 Markdown 报告。
  - 大量历史报告与 notes 中的路径文字已从 `docs/reports/...` 刷到 `docs/artifacts/...`，提交前需要和 Phase C 代码一起复查范围。
- 本轮新增的关键产物已经落盘：
  - Phase C 真实联合训练：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json`
  - thesis weak-label 报告：`docs/artifacts/stage_i/thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
  - private benchmark 分层资产：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json`
  - 最新中期证据包：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`

## 当前主线事实

- 当前鼎新私有任务验证主线仍是 `chronaris_opt`，但它属于 `private proxy benchmark / proxy evidence`。
- 当前公开支撑线为 `public opt closed`，但 `UAB robust-prior adapter / target_prior_median` 只能写成 `public adapter / calibration evidence`，不能写成双流连续对齐或因果融合模块本体的直接胜利。
- 当前公开第二模态应写成 `context proxy / public adapter evidence`，不是论文严格意义上的真实航电流。
- `T1/T2/T3` 是私有代理任务；`risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task builder，不等价于人工真值任务。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view。

## 编码层面还需要做什么

1. 固化当前 Phase C 工作区：复查未提交代码、路径迁移文档、真实产物和中期证据包改动，确认没有混入无关修改后选择性提交。
2. 补 Stage F 刚体运动物理约束：从现有 weak/full physics family 前进到可选择、可诊断、可单测的 `rigid_body` family。
3. 补 Stage G 语义事件融合：从时间步注意力升级到 `SemanticQueryBank / EventTokenExtractor / CausalEventFusion` 和事件级归因。
4. 补 runtime inference：不要把 `runtime_demo.py` 写成实时推理引擎，需要新增流式或准流式窗口缓存、checkpoint 推理和解释输出入口。

## 实验层面还需要做什么

1. 已完成 Phase C 真实联合训练证据：
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
   - 输入：`docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
   - 输出：`docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/`
2. 已完成 private benchmark 分层资产刷新：
   - 输出：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/`
   - `private_benchmark_summary.json` 已包含 `evidence_layers.proxy_evidence / thesis_task_evidence`
3. 已完成中期证据包重建：
   - 输出：`docs/artifacts/assets/stage_i_midterm/20260607T-stage-i-midterm-r3/`
   - 报告：`docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md`
4. 后续实验优先级转为：
   - 刚体运动物理约束 smoke / ablation
   - 语义事件融合 smoke / support
   - runtime inference 闭环

## 当前关键入口

- 当前执行计划：[implementation/PLAN.md](implementation/PLAN.md)
- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 论文需求入口：[requirements/SPEC.md](requirements/SPEC.md)
- 产物索引：[artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)
- 中期前目标笔记：[implementation/notes/midterm-goal-2026-06-07.md](implementation/notes/midterm-goal-2026-06-07.md)

## 本轮验证

使用指定 conda 解释器执行：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_multitask_train tests.test_stage_i_private_optimization tests.test_alignment_model_losses
```

结果：`Ran 10 tests in 5.827s`，`OK (skipped=2)`。`ConstantInputWarning` 来自合成样本常量输入的 Spearman 计算，不改变当前合约判断。

## 中期前降级处理

下面几类工作不是永久排除，而是中期答辩前不抢占主线投入；后续毕业论文整理和系统封装阶段，可以按边界清楚、证据分层、可复现的方式适当纳入。

- CPU-heavy `sklearn` 或 UAB torch 候选搜索：中期前不再扩搜；论文封装阶段可作为公开 adapter baseline / calibration baseline 的补充材料。
- NASA/UAB 公开数据适配器结果：中期前不改写成论文双流本体闭环；论文中可作为 public adapter evidence，用于说明方法在公开代理数据上的迁移与校准边界。
- `chronaris_opt` 与 `T1/T2/T3`：中期前不写成人工真值 thesis task fully closed；论文中可作为 private proxy benchmark evidence，用于支撑表示学习、对齐和因果融合增益。
- 上游接收器、入库链路和原始大文件入仓：中期前不重建；论文系统封装时可说明现有 MySQL / InfluxDB 接入边界，必要时补轻量接口说明或部署文档。
