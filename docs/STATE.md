# Chronaris 当前状态

更新时间：2026-06-07

## 一句话状态

项目已经具备中期答辩所需的历史实验资产和 Stage I 论文主线重构雏形；当前需要把工作区中的 Phase C 代码与文档固化，并补一条真实资产上的联合训练证据，然后再推进 Stage F 刚体运动物理约束补强、Stage G 语义事件融合补强和 runtime inference。

## 当前阶段

- 阶段 A/B/C：已完成。
- 阶段 E0：已完成 preview 路径。
- 阶段 E/F/G(min)：已完成真实链路收口，作为历史基线保留。
- 阶段 H：已完成标准化特征导出收口，`validation` profile 可稳定导出 3 个双流 view。
- 阶段 I 历史公开 benchmark：`Phase 0/1/2/3` 已完成并收口。
- 当前论文主线：Stage I thesis mainline `Phase A/B/C` 首轮代码收敛已完成；`Phase C` 仍处于当前工作区未提交状态。

## 当前主线事实

- 当前鼎新私有任务验证主线仍是 `chronaris_opt`，但它属于 `private proxy benchmark / proxy evidence`。
- 当前公开支撑线为 `public opt closed`，但 `UAB robust-prior adapter / target_prior_median` 只能写成 `public adapter / calibration evidence`，不能写成双流连续对齐或因果融合模块本体的直接胜利。
- 当前公开第二模态应写成 `context proxy / public adapter evidence`，不是论文严格意义上的真实航电流。
- `T1/T2/T3` 是私有代理任务；`risk_proxy / workload_proxy / event_replay_tag` 是 thesis weak-label task builder，不等价于人工真值任务。
- `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view。

## 当前关键入口

- 当前执行计划：[implementation/PLAN.md](implementation/PLAN.md)
- 当前任务队列：[implementation/TASKS.md](implementation/TASKS.md)
- 论文需求入口：[requirements/SPEC.md](requirements/SPEC.md)
- 产物索引：[artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)
- 中期前目标笔记：[implementation/notes/midterm-goal-2026-06-07.md](implementation/notes/midterm-goal-2026-06-07.md)

## 当前最值得推进的工作

1. 冻结当前 Stage I thesis mainline Phase C 工作区。
2. 基于真实 Stage H / private 资产补一条 C 阶段联合训练证据。
3. 刷新中期证据包。
4. 继续补 Stage F `rigid-body physics`。
5. 继续补 Stage G `semantic event fusion`。
6. 最后补 runtime inference。

## 当前不建议做的事

- 不继续扩大 CPU-heavy `sklearn` 或 UAB torch 候选搜索。
- 不把 NASA/UAB 公开数据适配器结果改写成论文双流本体闭环。
- 不把 `chronaris_opt` 或 `T1/T2/T3` 写成人工真值 thesis task fully closed。
- 不重写上游接收器、入库链路或原始大文件入仓。
