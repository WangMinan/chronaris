# 固定数据下游评估长程运行手册

日期：2026-07-10  
状态：与详细实施计划配套的运行、恢复和收口协议

主实施计划：

- [fixed-data-downstream-evaluation-2026-07-10.md](fixed-data-downstream-evaluation-2026-07-10.md)

## 1. 工作包 G：开发筛选与锁定训练

### 1.1 smoke run

范围：

- 六方法。
- 一个 G1 小数据 fold。
- 每方法候选 A、1 epoch。
- linear consumer。

目的只验证训练、导出、下游和报告闭环，不进入论文指标。

### 1.2 screen run

- seed 17。
- 每个深度方法四候选。
- G1 train/validation 完整。
- 鼎新主协议只在外层 train groups 内做 pretext 选择。
- 不运行 G2 locked test。

产物根：`docs/artifacts/runs/2026-07-10_application-benchmark-screen/`。

### 1.3 locked confirm

配置锁定后：

- seeds 17、29、43。
- 鼎新 leave-one-view-out 和辅助 leave-one-sortie-out。
- G1 -> G2 clean/canonical scenarios。
- frozen linear/MiniRocket/TCN 主结果。
- 六方法同预算 synthetic-to-real 辅助轨道。

产物根：`docs/artifacts/runs/2026-07-10_application-benchmark-confirm/`。

### 1.4 stress sweep

只使用 locked checkpoint：

- G2 单因素全部等级。
- mixed-severe。
- 不再调模型或 consumer。
- 输出 paired degradation curve 和机制误差。

### 1.5 消融

只对 locked Chronaris 跑：

- 无连续演化。
- 无物理。
- 无因果掩码。
- 单尺度 lag。

消融使用同 seed、相同 checkpoint 选择预算和相同 downstream consumer。

### 1.6 时间偏移与响应时延恢复

- 只比较朴素时间同步、MulT、ContiFormer 和 Chronaris 四种双流方法。
- 编码器锁定后导出 G1 六个成对观测场景的 train/validation 表示；此阶段不打开 oracle。
- downstream Ridge 只用 G1 train 拟合、G1 validation 在 `alpha={0.1,1,10,100}` 中选参。
- 目标固定为生理流相对航电流的绝对时钟偏移幅值，以及第一生理字段的真实响应时延。
- G2 的 35 个压力场景只用于最终评价，以 48 条潜在轨迹为配对单位，不允许反向调参。

### 1.7 运行恢复

每个 run：

- 原子更新 `progress.json`。
- 按 method/task/split/seed/fold 保存 status。
- best/last checkpoint 分开命名。
- `--resume` 跳过 hash 已匹配的 completed fold。
- 配置或输入 hash 改变时拒绝复用旧 fold。
- 本机 WSL/RTX 4090 同时只运行一个正式 CUDA 训练进程；仿真与鼎新 GPU 队列串行，避免驱动级 `cudaErrorLaunchFailure`。Chronaris 已确认 CPU 更快，继续使用 CPU。

## 2. 工作包 H：附录诊断与论文证据包

### 2.1 结构诊断附录

主实验锁定后再移植 ClaSP/STUMPY：

- 三条鼎新连续 view。
- G2 locked long trajectories。
- 只生成状态片段、motif/discord 和复盘图。
- 不生成综合 winner score，不参与候选选择。

### 2.2 论文图表

至少产出：

1. 六方法真实任务指标对比。
2. 双流相对最佳单流增益。
3. stress level–performance degradation 曲线。
4. 时间偏移/响应时延恢复误差。
5. Chronaris 四项组件消融。
6. 一个鼎新案例和一个仿真 oracle 案例。

图题、坐标轴、legend 和表头使用中文任务/指标名称，不显示内部候选 ID。

### 2.3 证据矩阵

新增下游证据包根：`docs/artifacts/runs/2026-07-10_downstream-evidence-pack/`。

将结果分为：

- 鼎新真实双流弱监督任务。
- 仿真真值机制与压力测试。
- UAB/NASA 公开数据适配。
- 结构诊断附录。

不直接覆盖 `2026-07-03_thesis-protocol-snapshot`；先生成候选矩阵并通过 review，之后再建立新的协议快照。

## 3. 每个正式 run 的固定文件

```text
data_manifest.json
split_manifest.json
representation_manifest.json
training_protocol.json
downstream_protocol.json
metric_long.csv
fold_metrics.csv
progress.json
run.log
resume_command.txt
claim_boundary.md
evidence_manifest.json
report.md
```

逐样本预测、dense bundle、checkpoint 和 batch-level log 留在忽略目录，只在 evidence manifest 中登记 hash。

## 4. 失败与停止规则

### 4.1 结构化 unavailable

以下情况只关闭对应证据，不伪造替代输入：

- 原始鼎新 snapshot 不可用。
- 机动标签字段元数据不足。
- 某 fold 训练类别不足。
- 某物理项缺少字段。
- optional 结构诊断依赖不可用。

### 4.2 训练失败

- OOM：按预先声明 batch size candidates 降级，不能改变模型 hidden dim。
- NaN：保存 failure packet，最多以相同配置重试一次；仍失败则该 fold unavailable。
- 结果不优：保留完整结果，不追加未计划 candidate。

### 4.3 长程任务停止点

只有下列情况可以把 goal 标记 blocked：

- 同一关键数据/环境阻塞连续三个 goal turn 无法绕过。
- 用户必须提供新的权限或改变研究决策。
- 存储/GPU 故障导致所有安全替代路径不可用。

单个 fold 失败、结果 mixed 或训练耗时长都不属于 goal blocker。

## 5. 文档同步节奏

每通过一个验收门：

1. 更新 `docs/STATE.md` 当前事实。
2. 更新 `docs/implementation/TASKS.md` 当前队列。
3. 新增/更新 `docs/artifacts/ARTIFACTS.md` run 入口。
4. 运行读者可见术语审计。
5. 记录测试、git diff 和 LFS 状态。

不得等全部实验结束后一次性补写状态。

## 6. 最终验证

```text
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall src scripts tests
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q
git diff --check
git lfs status
git lfs fsck
```

额外检查：

- Git 中不存在 raw snapshot、sequence bundle、checkpoint 或逐样本 dense predictions。
- 所有当前 docs 路径存在。
- 所有正式指标能从 evidence manifest 反查。
- 中文图件抽查通过，无文字重叠或内部黑话。
- reader-facing 文本不残留“私有/private”“代理/proxy”“T1/T2/T3”或裸阶段号；机器字段和历史路径除外。

## 7. 预定提交边界

按以下七个独立提交收敛：

1. 固定数据策略与详细规格。
2. 数据审计、任务 builder 和 split。
3. G1/G2 仿真器与审计。
4. 统一表示合同和六方法编码器。
5. 下游 consumer、指标和报告。
6. screen/confirm/stress/ablation 紧凑产物。
7. 论文证据包和状态文档闭环。

默认只做本地提交；push/PR 需用户另行明确授权。
