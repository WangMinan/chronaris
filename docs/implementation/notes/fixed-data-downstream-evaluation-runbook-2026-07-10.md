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
- 本机 WSL/RTX 4090 同时只运行一个正式 CUDA 训练进程；仿真与鼎新 GPU 队列串行。若独占 CUDA 后仍再次出现驱动级 `cudaErrorLaunchFailure`，保留 checkpoint 并允许只改变设备到 CPU 后恢复；checkpoint 必须记录 `training_device_history`，其他训练配置不得改变。仿真 Chronaris 使用同批实测更快的 CPU；鼎新 100 ms 合并输入的 Chronaris 使用实测更快的 GPU。
- 公共观测增强、pretext target 和错误时移固定在 CPU 构造；只把增强后的双流 batch 与 target tensor 送入训练设备。checkpoint 记录 `augmentation_device=cpu`，不得为了设备切换改变 augmentation realization。

### 1.8 鼎新 real-only 锁定主表

- 六方法输入统一先经过 100 ms 固定因果时间箱；缺少 `model_input_contract.json` 的旧 checkpoint 不允许恢复。
- 五个可训练方法使用三随机种子、三个留一视图主折和两个留一架次辅助折、最多 50 epoch、patience 8；任务目标与 outer-test 在 75 个 checkpoint 完整前保持关闭。
- 单折实测中，Chronaris GPU 为 40.07 秒/epoch，CPU 为 91.86 秒/epoch，因此五个可训练方法全部在唯一 GPU 队列串行。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_pretraining.py \
  --run-id 2026-07-12_dingxin-locked-pretraining-coalesced \
  --seed 17 --seed 29 --seed 43 \
  --max-epochs 50 --patience 8 --batch-size 32 \
  --baseline-device cuda --chronaris-device cuda --resume

/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_representations.py \
  --run-id 2026-07-12_dingxin-locked-representations-coalesced \
  --pretraining-run-id 2026-07-12_dingxin-locked-pretraining-coalesced \
  --baseline-device cuda --chronaris-device cuda --resume

/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_consumers.py \
  --run-id 2026-07-12_dingxin-locked-consumers-coalesced \
  --representation-run-id 2026-07-12_dingxin-locked-representations-coalesced \
  --minirocket-kernels 10000 --resume
```

### 1.9 端到端微调辅助表

- 必须等待仿真任务无关锁定训练、统一表示和冻结 consumer 三个 evidence manifest 均为 `completed` 后启动。
- 五个可训练编码器从各自任务无关 `best.pt` 初始化；朴素时间同步只更新相同容量任务头。
- train 负责拟合，validation 负责 epoch 早停，G2 held-out 只打开一次；结果写入 `end_to_end_finetuned_v1`，不得覆盖冻结表示目录。
- GPU 稳定时仅基线微调走唯一 CUDA 队列，Chronaris 默认 CPU；出现重复驱动故障时按 1.7 的设备迁移规则恢复。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_simulation_end_to_end_finetuning.py \
  --run-id 2026-07-12_simulation-end-to-end-finetuning \
  --seed 17 --seed 29 --seed 43 \
  --learning-rate 1e-4 --max-epochs 20 --patience 5 --batch-size 128 \
  --baseline-device cuda --chronaris-device cpu --resume
```

### 1.10 仿真预训练到鼎新无标签适配

- 五个可训练方法都从各自同 seed 的 G1 锁定 checkpoint 初始化；只复制同名同形状参数，鼎新 schema 相关输入层和重构层重新初始化。
- 六方法使用相同仿真额外数据预算；朴素时间同步仍只在鼎新训练折拟合无监督归一化与 PCA。
- 鼎新 inner-train/validation 上只运行公共自监督目标，任务标签和 outer-test 保持关闭；辅助轨道固定最多 20 epoch、patience 5。
- 表示和 consumer 必须使用独立 run_id，表示族固定为 `synthetic_pretrain_real_adapt_v1`，不得覆盖 real-only 主表。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_pretraining.py \
  --run-id 2026-07-12_dingxin-synthetic-pretrain-adapt-coalesced \
  --initialization-pretraining-run-id 2026-07-12_simulation-locked-pretraining \
  --seed 17 --seed 29 --seed 43 \
  --max-epochs 20 --patience 5 --batch-size 32 \
  --baseline-device cuda --chronaris-device cuda --resume

/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_representations.py \
  --run-id 2026-07-12_dingxin-synthetic-pretrain-adapt-representations-coalesced \
  --pretraining-run-id 2026-07-12_dingxin-synthetic-pretrain-adapt-coalesced \
  --baseline-device cuda --chronaris-device cuda --resume

/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_dingxin_locked_consumers.py \
  --run-id 2026-07-12_dingxin-synthetic-pretrain-adapt-consumers-coalesced \
  --representation-run-id 2026-07-12_dingxin-synthetic-pretrain-adapt-representations-coalesced \
  --minirocket-kernels 10000 --resume
```

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

正式锁定上游全部完成后运行：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evidence/build_downstream_application_pack.py \
  --run-id 2026-07-12_downstream-evidence-pack
```

该命令必须生成 7 幅中文图（包括一个鼎新代表时间线和一个仿真 oracle 复盘）、`figure_manifest.csv`、`evidence_matrix.csv`、预声明主指标表、`claim_boundary.md` 和 evidence manifest；随后逐图人工抽查，不以脚本成功代替可读性检查。

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
