# 阶段 5 显卡异常诊断与恢复

**原冻结主模型训练已恢复，并通过真实更新与检查点验收。** 新总控 `2464764`、训练子进程 `2464772` 于 10 月 4 日恢复原失败单元；13:43:54 已确认推进至 28/356 次预训练更新，第 25 次检查点可独立读取并通过完整训练状态散列核验。已完成量仍为 167/279 个模型单元、281/477 条表示路线，当前在训单元不提前计作完成。CLARE 生理负荷数据完成 135/135，CogPilot 虚拟飞行数据完成 32/120，鼎新完成 0/24。后台继续运行，按用户要求结束会话并等待手动唤醒。

## 诊断与处理依据

**训练错误与宿主机显卡异常发生在同一秒。** Windows 的 NVIDIA 驱动于 16:00:45.967 和 16:00:45.972 记录事件 13、153，内容分别为 `Graphics FECS Exception: Logging error 0x2` 与 `Error occurred on GPUID: c200`。训练于 16:00:46 写出 `CUDA error: unspecified launch failure`。首次调用栈位于负样本前向的观测整理，后续随机状态恢复错误属于异常清理阶段。原始记录见[显卡事件](host_driver_events.json)与[失败日志](failed_attempt.log)。

**当前证据支持先恢复显卡执行进程，并通过真实训练重试检验。** 原冻结源码的 CUDA（统一计算设备架构）图输出、梯度、梯度累积和状态恢复，以及队列显式重试测试共 11 项通过，耗时 51.11 秒。同类梯度流警告在测试中出现，但测试正常通过；警告本身不能确定此次退出的原因。没有修改计算源码、驱动设置、损失、批量、种子或验证预算。尚不能仅凭宿主机事件确定最终触发源，也不将一次成功重试解释为长期显卡故障已根治。

本轮使用现有入口 `--retry-failed`，同源码重试保留在原根目录 `/mnt/e/chronaris-v4-results/2026-10-02-stage5-performance-recovery`。冻结运行副本为 `/home/wangminan/projects/chronaris-runtime-stage5-performance-20261002`，源码提交为 `a80289e88c27a214ed333aa06abe90ba5a546089`，工作流散列为 `361107d7d0e11468e0e4ff861df7e80662bb91efd1bc48913c4919cb9328157f`。未知错误仍会停止队列；本轮没有增加自动无限重试。

## 进度与成本保留

失败单元为 `stage5_formal_unit__167`，即 CogPilot、第二折（`fold_index=1`）、种子 29 的 Chronaris 模型。预训练仅完成 4/356 次参数更新，实际 `last.pt` 尚不存在，因此该单元从固定种子重新开始；其后任务引导训练尚未执行。已完成的 167 个单元直接保留，不重跑也不改写研究结论。

原失败尝试耗时 240.078 秒，单独保留在[尝试成本](failed_attempt_cost.json)，不会随重试消失，也不计作完成单元。原进度、错误与配方位于 [失败单元](failed_unit_167/cogpilot/progress.json)；较大的原数据合同已独立归档并记录[路径与散列](failed_data_contract_preserved.json)。原队列状态和配置分别见[状态快照](before_pipeline_state.json)、[配置快照](before_pipeline_config.json)。

## 验收入口

**恢复前 3,457 份文件的散列全部一致，恢复后完成收据及配置保持原样。** 新尝试已经越过原第 4 次更新的失败位置，历史失败仍为一条，未新增失败。第 25 次检查点保存了 400 个样本、100 个微批次的数据游标，以及模型、优化器与随机状态；其规范训练状态散列与保存值一致，种子 29、微批量 4、有效批量 16、356 次预算和显卡图执行均与原配方一致。检查点快照独立保存，后续 `last.pt` 更新不覆盖这份验收证据。

宿主机事件复查截至 13:45:10，没有查到本次重启以来新增的 NVIDIA 驱动事件。该观察与训练推进共同支撑当前恢复成功；最初故障的最终触发因素仍未确认。

- [测试结果](pytest.xml)与[完整测试日志](pytest.log)：11 通过、1 条警告。
- [恢复前产物核验](recovery_preflight.json)：完成收据、模型、下游文件、来源输入与冻结源码。
- [启动命令及新进程](launch.json)、[实际更新和检查点核验](startup_check.json)、[启动期间逐次观察](startup_progress.jsonl)。
- [恢复后队列快照](pipeline_state_snapshot.json)、[新尝试日志快照](retry_attempt_snapshot.log)、[显卡现场](gpu_snapshot.txt)、[重启后宿主机事件查询](host_driver_events_after_retry.json)。
- [复核过程](../../../review/stage/thesis-v4/stage5-cuda-retry-20261004/README.md)：诊断顺序、证据判断及恢复门。

当前恢复范围结束于主模型训练与模型文件冻结。外层评价继续保持关闭。
