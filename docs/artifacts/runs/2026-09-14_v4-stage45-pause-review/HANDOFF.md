# 手动唤醒交接：阶段 4.5 暂停复核

2026-09-14 17:54，本轮完整来源与实际恢复入口核验完成，原核验 PID 631521 已退出，无需再次启动。按用户最新指令结束会话，完整回归和长期实验尚未启动。

已完成：原总控 445416、子进程 453545 停止；第 75 次落盘状态完整保留。相同状态的普通/图执行两次更新通过原容差，核心计算约 9.13 倍提速，独立进程恢复逐值一致。图预训练到普通微调的短入口 4 项通过。完整来源核验检查旧运行 1431 文件及 7 份合同；真实共同入口从 75 续跑 76、77，三组模型状态与图执行对照逐值一致，随后按诊断上限主动结束。此处不是完整测试回归，也不是阶段 4.5 候选结果。

关键输出均在本目录：entry_validation.log、entry_validation.json、entry_validation/recovery.json、graph_comparison.json、graph_resume_comparison.json、recovery_evidence.json、profile_summary.json。大型 trace.json 位于 torch_profile/。旧运行和检查点保持原样。

py-spy 已安装，直接附加被系统权限拒绝，sudo -n 要求密码；两个尝试日志分别为 entry-py-spy.txt、entry-py-spy-sudo.txt。没有为剖析重跑已完成核验，已有 PyTorch 算子剖析保留。

下次用户手动唤醒后：先核验 Git 与落盘结果；完成可选第 50 次能力快照诊断（probe_snapshot.py 尚未运行），完成全部代码的完整 CUDA 回归（v4_configuration_freeze.validate_current_cuda），再整理文档、提交代码并冻结新源码副本。最终恢复根建议 /mnt/e/chronaris-v4-results/2026-09-14-stage45-recovery，使用 stage45_resume_parent 与 recovery_evidence.json，stage45-budget-hours 0；不得直接使用工程诊断目录作长期运行根。启动后只做一次现场核验再结束会话。保持既定有限矩阵、原阈值、原数据边界和阶段 5 关闭。

当前分支 codex/thesis-v4-recovery-20260905，本轮修改尚未提交，未运行完整 CUDA 回归。保留当前工作区，不丢弃变更。


## 2026-09-14 17:58 完整回归启动

用户手动唤醒后已授权启动完整 CUDA 回归。总控 PID 635589，独立会话运行；输出目录 `/mnt/e/chronaris-v4-results/2026-09-14-stage45-recovery/cuda_validation/attempt_1`，测试输出为 `pytest.log`，结束时生成 `pytest.xml` 和 `validation_receipt.json`。只执行完整测试套件，完成后自行退出，不接续长期实验。源码与测试指纹已写入 `launch.json`；运行中不修改源码和测试。下次唤醒先检查凭证状态、退出结果和源码指纹，再决定阶段 4.5 后续执行。


## 完整回归验收与长期恢复授权

完整显卡回归 691 项通过、22 项跳过、0 失败，耗时 305.28 秒；凭证与当前源码、测试集、日志散列一致。用户已授权长期恢复，采用独立冻结源码和新运行根，从第 75 次状态继续固定阶段 4.5 矩阵，72 小时硬上限取消，阶段 5 关闭。第 50 次独立分支快照诊断不作为启动依赖；既有队列将按协议导出保留快照的共同下游结果。
