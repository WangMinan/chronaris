# 冻结运行源码按需恢复

**本轮按用户确认，将 34 个闲置 runtime 检出改为 Git 归档与按需恢复。** 每个原版本均已包含在远端 main 的历史中，源码由固定提交重建；额外保存完整文件 SHA-256 校验清单、执行权限、符号链接及唯一一份未提交修改补丁。模型、数据、正式结果和历史实验清单保持原状。

**归档先完成恢复验证和远端备份，再移除原检出。** `inventory.json` 保存提交、原路径和资产链接；`checksums/` 为每个工作区的压缩文件清单；`patches/` 保存设备兼容工作区原有的未提交修改。`verification.json` 记录在临时目录逐个重新检出、应用补丁、恢复链接并逐文件校验的结果。最终清理状态见 `cleanup.json`。

## 需要复跑历史实验时

清理后，历史记录中的 runtime 绝对路径暂时不存在。旧队列的严格来源核验仍会要求这些文件；先恢复对应源码，再执行历史入口。不要修改冻结清单或放宽来源校验。

从主仓根目录恢复某个版本：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python docs/maintenance/2026-10-18-runtime-archive/restore.py chronaris-runtime-simulation-contract-resume-20261017
```

历史队列可能同时绑定多个版本。需要完整恢复原有运行环境的源码检出时：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python docs/maintenance/2026-10-18-runtime-archive/restore.py --all
```

恢复脚本只使用 Python 标准库和 Git；默认恢复到原路径，拒绝覆盖已有目录，恢复后核验全部登记文件及工作区修改状态。模型和数据必须仍位于清单记录的位置；脚本不会复制、训练或改写它们。若部分版本已经恢复，请指定尚未恢复的目录名；`--all` 不覆盖现有目录。需要再次独立检验恢复，可用 `--root /tmp/新的验证目录` 指定另一个父目录。

## 归档目录清单

下表对应本次移除的 34 个检出。完整提交及各项校验值见 `inventory.json`，因此以后不需要根据目录日期猜测版本。

| 原目录（位于 ~/projects/） | 固定提交 | 未提交修改 |
| --- | --- | --- |
| chronaris-runtime-causality-policy-20261015 | `767379ef6a44` | 无 |
| chronaris-runtime-chapter5-20261011 | `cfd12a83b717` | 无 |
| chronaris-runtime-chapter5-reload-20261011 | `8b198dc61210` | 无 |
| chronaris-runtime-chapter5-writing-20261015 | `d97c5c522378` | 无 |
| chronaris-runtime-chronos-reference-20261016 | `6f873ccad80a` | 无 |
| chronaris-runtime-device-compatible-20261012 | `8b198dc61210` | 有，随包保存 |
| chronaris-runtime-diagnostics-20261017 | `1e9151ea58a0` | 无 |
| chronaris-runtime-local-validation-20261012 | `4c391e87f827` | 无 |
| chronaris-runtime-power-recovery-20261013 | `2ae4a70bced3` | 无 |
| chronaris-runtime-semantic-20261012 | `35cf42ced4ef` | 无 |
| chronaris-runtime-sensor-reference-20261016 | `293ae6b277f9` | 无 |
| chronaris-runtime-simulation-contract-resume-20261017 | `14577462758a` | 无 |
| chronaris-runtime-stage4-20260911 | `9b4548d98fe0` | 无 |
| chronaris-runtime-stage4-graph-20260913 | `2b7db555f72b` | 无 |
| chronaris-runtime-stage4-recovery-20260912 | `475fc7b4d270` | 无 |
| chronaris-runtime-stage45b-20260921 | `0150c397717b` | 无 |
| chronaris-runtime-stage45c-20260922 | `c3b8f50d9b24` | 无 |
| chronaris-runtime-stage5-20260923 | `dd16e6885c26` | 无 |
| chronaris-runtime-stage5-formal-20260928 | `0b8f0d14fd4d` | 无 |
| chronaris-runtime-stage5-performance-20261002 | `a80289e88c27` | 无 |
| chronaris-runtime-stage5-recovery-20260926 | `389cea665c92` | 无 |
| chronaris-runtime-stage5-recovery-scaled-20260926 | `63753427db9a` | 无 |
| chronaris-runtime-stage5-sensor-recovery-qualified-20260930 | `008f7065e804` | 无 |
| chronaris-runtime-target-guard-20261016 | `c75253f14c9d` | 无 |
| chronaris-runtime-v4-candidate-pressure-gpu-20260908 | `87afc941f983` | 无 |
| chronaris-runtime-v4-candidates-20260908 | `ed2e24e3629b` | 无 |
| chronaris-runtime-v4-code-complete-20260908 | `bee5112e76ec` | 无 |
| chronaris-runtime-v4-complete-confirmation-20260908 | `f9a0aada1deb` | 无 |
| chronaris-runtime-v4-confirmation-20260908 | `b763ac2e053b` | 无 |
| chronaris-runtime-v4-development-chain-20260908 | `dce2625e586a` | 无 |
| chronaris-runtime-v4-diagnostic-20260906 | `ef8d88eb1458` | 无 |
| chronaris-runtime-v4-dingxin-target-order-20260908 | `d23dc66fa9cc` | 无 |
| chronaris-runtime-v4-naive-development-20260908 | `847c9c4e4279` | 无 |
| chronaris-runtime-v4-public-preparation-20260908 | `b8b53d8100bb` | 无 |

`~/projects/` 中主开发仓、第六章工作区、含独有未提交代码的预算工作区，以及仍被模型资产引用的上游源码保留。上次已清理的七个无引用检出另见 [第一轮清理记录](../2026-10-18-local-cleanup/README.md)。
