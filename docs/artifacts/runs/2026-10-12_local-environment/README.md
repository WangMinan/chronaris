# 本机开发环境初始化与迁移验收

日期：2026-10-12（北京时间）。范围：按用户要求初始化本机，仅补齐当前实验必需资产，全部训练队列保持关闭。

**本机已具备开发、数据读取和显卡推理能力，冻结资产完整恢复。** 27,150 个迁移文件共 54,246,646,298 字节，5,210 项原始冻结散列全部一致；主模型清单另外核验了 2,986 份来源、279 个单元、477 条路线、477 个模型文件和 1,614 个下游文件。迁移包的两个分卷也都通过 SHA-256（文件完整性散列）校验。

**跨设备数值一致性已分项记录，原验收阈值保持不变。** CLARE 生理负荷数据、CogPilot 飞行任务数据和鼎新数据各抽取四个内部验证窗口，自监督与任务引导共六条线性下游路线全部通过 `atol=rtol=1e-5` 的原判据。两个公开数据集的四条 MiniROCKET 时序卷积特征路线未通过这一严格阈值；抽查分类标签均未变，最大概率差约 0.000445，最大回归预测差约 0.000992，后者是不同任务原输出尺度上的最大值，不是归一化成绩。完整分项数值见 [replay-summary.json](replay-summary.json)。

**固定旧机表示的对照支撑下游依赖恢复正确。** 将旧机保存的 CLARE 自监督表示直接交给本机线性与 MiniROCKET 下游，预测差均为零；同一组样本在本机重算的序列表示最大绝对差为 `4.76837158203125e-7`。这把该抽查差异定位到表示重算一侧，而非模型文件损坏或该下游依赖不兼容；尚未区分显卡硬件、驱动与计算内核选择的贡献，也不扩展为全部方法的等价结论。见 [reference-check.json](reference-check.json)。后续正式恢复前应专项处理表示重算对阈值型特征的影响，本轮不调整算法或容差。

## 软件与设备

下表记录本机实际导入版本，用于重建环境和解释跨设备验证结果。

| 项目 | 验证结果 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5，WSL2（Windows 的 Linux 子系统） |
| 中央处理器 | AMD Ryzen 9 9900X，24 个逻辑处理器 |
| 内存 | WSL 当前可见约 30 GiB |
| 图形处理器 | NVIDIA GeForce RTX 4070 SUPER，12,282 MiB 显存 |
| Python | 3.11.15 |
| PyTorch 张量计算库 | 2.11.0+cu130 |
| CUDA 显卡计算运行时 | 13.0；Windows 驱动 617.42 |
| NumPy 数组库 / SciPy 科学计算库 | 2.2.6 / 1.15.3 |
| pandas 表格库 / scikit-learn 机器学习库 | 2.3.3 / 1.7.2 |
| torchdiffeq 常微分方程库 / aeon 时序学习库 | 0.2.5 / 1.5.0 |
| transformers 预训练模型库 | 4.57.6 |

环境目录为 `/home/wangminan/env/anaconda3/envs/chronaris`，已配置交互式 `conda activate chronaris`，不自动进入基础环境。`PYTHONNOUSERSITE=1` 防止用户目录依赖覆盖；编辑器本地配置指向同一解释器。精确依赖和重建命令见[环境配置](../../../../configs/environments/README.md)。旧 YAML 环境模板不用于本机更新。

Ubuntu 软件源已由阿里云切回官方 `archive.ubuntu.com`，原配置备份为 `/etc/apt/sources.list.chronaris-20261012.bak`。编译工具、Zstandard 压缩工具和 Noto 中文字体已安装。原有 MySQL 软件源的过期密钥已按[官方软件源](https://repo.mysql.com/)更新并备份，最后一次 `apt-get update` 无签名错误；没有安装或迁移数据库服务。

## 文件与路径

**数据实际保存在 WSL 的 Linux 文件系统中，冻结记录中的原路径保持有效。** `/home/wangminan/chronaris-data/chronaris-v4-results` 通过绑定挂载映射到 `/mnt/e/chronaris-v4-results`；已写入 `/etc/fstab` 并验证当前挂载来自 Linux 文件系统，原配置有备份。本轮未重启 WSL，重启后的自动挂载应以 `findmnt /mnt/e/chronaris-v4-results` 复核。

- 主仓库保持 `main`，本次验收使用的算法源码提交为 `bb417051d54b971d4b6f380f18bafec50f2be51a`；只新增环境配置、验收脚本和文档。
- 当前准备数据位于 `/home/wangminan/projects/chronaris/artifacts`。清单内 30 个源码位置的提交全部匹配，其中包含主仓库及 29 个分离头指针工作区；28 个实验运行工作区的 `artifacts` 链接按旧机原样恢复，冻结源码无修改。
- TimeCMA 时间序列与语言融合方法、SensorLLM 传感器与语言模型方法及 Chronos 时序预训练模型的上游源码已恢复到 `/home/wangminan/projects/chronaris-stage3-upstream`，版本见 [upstream-source.json](upstream-source.json)。其中 Chronos 以可编辑方式安装。
- 旧机未提交改动保存在 `/home/wangminan/chronaris-migration-20261012/server-uncommitted.patch` 和 `server-untracked/`，没有覆盖当前算法源码。这是读取时点的副本，不保证涵盖旧机后续继续产生的修改。
- 本机迁移日志、完整主模型审计、逐预测诊断与测试输出保存在 `/home/wangminan/chronaris-migration-20261012`。凭证文件仅保存在 Git 忽略的 `docs/SECRETS.md`，权限为 `0600`。

用户明确选择仅补必需资产，因此未补传约 89 GiB 原始数据、约 19 GiB 通用预训练权重，也未重建 MySQL / InfluxDB 数据库。已经开始的一小部分原始文件移入上述本机证据目录的 `partial-raw-dataset/`，不能作为完整数据集使用。当前准备数据支持已验证的实验路线；重新预处理原始数据或执行需要外部权重的对照模型时，需要再补相应资产。

## 实际验证

以下证据共同支撑本机开发和只读推理可用，同时保留跨设备一致性未通过项。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 分卷与迁移清单 | 两卷校验通过；27,150 文件及 5,210 冻结散列通过 | [恢复核验](restored-verification.json) |
| 主模型来源与计数 | 2,986 份来源通过；279 单元、477 路线完整 | [主模型清单](main-asset-summary.json) |
| 包依赖 | `python -m pip check` 通过 | 本机初始化日志 |
| 基础显卡运行 | 张量乘法、编译前向、常微分方程求解通过 | [运行检查](runtime-check.json) |
| 数据与入口回归 | 29 项通过 | [测试记录](pytest-environment.xml) |
| 仿真准备数据读取 | 原始异步双流样本读取通过 | [读取记录](simulation-read.json) |
| 冻结预测重放 | 六条线性路线通过；四条时序卷积特征路线记录差异 | [重放摘要](replay-summary.json) |
| 固定表示下游对照 | 两种下游均与原预测完全一致 | [固定表示对照](reference-check.json) |

所有检查均没有优化器更新、模型再拟合或训练队列启动，也没有打开外层留出评价。仓库完整测试中包含实际训练用例，本轮只执行下列不训练的测试集合。

```bash
conda activate chronaris
python -m pip check
python -m pytest -q tests/dataset tests/representation/test_loaders.py \
  tests/representation/test_contracts.py tests/test_dataset_pipeline.py \
  tests/test_sortie_validation.py tests/test_local_store_settings.py \
  tests/test_repository_entrypoints.py
python docs/artifacts/runs/2026-10-12_local-environment/verify_migration.py
python docs/artifacts/runs/2026-10-12_local-environment/runtime_check.py
python docs/artifacts/runs/2026-10-12_local-environment/replay_check.py
python docs/artifacts/runs/2026-10-12_local-environment/reference_check.py
```

这些验收脚本针对已恢复的本机路径，复用仓库现有加载、推理和比较接口，不提供训练入口。`replay_check.py` 会逐项保留严格比较的布尔结果；必须检查 `strict_replay_passed`，不能把脚本执行结束视为全部一致性通过。已有表示诊断缓存会被读取；要强制重新执行显卡编码，应先将本机 `replay-diagnostic` 输出目录重命名保留，再运行重放检查。参考表示文件仅用于固定输入的下游对照。

本轮未执行训练显存峰值、训练速度或长时间稳定性测试，因此不能据此确认原 RTX 4090 的所有训练批量都可直接用于 12 GiB 显卡。
