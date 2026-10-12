# Chronaris 环境配置

当前本机开发使用 **2026-10-12 的 WSL2（Windows 的 Linux 子系统）环境快照**，与旧开发机的实际计算库版本对齐。历史 YAML（结构化配置格式）模板仍供旧阶段追溯，不用于更新当前环境。

## 当前本机环境

解释器固定为 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。交互式终端执行：

```bash
conda activate chronaris
python -m pip check
```

已配置 `PYTHONNOUSERSITE=1`，避免用户目录中的包覆盖环境依赖；编辑器的本地解释器设置在忽略文件 `.vscode/settings.json` 中。

核心版本为 Python 3.11.15、PyTorch 张量计算库 2.11.0、CUDA 显卡计算运行时 13.0、NumPy 数组库 2.2.6、scikit-learn 机器学习库 1.7.2、torchdiffeq 常微分方程库 0.2.5。完整版本分别锁定在：

- `chronaris-wsl-20261012-conda.txt`：Linux x86-64 的 Conda 基础环境。
- `chronaris-wsl-20261012-requirements.txt`：Python 包管理器 pip 的精确版本。

在新机器重建时执行以下命令；已经初始化的本机无需重复创建。

```bash
conda create -n chronaris --file configs/environments/chronaris-wsl-20261012-conda.txt
conda activate chronaris
conda env config vars set PYTHONNOUSERSITE=1
python -m pip install -r configs/environments/chronaris-wsl-20261012-requirements.txt
python -m pip install -e .
python -m pip install -e /home/wangminan/projects/chronaris-stage3-upstream/chronos-forecasting
python -m pip check
```

最后一个安装命令依赖已经恢复的 Chronos 时序预训练模型源码，其冻结提交为 `4dbf163c2734c089cdf7da2b86fde48862ff9c6f`；其他上游源码版本见[本地验收记录](../../docs/artifacts/runs/2026-10-12_local-environment/README.md)。系统还需编译工具、Zstandard 压缩工具和 Noto 中文字体；本机已安装。显卡驱动由 Windows 宿主机提供。

本次验收覆盖数据读取、模型加载、只读预测与基础显卡运算，没有训练更新。六条线性下游路线重放通过，四条 MiniROCKET 时序卷积特征路线存在跨设备数值差异，原阈值保持不变；详见上述验收记录。

## 历史模板

以下模板保留早期阶段的依赖组合：

- `chronaris-stage-e-cpu.yml`、`chronaris-stage-e-gpu.yml`：早期对齐原型。
- `chronaris-task-eval-cpu.yml`、`chronaris-task-eval-gpu.yml`：早期任务评价。

其中显卡模板的 PyTorch 2.5.1 / CUDA 12.1 不代表当前实际环境，不应对当前 `chronaris` 环境执行这些模板的 `--prune` 更新。数据库凭证继续保存在 Git 忽略的 `docs/SECRETS.md`，不写入环境锁定文件。
