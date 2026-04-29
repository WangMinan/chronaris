# Environment Configs

这里放阶段 E / I 相关的环境依赖文件。

当前提供：

1. `chronaris-stage-e-cpu.yml`
   适用于阶段 E/F/G/H 的本地 Windows / 无 GPU 开发环境
2. `chronaris-stage-e-gpu.yml`
   适用于阶段 E/F/G/H 的远程 WSL Ubuntu + NVIDIA GPU 环境
3. `chronaris-stage-i-cpu.yml`
   适用于阶段 I 的 CPU 基线、公开数据准备与 sklearn 评测
4. `chronaris-stage-i-gpu.yml`
   适用于阶段 I 后续需要补充 GPU 依赖时的统一环境入口

设计原则：

1. `numpy`、`pytorch`、`pandas`、`pyarrow`、`scikit-learn` 优先用 `conda`
2. `torchdiffeq` 与 `mne` 放在 `pip` 子段
3. 不把数据库密钥、token、代理地址或机器路径写进环境文件

说明：

1. `torchdiffeq` 是纯 Python 小包，放进 `pip` 子段可以避免额外求解成本
2. `mne` 当前是阶段 I 公开 EEG/ECG 数据处理的预留依赖，首轮 UAB baseline 本身不强依赖它
3. GPU 环境当前默认使用 `pytorch-cuda=12.1`
4. 如果远程机器驱动或 CUDA 运行时版本不一致，再按实际环境调整 `pytorch-cuda`

## 推荐命令

新建环境：

```powershell
conda env create -f configs/environments/chronaris-stage-e-cpu.yml
conda env create -f configs/environments/chronaris-stage-e-gpu.yml
conda env create -f configs/environments/chronaris-stage-i-cpu.yml
conda env create -f configs/environments/chronaris-stage-i-gpu.yml
```

更新已有环境：

```powershell
conda env update -n chronaris -f configs/environments/chronaris-stage-e-cpu.yml --prune
conda env update -n chronaris -f configs/environments/chronaris-stage-e-gpu.yml --prune
conda env update -n chronaris -f configs/environments/chronaris-stage-i-cpu.yml --prune
conda env update -n chronaris -f configs/environments/chronaris-stage-i-gpu.yml --prune
```

如果是远程 WSL Ubuntu GPU 环境，默认优先使用：

```powershell
conda env update -n chronaris -f configs/environments/chronaris-stage-e-gpu.yml --prune
```

如果是阶段 I 的公开数据准备与 CPU baseline，默认优先使用：

```powershell
conda env update -n chronaris -f configs/environments/chronaris-stage-i-cpu.yml --prune
```
