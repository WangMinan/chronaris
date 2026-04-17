# Environment Configs

这里放阶段 E 相关的环境依赖文件。

当前提供：

1. `chronaris-stage-e-cpu.yml`
   适用于本地 Windows / 无 GPU 的开发环境
2. `chronaris-stage-e-gpu.yml`
   适用于远程 WSL Ubuntu + NVIDIA GPU 的训练环境

设计原则：

1. `numpy` 和 `pytorch` 优先用 `conda`
2. `torchdiffeq` 放在 `pip` 子段
3. 不把数据库密钥、token、代理地址或机器路径写进环境文件

说明：

1. `torchdiffeq` 是纯 Python 小包，放进 `pip` 子段可以避免 `conda-forge` 的额外求解成本
2. GPU 环境当前默认使用 `pytorch-cuda=12.1`
3. 如果远程机器驱动或 CUDA 运行时版本不一致，再按实际环境调整 `pytorch-cuda`

## 推荐命令

新建环境：

```powershell
conda env create -f configs/environments/chronaris-stage-e-cpu.yml
conda env create -f configs/environments/chronaris-stage-e-gpu.yml
```

更新已有环境：

```powershell
conda env update -n chronaris -f configs/environments/chronaris-stage-e-cpu.yml --prune
conda env update -n chronaris -f configs/environments/chronaris-stage-e-gpu.yml --prune
```

如果是远程 WSL Ubuntu GPU 环境，默认优先使用：

```powershell
conda env update -n chronaris -f configs/environments/chronaris-stage-e-gpu.yml --prune
```
