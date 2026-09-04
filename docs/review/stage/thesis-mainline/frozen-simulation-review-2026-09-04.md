# 冻结仿真实跑与安全门复核

复核日期：2026-09-04。总体判断：结构消融产物完整，压力导出因无观测样本合同冲突中止；总硬门尚未获得证明，公开数据和鼎新外层评价继续关闭。

后续确认：用户已明确批准 [v3.2.3](../../../requirements/thesis-frozen-paper-evaluation-v3.2.3.md)，仅修复无观测表示和严格安全门；不调模型、不放宽阈值、不开放外层。下文保留确认前的发现过程和全部原始结论，后续执行以新协议为准。

## 复核范围与方法

本次复核先检查训练进程、编排状态和训练检查点，再从原始观测掩码定位导出异常，最后独立重算干净场景与结构消融的三随机种子指标。仅检查既有冻结配置，不选择新模型、不调整权重、不读取公开数据或鼎新外层结果。

- 源码提交：`7937334239e2cdc0263754de03328769ab90987c`。
- 编排状态：`docs/artifacts/runs/2026-09-03_thesis-simulation-v3p2p1/run_state.json`。
- 原训练进程 `22688` 已退出；日志末尾为 `RepresentationContractError`，不是仍在训练或等待。
- 十二个结构消融 `last.pt` 均为 `epoch=50`、`best_epoch=50`、`training_status=completed`；检查点未重写。

## 已完成的证据

四项结构消融与三个随机种子的十二个组合全部完成训练，训练验收 6/6；三角色表示共 36 份，验收 5/5；消费者产生 768 条指标和 72 条配对统计，验收 6/6。干净场景消费者保留 1152 条指标，验收 7/7。上述验收证明产物完整性，不等价于机制门或应用优势成立。

指标来源与文件哈希如下，原始数值不覆盖：

- 干净场景：`docs/artifacts/runs/2026-09-03_thesis-simulation-consumers-v3p2p1/metric_long.csv`，SHA-256 为 `55e1437074433399055b302180898f4fad2432ccb96c80f59f3c712ee7b5e77b`。
- 结构消融：`docs/artifacts/runs/2026-09-03_thesis-simulation-ablation-consumers-v3p2p1/metric_long.csv`，SHA-256 为 `7199890f66469de11218d069ca6ca5c94227fed01863a93c8365062514ee29bf`。

## 无观测窗口导出异常

异常由设计内的连续缺失触发，不是已经证明的数据损坏。压力表示已完成首个随机种子的前 25 个场景，在第 26 个场景 `contiguous_gap_30s` 的生理单流导出时退出。对 192 个固定窗口进行原始观测因果查询复核，生理全窗无观测为 1 个（0.52%），航电全窗无观测为 0 个，双流同时无观测为 0 个。

受影响样本为 `g2_event_spline__locked_test_profile_007__trajectory_046__seed_300046__contiguous_gap_30s::context_030.000`。额外检查低信噪比场景 `observation_snr_05db` 和混合严重扰动场景 `mixed_severe`，各 192 个样本均未出现全窗无观测。未据此声称其他全部场景已经复核。

根因位于共享表示合同：`TrainedFusionAdapter` 保留编码器的真实有效查询掩码，且空掩码池化采用分母下限 1；`FusionStreamBatch.__post_init__` 却拒绝任何零有效点样本。把某个点改成有效会伪造观测支持，删除该样本会改变冻结比较总体，均不采用。

待确认的最小修订：允许无观测样本保持全假掩码、零序列和零池化表示；保留原样本与 `[B,96,64]` 形状，继续使用已冻结消费者，不重拟合；输出无观测覆盖率及其单独诊断。必须测试非空样本数值不变、空样本严格为零、导出恢复一致和样本不丢失。该修订涉及输出可用性语义，尚未实施，也未创建新的协议版本。

观测侧复核可通过以下既有入口重现，不读取任务目标：

```python
from chronaris.evaluation.application_tasks.simulation_stress_context_data import load_simulation_stress_context_data
from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream

data = load_simulation_stress_context_data(
    "artifacts/application_evaluation/2026-07-12_aviation-simulation-locked-stress",
    scenario_id="contiguous_gap_30s",
)
for stream in ("physiology", "vehicle"):
    mask = causal_query_stream(data.batch, stream_name=stream).modality_mask
    print(stream, int((~mask.any(dim=1)).sum()), len(data.batch.sample_ids))
```

## 三随机种子的应用与消融分析

现有结果支持“旁路在部分消费者上保护信息，应用增量依赖消费者”，不支持所有配置全面领先。下表逐个随机种子与该随机种子的最佳单流配对，再计算中位增量；分类使用宏平均 F1（macro-F1）差，回归使用均方根误差（RMSE）的“单流误差减 Chronaris 误差”，因此正值均表示 Chronaris 更好。随机种子固定为 17、29、43，只包含受控仿真留出集。

| 消费者 | 任务与指标 | 相对最佳单流的配对增量中位数 | 正增量种子数 |
| --- | --- | ---: | ---: |
| 固定线性消费者 | 未来负荷分类，宏平均 F1 | −0.02126 | 1/3 |
| 固定线性消费者 | 未来负荷回归，均方根误差 | +0.01903 | 3/3 |
| MiniROCKET 时序卷积特征消费者 | 未来负荷分类，宏平均 F1 | +0.03196 | 2/3 |
| MiniROCKET 时序卷积特征消费者 | 未来负荷回归，均方根误差 | −0.00631 | 1/3 |

相对移除单流旁路，完整模型在线性分类、线性回归、时序卷积特征分类上的配对增量中位数分别为 +0.08102、+0.06619、+0.06246，同方向种子数分别为 2/3、3/3、3/3；时序卷积特征回归为 −0.00204，只有 1/3 种子为正。不能把后一项不利结果省略，也不能用应用任务代替连续演化或物理残差机制诊断。

## 安全门判定缺口

当前未提交的 `thesis_simulation_gates.py::_bypass_gate` 使用每个任务—消费者组至少 2/3 种子达标的规则。冻结 v3 原协议只规定分类劣化不超过 0.05、回归均方根误差劣化不超过 5%，未规定该门可容忍一个种子失败；不得把其他机制门的 2/3 规则自动挪用到安全门。

已重算的超阈值单元如下。这些结果保留为负面证据，不因检查点完整或其他种子良好而删除。

| 消费者 | 指标 | 随机种子 | 相对最佳单流的劣化 | 冻结阈值 |
| --- | --- | ---: | ---: | ---: |
| 固定线性消费者 | 宏平均 F1 | 29 | 0.10969 | 0.05 |
| MiniROCKET 时序卷积特征消费者 | 宏平均 F1 | 43 | 0.05563 | 0.05 |
| MiniROCKET 时序卷积特征消费者 | 均方根误差 | 43 | 20.98% | 5% |

当前结论是安全门未获得冻结协议下的完整证明。不能在看到结果后自行选择更宽松的聚合口径，也不能据此继续模型调参。待人工确认后，修正审计规则、完成剩余仿真分析；在确认前不生成“全部硬门通过”结论、不启动外层评价。

## CPU 全量测试补充

同一测试会话完成 CPU 全量测试：473 项中 458 项通过、15 项跳过、0 失败、0 错误，用时 130.29 秒。命令为 `CUDA_VISIBLE_DEVICES='' /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q --junitxml=artifacts/application_evaluation/2026-09-04_frozen-simulation-review/cpu-pytest.xml`。七项因本次显式关闭 CUDA 而跳过，另外八项为未启用的现场数据或运行环境测试；这不是 CUDA 验证，也不代表全部研究硬门通过。

测试结果 XML 位于上述被忽略产物目录，SHA-256 为 `3def6703a346657578025e78197b5e4ec0bd6d643b45954793eb53e1f9ddcc80`；`git diff --check` 通过。本次还核验两个外层 CLI：未提供 `all_hard_gates_passed=true` 的总审计结果时均拒绝启动。现有安全门测试未覆盖单个种子超阈值的边界，因此测试通过不能消除该审计缺口。

## 下一步与时间边界

本次没有修改模型、表示合同、消费者或冻结协议，只保存复核事实。等待确认的事项是无观测样本语义及安全门审计口径，不是训练是否还活着。即使恢复压力导出成功，仍需完成剩余压力场景、时间机制、总硬门和外层实验，不能承诺一个小时内完成完整论文任务。
