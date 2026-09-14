# 公开数据的时序下游补充

MiniROCKET 时序卷积特征算法只接收真实序列。下表保留全部公开数据补充成绩，不能用它与仅有窗口末端表示的近期方法形成不对称主排名。

以下为自监督路线的公开数据补充结果。

| 数据 | 方法 | 任务 | 指标 | 数值 |
| --- | --- | --- | --- | --- |
| CLARE 生理数据 | Chronaris 连续融合 | 负荷分类 | macro_f1 | 0.5135 |
| CLARE 生理数据 | Chronaris 连续融合 | 负荷评分回归 | rmse | 2.9209 |
| CLARE 生理数据 | 生理单流 | 负荷分类 | macro_f1 | 0.3969 |
| CLARE 生理数据 | 生理单流 | 负荷评分回归 | rmse | 1.9686 |
| CLARE 生理数据 | 航电／第二输入流单流 | 负荷分类 | macro_f1 | 0.3484 |
| CLARE 生理数据 | 航电／第二输入流单流 | 负荷评分回归 | rmse | 2.5374 |
| CLARE 生理数据 | 朴素时间同步 | 负荷分类 | macro_f1 | 0.3820 |
| CLARE 生理数据 | 朴素时间同步 | 负荷评分回归 | rmse | 3.3214 |
| CLARE 生理数据 | MulT 多模态变换器 | 负荷分类 | macro_f1 | 0.4901 |
| CLARE 生理数据 | MulT 多模态变换器 | 负荷评分回归 | rmse | 2.2958 |
| CLARE 生理数据 | ContiFormer 连续时间变换器 | 负荷分类 | macro_f1 | 0.3610 |
| CLARE 生理数据 | ContiFormer 连续时间变换器 | 负荷评分回归 | rmse | 2.1084 |
| CogPilot 虚拟飞行数据 | Chronaris 连续融合 | 飞行难度分类 | macro_f1 | 0.4183 |
| CogPilot 虚拟飞行数据 | Chronaris 连续融合 | 离线事件条件响应 | rmse | 0.4284 |
| CogPilot 虚拟飞行数据 | 生理单流 | 飞行难度分类 | macro_f1 | 0.1622 |
| CogPilot 虚拟飞行数据 | 生理单流 | 离线事件条件响应 | rmse | 0.6558 |
| CogPilot 虚拟飞行数据 | 航电／第二输入流单流 | 飞行难度分类 | macro_f1 | 0.5743 |
| CogPilot 虚拟飞行数据 | 航电／第二输入流单流 | 离线事件条件响应 | rmse | 0.4024 |
| CogPilot 虚拟飞行数据 | 朴素时间同步 | 飞行难度分类 | macro_f1 | 0.5120 |
| CogPilot 虚拟飞行数据 | 朴素时间同步 | 离线事件条件响应 | rmse | 0.3856 |
| CogPilot 虚拟飞行数据 | MulT 多模态变换器 | 飞行难度分类 | macro_f1 | 0.4816 |
| CogPilot 虚拟飞行数据 | MulT 多模态变换器 | 离线事件条件响应 | rmse | 0.3620 |
| CogPilot 虚拟飞行数据 | ContiFormer 连续时间变换器 | 飞行难度分类 | macro_f1 | 0.4467 |
| CogPilot 虚拟飞行数据 | ContiFormer 连续时间变换器 | 离线事件条件响应 | rmse | 0.3950 |


以下为任务引导路线的公开数据补充结果。

| 数据 | 方法 | 任务 | 指标 | 数值 |
| --- | --- | --- | --- | --- |
| CLARE 生理数据 | Chronaris 连续融合 | 负荷分类 | macro_f1 | 0.5095 |
| CLARE 生理数据 | Chronaris 连续融合 | 负荷评分回归 | rmse | 2.9184 |
| CLARE 生理数据 | 生理单流 | 负荷分类 | macro_f1 | 0.5038 |
| CLARE 生理数据 | 生理单流 | 负荷评分回归 | rmse | 2.3403 |
| CLARE 生理数据 | 航电／第二输入流单流 | 负荷分类 | macro_f1 | 0.3733 |
| CLARE 生理数据 | 航电／第二输入流单流 | 负荷评分回归 | rmse | 4.3735 |
| CLARE 生理数据 | MulT 多模态变换器 | 负荷分类 | macro_f1 | 0.4512 |
| CLARE 生理数据 | MulT 多模态变换器 | 负荷评分回归 | rmse | 2.1772 |
| CLARE 生理数据 | ContiFormer 连续时间变换器 | 负荷分类 | macro_f1 | 0.3421 |
| CLARE 生理数据 | ContiFormer 连续时间变换器 | 负荷评分回归 | rmse | 5.8397 |
| CogPilot 虚拟飞行数据 | Chronaris 连续融合 | 飞行难度分类 | macro_f1 | 0.4091 |
| CogPilot 虚拟飞行数据 | Chronaris 连续融合 | 离线事件条件响应 | rmse | 0.3940 |
| CogPilot 虚拟飞行数据 | 生理单流 | 飞行难度分类 | macro_f1 | 0.1512 |
| CogPilot 虚拟飞行数据 | 生理单流 | 离线事件条件响应 | rmse | 0.5064 |
| CogPilot 虚拟飞行数据 | 航电／第二输入流单流 | 飞行难度分类 | macro_f1 | 0.5668 |
| CogPilot 虚拟飞行数据 | 航电／第二输入流单流 | 离线事件条件响应 | rmse | 0.4051 |
| CogPilot 虚拟飞行数据 | MulT 多模态变换器 | 飞行难度分类 | macro_f1 | 0.4687 |
| CogPilot 虚拟飞行数据 | MulT 多模态变换器 | 离线事件条件响应 | rmse | 0.3774 |
| CogPilot 虚拟飞行数据 | ContiFormer 连续时间变换器 | 飞行难度分类 | macro_f1 | 0.4586 |
| CogPilot 虚拟飞行数据 | ContiFormer 连续时间变换器 | 离线事件条件响应 | rmse | 0.4336 |
