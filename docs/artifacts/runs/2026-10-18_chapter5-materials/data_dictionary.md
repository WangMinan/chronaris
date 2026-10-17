# 数据字典

| 文件 | 行粒度与用途 | 单位、统计口径 |
| --- | --- | --- |
| native_metrics.csv | 模型、折、路线、下游、任务的一条指标 | 分类为宏平均 F1；回归原单位或训练尺度，见 error_units |
| native_groups.csv | 受试者或鼎新时间块、任务、字段 | 保留有效量、混淆计数、持久性参照、尾部和不可用状态 |
| native_summary.csv | 各方法配置、路线、下游、任务 | mean 为折等权均值；std 为折间总体标准差，不是种子误差或置信区间 |
| native_predictions/*.csv.gz | 窗口、任务、字段 | 原目标、预测、有效性、概率、分组、共享视图权重 |
| simulation_groups.csv | 生成参数档案、方法、路线、条件、指标 | 参数档案为分组单位；不是把每个查询点当独立样本 |
| simulation_summary.csv | 方法、路线、条件、指标 | 参数档案均值及描述性总体标准差 |
| simulation_predictions/*.csv.gz | 一个窗口 | JSON 数组保存完整逐点真值与预测，其他列保留负荷目标和预测 |
| regression_tails.csv | 条件、下游、档案或全部窗口 | 均方根误差、95% 绝对误差分位、最大五窗口平方误差占比、无观测覆盖 |
| sensitivity.csv | 物理残差权重、路线、下游、开发指标 | 仅内部验证，不能并入留出主表 |
| timing_metrics.csv | 方法、路线、观测条件、三类时间目标 | 秒；固定岭回归探针，不是直接时延识别结果 |
| causality_queries.csv | 窗口、历史截止时刻、查询时刻、扰动 | 表示最大绝对差；容差 1e-5，掩码须严格相同 |
| inference_costs.csv | 方法、路线、重复计时 | 四窗口，秒，显存字节；CUDA 同步后计时 |
| costs.csv | 训练单元或推理路线 | 秒；历史继承成本单列，不把重启等待当计算时间 |
| historical_attempts.csv | 原执行命令收据 | 保留成功、失败和中断记录；不与成功成本直接累加 |
| device_replay.csv | 内部重放、路线、下游 | 保留原严格容差判定、类别变化和指标前后值 |
| training_contracts.csv.gz | 表示路线 | 完整声明、数据角色、训练预算、监督与外部预训练来源 |

`variant` 是机器配置索引，读者图表使用 `style.json` 中的中文名称。`role=held_out` 表示最终留出评价，`validation` 表示内部验证。统计汇总保留 `settings` 与 `available_settings`，缺失指标不填零。
