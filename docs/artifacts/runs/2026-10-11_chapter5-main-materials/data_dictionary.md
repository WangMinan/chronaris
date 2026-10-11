# 数据字段与使用说明

本包以 UTF-8 编码的逗号分隔值（CSV）保存数据。嵌套对象和数组用 JSON 文本保存在单元格中；空单元格表示原值缺失或该方法不适用。逐预测文件仅做无损压缩，解压后仍为 CSV。下列字段均为稳定机器键，图中使用的中文名称在 `style.json` 集中配置。

## 标识、角色和粒度

- `domain`：数据域，`clare` 为生理负荷数据，`cogpilot` 为虚拟飞行数据，`dingxin` 为鼎新实测数据。
- `method`、`method_name`：方法稳定标识；后者保留原训练记录字段。中文映射见样式文件。
- `unit_index`：冻结主训练矩阵内从零开始的单元索引；`fold_index` 为从零开始的划分索引；`seed` 为随机种子。
- `route`：实际表示路线；`self_supervised`、`task_guided`、`summary_adapted`、`frozen`、`history_adapted` 分别为自监督、任务引导、任务标签适配、外部预训练冻结和历史观测适配。
- `consumer`：下游算法族，`linear` 为共同线性下游，`minirocket` 为时序卷积特征补充。
- `task`：任务标识，中文名称见样式；`field` 为该任务内从零开始的输出字段索引，`field_name` 为原字段名。
- `data_role`：本包统一为 `inner_validation`，表示内部验证角色；训练曲线文件中的此字段表示所属材料批次，曲线本身同时保留训练与内部验证损失。
- `sample_id`：窗口/视图样本标识；`group_id` 为公开受试者或鼎新记录标识；`vehicle_context_id` 为鼎新共享航电时段标识。样本标识包含时间和视图来源，不应在图中直接作为中文标签。
- `status`：来源记录的完成、失败或指标不可用状态；`row_index` 为原训练历史列表内行号，从零开始。

设置指标唯一键为数据域、方法、折、种子、路线、下游、任务与指标；分组指标在此基础上增加分组和字段。逐预测数据使用单元、路线、下游、样本、任务与字段唯一定位。不能把同一窗口的不同字段或多视图视作独立受试者。

## 预测与分组指标

- `target_valid`：目标是否有效；无效目标仍保留预测和样本，目标值为空。
- `target`、`prediction`：目标与模型预测；分类为声明类别编码，回归保留冻结目标的数值单位。
- `probability`：按声明类别顺序排列的分类概率数组，回归为空。
- `sample_weight`：鼎新共享视图权重或公开样本默认权重；`no_observation`：是否没有有效历史观测。
- `support`：该组该字段实际有效评价数量；鼎新机动任务已合并共享航电视图，可能小于原窗口数。`weight_sum` 为有效权重总和。
- `macro_f1`：声明类别上的宏平均 F1；`balanced_accuracy` 为来源实现中的类别平均召回；`confusion_matrix` 的行是真实类别、列是预测类别，数值为计数。
- `rmse`、`mae`：均方根误差与平均绝对误差，鼎新使用训练尺度归一化数值；公开任务为原生数值。
- `target_scale`：训练角色拟合的目标尺度；`native_rmse` 为恢复原生单位后的均方根误差。无归一化时尺度为 1。
- `p95_absolute_error`：有效样本绝对误差的第 95 百分位；`top_five_squared_error_fraction` 为最大五项加权平方误差占总量比例；`top_five_windows` 保留对应标识与误差贡献。
- `spearman`：斯皮尔曼秩相关；常量或有效数不足时为空。
- `no_observation_fraction`：无观测比例；按来源任务规则计算，不能据此删除困难样本。
- `persistence_support`：持久性参照可配对的有效数；`persistence_rmse` 为同尺度参照误差；`skill_vs_persistence` 为一减配对平方误差之比，参照平方误差为零时未定义。

逐预测回归数值本身没有除以 `target_scale`。分组和设置指标的误差尺度需结合 `target_scale`、`native_rmse`、`error_units` 使用。鼎新不同生理字段具有不同原生单位；本包不把其原生误差直接平均。

## 设置汇总与误差条

- `metric`、`value`：指标名称与该设置值；`group_count` 为有有效指标的组数。
- `aggregation`：原汇总顺序，通常先字段、后受试者或记录；`error_units` 为 `native`（原生）或 `training_scale`（训练尺度归一化）。分类忽略此误差单位字段。
- `settings`：应保留的折与种子设置数；`available_settings`：其中指标非空的设置数。
- `mean`、`std`、`minimum`、`maximum`：非空设置值的等权均值、总体标准差、最小值和最大值。
- `error_type`：误差条类型，本包为设置间描述性总体标准差；不是置信区间。图中直接读取 `mean` 与 `std`，绘图时不重新统计。

## 训练曲线与模型来源

- `checkpoint_path`、`checkpoint_sha256`：模型来源路径与 SHA-256 内容散列。路径只供开发机追溯，本地绘图不读取模型。
- `feature_dim`：实际表示维数；`kind` 为逐时刻序列或窗口末端向量；`supervision_stratum` 描述监督路线与实际标签使用；`external_pretraining` 为外部预训练来源对象。
- `optimizer_updates`：累计实际优化器更新；`joint_updates`：联合微调更新；`update`：近期适配方法的更新编号；`best_update`：所选模型对应更新编号，未记录时为空。
- `history_rows`：原训练记录行数；`epoch` 为原训练器阶段轮次字段，不应代替实际更新数比较不同方法预算。
- `train_losses`、`validation_losses`、`mechanism_validation`：按原目标名称保存的损失/机制诊断对象；`public_selection_loss` 为训练器实际选模的验证目标值。
- `train_*_loss`、`validation_*_loss`：相应任务的训练与内部验证损失；任务中文映射见样式。`validation_selection_loss` 为微调选模目标，`loss` 为近期适配单步总损失。
- `improved`、`epochs_without_improvement`：原训练器记录的改善标记与连续未改善计数，不是方法间胜负。
- `mean_gradient_norm_before_clip`、`gradient_norm_before_clip_last_step`、`gradient_norm`：原训练器对应粒度的梯度范数；`task_counts` 为单步任务监督有效数；`parameters_with_gradient` 为收到梯度的参数数量。
- `candidate_id`：历史配置标识，仅用于机器追溯，不作为图题或图例。

训练曲线中的不同方法损失定义和归一化不同，不跨方法比较原始损失高低。需要新增曲线时，可直接从开放数据选择同一方法和同一损失字段绘制，不必重训或读取检查点。

## 时间与成本

- `current_seconds`、`historical_seconds`：本轮及继承的记录秒数；`accounted_seconds` 为两者之和。继承链不重复相加。
- `phase_seconds`：阶段名称到秒数的对象；`cost_scope` 标明只覆盖已记录单元成本，未归属的失败尝试另存。
- `origin_root`、`attempt_file`：尝试成本原始来源；`stage` 为执行阶段标识；`pid` 为当时进程号，不能用于判断当前运行。
- `exit_code`、`status`、`seconds`：尝试退出码、状态、墙钟秒数；`accounting` 提醒成功尝试不能重复加入完成单元成本。
- `units`：方法在该数据域的单元数；`mean_hours`、`total_hours`、`std_hours`：按单元成本换算的平均、总和及总体标准差，单位均为小时。

历史原始尝试如果没有保存完整计时，本包保留这一信息缺口，不编造额外消耗。性能横比需结合实际任务、训练规模与预训练条件使用。
