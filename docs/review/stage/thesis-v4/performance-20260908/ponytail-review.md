# 冗余复核

以下两处已按用户要求实际清理；行号指向清理后的复用入口。

- `src/chronaris/modeling/training/candidate_validation.py:L5`: shrink: 重复实现批次读取与样本分批。复用机制训练模块已有函数。
- `src/chronaris/evaluation/application_tasks/application_consumer_runtime.py:L63`: shrink: 两套 CPU 消费者拟合、恢复和保存分支。共用一个可独立执行的准备函数。
- 本轮未入库的通用候选排名草稿：yagni: 尚未连接实际结果读取，且对各域缺失时长的要求不合适。移除草稿，选型仍按既定计划接入真实结果合同；不计入 Git 净变化。

前两项对应文件合计净减少 18 行；其他文件新增的压力与并行入口单独验收，不计作冗余删除。

net: -18 lines possible.
