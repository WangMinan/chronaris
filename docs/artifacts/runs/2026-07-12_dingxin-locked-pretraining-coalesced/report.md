# 鼎新选定配置锁定重训

状态：completed；验收 8/8。
完成 75 个方法—折—随机种子训练；任务目标和 outer-test 始终关闭。
六种方法共用 0.1 秒固定因果时间箱；每箱时间戳取最后一次真实观测，任何输入都不会前移。
深度基线设备为 cuda，Chronaris 设备为 cuda；逐方法实测耗时与设备历史见 locked_pretraining_results.csv。
Chronaris 方法专属损失参与反向传播，早停只读取公共自监督 validation 损失。
