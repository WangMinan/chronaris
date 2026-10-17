# 系统架构与设备使用

**系统以同一观测与特征合同承接CPU和GPU。** 归档或标准文件接入后形成观测包；模型目录按字段兼容性供选择；任务管理保存请求、窗口和执行身份；推理适配器依次完成训练归一化、设备搬运、前向和标准输出；浏览器查询与下游读取使用保存结果。原模块输入输出详见[基础架构说明](../2026-10-15_chapter6-ready/architecture.md)，本轮增加显式设备层而保持方法核心和标准格式。

下表说明本轮增加或强化的职责，供6.2架构图和6.3实现说明使用。

| 模块 | 输入与输出 | 关键行为 |
| --- | --- | --- |
| 启动与资源协调 | --device cpu/cuda → 设备身份 | GPU使用既有跨实验互斥锁；占用或不可用时明确拒绝，CPU须显式选择 |
| 设备适配 | 原生观测 → 同合同标准特征 | 固定训练归一化；张量与模型同设备；前向后移回CPU标准导出 |
| 同步计时 | 模型加载、输入搬运、前向、输出 → 阶段计时 | GPU阶段前后同步；记录参数、输入及前向输出的实际设备 |
| 任务与片段身份 | 窗口、模型、源码、设备和精度 → 缓存键 | CPU/GPU互不复用片段；恢复保留原失败记录 |
| 结果与界面 | 保存任务 → 设备标识、曲线和下载 | 服务、任务、结果设备分别显示；历史GPU结果保持原身份 |
| 模型目录 | 保存配置 → 兼容案例与机制信息 | 完整仿真模型对应原生7＋12字段；实测模型对应12＋965字段，拒绝交叉套用 |

服务入口为 scripts/runtime/run_fusion_workbench.py，完整验证入口为 scripts/runtime/verify_fusion_gpu.py。设备适配在 src/chronaris/serving/fusion_workbench/device.py，原任务、传输、字段复核和标准表示模块继续复用。

恢复登记资产后可启动GPU服务；CPU运行只需将设备参数改为cpu。

    PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/run_fusion_workbench.py --data-root artifacts/chapter6-full-gpu/service --device cuda --port 8768

浏览器打开 http://127.0.0.1:8768。选择“仿真数据·完整方法案例”或“实测数据集·人机观测记录”，选择兼容模型与窗口，处理后进入结果分析及特征交付。字段页展示已有定义及复核状态，人工确认由使用者完成。服务保持单任务，窗口增量不等于跨窗口连续隐状态递推。

完整验收使用新目录执行，复用现成权重，**不启动新训练**。该命令依次核验物理来源和实测字段，运行同配置CPU/GPU测量、数值对照、两类浏览器案例及真实GPU恢复，最后释放服务及显卡锁。

    PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/verify_fusion_gpu.py --root artifacts/chapter6-full-gpu-recheck/service --evidence artifacts/chapter6-full-gpu-recheck/evidence

接口继续沿用原记录、模型、任务、结果和下载路径，新增 GET /api/runtime 返回服务设备。结果中的 execution_batches 记录参数、输入张量和前向输出设备；GPU任务失败会保留原请求及错误，不自动切换CPU。运行期复用的CUDA计算图来自保存配置，首轮构建开销计入前向时间；执行警告及原始日志随资产包保留。

本轮GPU验收已在空闲窗口完成并释放锁。后续若第五章队列持锁，GPU服务会拒绝启动；可待队列释放后运行，或明确以CPU启动查看既有结果，历史GPU任务身份保持不变。

