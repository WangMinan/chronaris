# 命令入口

默认解释器为 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。脚本负责命令行编排，可复用实现放在 `src/chronaris`；长任务须保留进度和日志。

## 当前研究入口

`research/` 下的论文命令调用 `src/chronaris/evaluation/application_tasks/thesis_*`。当前配置和执行顺序以[任务页](../docs/implementation/TASKS.md)为准：

| 入口 | 职责 |
| --- | --- |
| `research/run_thesis_frozen_simulation.py` | 冻结仿真训练、表示、消融、压力与时间机制编排 |
| `research/run_thesis_simulation_gates.py` | 总机制审计，失败时返回非零退出码 |
| `research/run_thesis_native_outer.py` | 公开数据外层评价，当前保持关闭 |
| `research/run_thesis_dingxin_outer.py` | 鼎新分组外层确认，当前保持关闭 |
| `research/run_thesis_runtime.py`、`research/run_thesis_case_evidence.py`、`research/run_thesis_final_reporting.py` | 运行评估、案例与最终材料 |

v3.2.3 只批准剩余仿真与严格审计，外层命令仍保留 v3.2.2 的冻结接口，不因合并或清理自动升级、开放。复现旧仿真须使用证据标签 `evidence/thesis-v3p2p3-repair-20260904`，不能将清理后的源码写回旧运行状态。

## 其他职责目录

- `simulation/`：仿真观测生成与审计。
- `evaluation/application_tasks/`：通用任务评价、表示与历史冻结实验。
- `feature_export/`：标准化融合特征导出。
- `modeling/`：早期骨干与多任务训练。
- `evaluation/dingxin/`、`evaluation/public_datasets/`：鼎新弱监督评价与公开数据适配。
- `evidence/`：组件诊断、报告与图表材料。
- `runtime/`：重放、推理及接口校验。
- `llm_preprocessing/`：语言模型辅助字段归一、规则复核与解释材料。
- `archive/legacy_public_benchmark/`：历史公开基准。

历史入口使用其对应的输入合同与证据，不因仍可运行就视为当前论文模型。各脚本中的仓库根路径须指向当前检出目录；环境变量优先，缺少的数据库设置再读取本地配置。
